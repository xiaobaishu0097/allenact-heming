from typing import Any, Optional, List, Dict, cast, Tuple, Sequence

import einops
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Dict as SpaceDict

from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.embodiedai.models import resnet as resnet
from allenact.embodiedai.models.basic_models import SimpleCNN
from allenact.embodiedai.models.visual_nav_models import (
    VisualNavActorCritic,
    FusionType,
)

from torch.nn import (
    Module,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)


def get_2d_positional_embedding(
    size_feature_map, c_pos_embedding, gpu_id=None
) -> torch.Tensor:
    mask = torch.ones(1, size_feature_map, size_feature_map)

    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(c_pos_embedding, dtype=torch.float32)
    # dim_t = 10000 ** (2 * (dim_t // 2) / c_pos_embedding)
    dim_t = 10000 ** torch.div(
        2 * torch.div(dim_t, 2, rounding_mode="trunc"),
        c_pos_embedding,
        rounding_mode="trunc",
    )

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos


class SAMResnetTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        resnet_preprocessor_uuid: str,
        sam_preprocessor_uuid: str,
        goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.resnet_uuid = resnet_preprocessor_uuid
        self.sam_uuid = sam_preprocessor_uuid
        self.goal_embed_dims = goal_embed_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.goal_space = observation_spaces.spaces[self.goal_uuid]
        if isinstance(self.goal_space, gym.spaces.Discrete):
            self.embed_goal = nn.Embedding(
                num_embeddings=self.goal_space.n,
                embedding_dim=self.goal_embed_dims,
            )
        elif isinstance(self.goal_space, gym.spaces.Box):
            self.embed_goal = nn.Linear(self.goal_space.shape[-1], self.goal_embed_dims)
        else:
            raise NotImplementedError

        self.blind = self.resnet_uuid not in observation_spaces.spaces
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuid].shape
            self.resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(self.resnet_hid_out_dims[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
            )
            self.target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.combine_hid_out_dims[0],
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )
            self.sam_compressor = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 128)
            )

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=128, dim_feedforward=256, nhead=4
            )
            self.transformer = nn.TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=1
            )

            self.state_query = nn.Embedding(1, 128)

            self.global_pos_embedding = get_2d_positional_embedding(2, 64)

            self.resnet_embedding = nn.Embedding(1, 128)
            self.sam_segmentation_embedding = nn.Embedding(1, 128)
            self.target_embedding = nn.Embedding(1, 128)

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            return 128

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_resnet(self, observations):
        return self.resnet_compressor(observations[self.resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_embed_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def compress_sam(self, observations) -> Tuple[torch.Tensor, torch.Tensor]:
        sam_feature = observations[self.sam_uuid]

        segmentation_features = (
            self.sam_segmentation_embedding.weight
            + self.sam_compressor(sam_feature["segmentation_features"].float())
        )

        target_features = self.target_embedding.weight + self.sam_compressor(
            sam_feature["target_features"]
        )

        return segmentation_features, target_features

    def process_resnet_detr(self, resnet_feature, detr_feature) -> torch.Tensor:
        pass

    def adapt_input(self, observations):
        resnet = observations[self.resnet_uuid]
        sam = observations[self.sam_uuid]
        goal = observations[self.goal_uuid]

        use_agent = False
        nagent = 1

        if len(resnet.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = resnet.shape[:3]
        else:
            nstep, nsampler = resnet.shape[:2]

        observations[self.resnet_uuid] = resnet.view(-1, *resnet.shape[-3:])
        observations[self.sam_uuid] = {
            k: v.view(-1, *v.shape[-2:]) for k, v in sam.items()
        }
        observations[self.goal_uuid] = goal.view(-1, goal.shape[-1])

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        resnet_features = self.compress_resnet(observations)
        resnet_features = einops.rearrange(
            F.max_pool2d(resnet_features, (3, 3))
            + self.global_pos_embedding.to(resnet_features.device),
            "b c h w -> b (h w) c",
        )  # (batch_size, 7*7, 128)
        resnet_features = self.resnet_embedding.weight + resnet_features

        sam_segmentation_features, target_features = self.compress_sam(
            observations
        )  # (batch_size, 128, 100/10)

        state_representation = self.transformer(
            torch.repeat_interleave(
                einops.rearrange(self.state_query.weight, "1 c -> 1 1 c"),
                resnet_features.shape[0],
                1,
            ),
            einops.rearrange(
                torch.cat(
                    [resnet_features, sam_segmentation_features, target_features], dim=1
                ),
                "b n c -> n b c",
            ),
        )
        state_representation = einops.rearrange(state_representation, "n b c -> b n c")

        return self.adapt_output(
            state_representation, use_agent, nstep, nsampler, nagent
        )


class SAMResnetDualTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        rgb_resnet_preprocessor_uuid: str,
        depth_resnet_preprocessor_uuid: str,
        goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.rgb_resnet_uuid = rgb_resnet_preprocessor_uuid
        self.depth_resnet_uuid = depth_resnet_preprocessor_uuid
        self.goal_embed_dims = goal_embed_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.goal_space = observation_spaces.spaces[self.goal_uuid]
        if isinstance(self.goal_space, gym.spaces.Discrete):
            self.embed_goal = nn.Embedding(
                num_embeddings=self.goal_space.n,
                embedding_dim=self.goal_embed_dims,
            )
        elif isinstance(self.goal_space, gym.spaces.Box):
            self.embed_goal = nn.Linear(self.goal_space.shape[-1], self.goal_embed_dims)
        else:
            raise NotImplementedError

        self.blind = (
            self.rgb_resnet_uuid not in observation_spaces.spaces
            or self.depth_resnet_uuid not in observation_spaces.spaces
        )
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[
                self.rgb_resnet_uuid
            ].shape
            self.rgb_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.depth_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.rgb_target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_embed_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )
            self.depth_target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_embed_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            return (
                2
                * self.combine_hid_out_dims[-1]
                * self.resnet_tensor_shape[1]
                * self.resnet_tensor_shape[2]
            )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_rgb_resnet(self, observations):
        return self.rgb_resnet_compressor(observations[self.rgb_resnet_uuid])

    def compress_depth_resnet(self, observations):
        return self.depth_resnet_compressor(observations[self.depth_resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_embed_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def adapt_input(self, observations):
        rgb = observations[self.rgb_resnet_uuid]
        depth = observations[self.depth_resnet_uuid]

        use_agent = False
        nagent = 1

        if len(rgb.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = rgb.shape[:3]
        else:
            nstep, nsampler = rgb.shape[:2]

        observations[self.rgb_resnet_uuid] = rgb.view(-1, *rgb.shape[-3:])
        observations[self.depth_resnet_uuid] = depth.view(-1, *depth.shape[-3:])
        observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        rgb_embs = [
            self.compress_rgb_resnet(observations),
            self.distribute_target(observations),
        ]
        rgb_x = self.rgb_target_obs_combiner(
            torch.cat(
                rgb_embs,
                dim=1,
            )
        )
        depth_embs = [
            self.compress_depth_resnet(observations),
            self.distribute_target(observations),
        ]
        depth_x = self.depth_target_obs_combiner(
            torch.cat(
                depth_embs,
                dim=1,
            )
        )
        x = torch.cat([rgb_x, depth_x], dim=1)
        x = x.reshape(x.shape[0], -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class SAMResnetTensorNavActorCritic(VisualNavActorCritic):
    def __init__(
        # base params
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
        add_prev_actions=False,
        add_prev_action_null_token=False,
        action_embed_size=6,
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
        # custom params
        rgb_resnet_preprocessor_uuid: Optional[str] = None,
        rgb_sam_preprocessor_uuid: Optional[str] = None,
        depth_resnet_preprocessor_uuid: Optional[str] = None,
        goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        **kwargs: Any,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
        )

        if (
            rgb_resnet_preprocessor_uuid is None
            or depth_resnet_preprocessor_uuid is None
        ):
            resnet_preprocessor_uuid = (
                rgb_resnet_preprocessor_uuid
                if rgb_resnet_preprocessor_uuid is not None
                else depth_resnet_preprocessor_uuid
            )
            self.goal_visual_encoder = SAMResnetTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                resnet_preprocessor_uuid,
                rgb_sam_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )
        else:
            self.goal_visual_encoder = SAMResnetDualTensorGoalEncoder(  # type:ignore
                self.observation_space,
                goal_sensor_uuid,
                rgb_resnet_preprocessor_uuid,
                depth_resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        self.train()

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.goal_visual_encoder(observations)
