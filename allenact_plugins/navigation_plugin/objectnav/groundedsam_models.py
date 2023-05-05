from typing import Dict, List, Optional, Tuple, cast

import einops
import gym
import timm
import torch
import torch.nn as nn
from gym.spaces import Dict as SpaceDict

from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.embodiedai.models import resnet as resnet
from allenact.embodiedai.models.visual_nav_models import (FusionType,
                                                          VisualNavActorCritic)


def get_2d_positional_embedding(size_feature_map,
                                c_pos_embedding,
                                gpu_id=None) -> torch.Tensor:
    mask = torch.ones(1, size_feature_map, size_feature_map)

    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(c_pos_embedding, dtype=torch.float32)
    # dim_t = 10000 ** (2 * (dim_t // 2) / c_pos_embedding)
    dim_t = 10000**torch.div(2 * torch.div(dim_t, 2, rounding_mode='trunc'),
                             c_pos_embedding,
                             rounding_mode='trunc')

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
        dim=4).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
        dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos


class CustomViT(nn.Module):

    def __init__(self,
                 encoder_layers: Optional[int] = None,
                 mask_channels: Optional[int] = 2,
                 img_size: Optional[int] = 224,
                 embed_dim: Optional[int] = 512):
        super(CustomViT, self).__init__()
        self.vit = timm.create_model("vit_base_patch16_224",
                                     pretrained=False,
                                     img_size=img_size,
                                     in_chans=mask_channels,
                                     num_classes=embed_dim)

        if encoder_layers is not None:
            self.vit.blocks = self.vit.blocks[:encoder_layers]

        # Modify the ViT's head to produce the desired output feature shape
        self.vit.head = nn.Sequential(
            nn.Linear(self.vit.head.in_features, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim))

    def forward(self, x):
        return self.vit(x)


class ResNetGroundedSAMTensorGoalEncoder(nn.Module):

    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        resnet_preprocessor_uuid: str,
        grounded_sam_preprocessor_uuid: str,
        goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        vit_encoder_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.resnet_uuid = resnet_preprocessor_uuid
        self.grounded_sam_uuid = grounded_sam_preprocessor_uuid
        self.goal_embed_dims = goal_embed_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.goal_space = observation_spaces.spaces[self.goal_uuid]

        self.blind = self.resnet_uuid not in observation_spaces.spaces and self.grounded_sam_uuid not in observation_spaces.spaces
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[
                self.resnet_uuid].shape
            self.resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0],
                          self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )

            self.grounded_sam_tensor_shape = observation_spaces.spaces[
                self.grounded_sam_uuid].shape

            # a VIT encoder to process the (300, 300, 2) binary mask as input and output a feature with (512, 1, 1)
            self.vit_encoder = CustomViT(encoder_layers=vit_encoder_layers,
                                         mask_channels=2,
                                         embed_dim=392)

            self.target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + 8,
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
            return (self.combine_hid_out_dims[-1] *
                    self.grounded_sam_tensor_shape[1] * self.grounded_sam_tensor_shape[2])

    def generate_goal_proposal_masks(self, observations):
        # TODO: Implement this
        # return self.resnet_compressor(observations[self.resnet_uuid])
        raise NotImplementedError

    def compress_resnet(self, observations):
        return self.resnet_compressor(observations[self.resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_embed_dims, 1,
                               1).expand(-1, -1,
                                         self.grounded_sam_tensor_shape[-2],
                                         self.grounded_sam_tensor_shape[-1])

    def process_proposal_mask(self, observations: dict) -> torch.Tensor:
        raise NotImplementedError

    def adapt_input(self, observations):
        resnet = observations[self.resnet_uuid]
        semantic_mask = observations[self.grounded_sam_uuid]
        goal = observations[self.goal_uuid]

        use_agent = False
        nagent = 1

        if len(semantic_mask.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = semantic_mask.shape[:3]
        else:
            nstep, nsampler = semantic_mask.shape[:2]

        observations[self.resnet_uuid] = resnet.view(-1, *resnet.shape[-3:])
        observations[self.grounded_sam_uuid] = semantic_mask.view(
            -1, *semantic_mask.shape[-3:])
        observations[self.goal_uuid] = goal.view(-1, goal.shape[-1])

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations)

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])

        embeds = self.vit_encoder(observations[self.grounded_sam_uuid].to(
            torch.float32))

        x = self.target_obs_combiner(
            torch.cat((self.compress_resnet(observations),
                       einops.rearrange(
                           embeds, 'b (n h w) -> b n h w', n=8, h=7, w=7)),
                      dim=1))
        # embeds = einops.rearrange(embeds, 'b c -> b 1 c')

        # resnet_features = self.compress_resnet(observations)
        # resnet_features = einops.rearrange(resnet_features + self.global_pos_embedding.to(resnet_features.device), 'b c h w -> b c (h w)')
        # embs, _ = self.visual_transformer(self.compress_detr(observations), resnet_features)
        # x = self.target_obs_combiner(einops.rearrange(embs, 'b (h w) c -> b c h w', h=7, w=7))
        # x = x.reshape(x.size(0), -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class GroundedSAMTensorGoalEncoder(nn.Module):

    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        grounded_sam_preprocessor_uuid: str,
        goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        vit_encoder_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.grounded_sam_uuid = grounded_sam_preprocessor_uuid
        self.goal_embed_dims = goal_embed_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.goal_space = observation_spaces.spaces[self.goal_uuid]

        self.blind = self.grounded_sam_uuid not in observation_spaces.spaces
        if not self.blind:
            self.grounded_sam_tensor_shape = observation_spaces.spaces[
                self.grounded_sam_uuid].shape

            # a VIT encoder to process the (300, 300, 2) binary mask as input and output a feature with (512, 1, 1)
            self.vit_encoder = CustomViT(encoder_layers=vit_encoder_layers,
                                         mask_channels=2,
                                         embed_dim=512)

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            # return (self.combine_hid_out_dims[-1] *
            #         self.grounded_sam_tensor_shape[1] * self.grounded_sam_tensor_shape[2])
            return 512

    def generate_goal_proposal_masks(self, observations):
        # TODO: Implement this
        # return self.resnet_compressor(observations[self.resnet_uuid])
        raise NotImplementedError

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_embed_dims, 1,
                               1).expand(-1, -1,
                                         self.grounded_sam_tensor_shape[-2],
                                         self.grounded_sam_tensor_shape[-1])

    def process_proposal_mask(self, observations: dict) -> torch.Tensor:
        raise NotImplementedError

    def adapt_input(self, observations):
        semantic_mask = observations[self.grounded_sam_uuid]
        goal = observations[self.goal_uuid]

        use_agent = False
        nagent = 1

        if len(semantic_mask.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = semantic_mask.shape[:3]
        else:
            nstep, nsampler = semantic_mask.shape[:2]

        observations[self.grounded_sam_uuid] = semantic_mask.view(
            -1, *semantic_mask.shape[-3:])
        observations[self.goal_uuid] = goal.view(-1, goal.shape[-1])

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations)

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])

        embeds = self.vit_encoder(observations[self.grounded_sam_uuid].to(
            torch.float32))
        # embeds = einops.rearrange(embeds, 'b c -> b 1 c')

        # resnet_features = self.compress_resnet(observations)
        # resnet_features = einops.rearrange(resnet_features + self.global_pos_embedding.to(resnet_features.device), 'b c h w -> b c (h w)')
        # embs, _ = self.visual_transformer(self.compress_detr(observations), resnet_features)
        # x = self.target_obs_combiner(einops.rearrange(embs, 'b (h w) c -> b c h w', h=7, w=7))
        # x = x.reshape(x.size(0), -1)  # flatten

        return self.adapt_output(embeds, use_agent, nstep, nsampler, nagent)


class ResnetDualGroundedSAMTensorGoalEncoder(nn.Module):

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
            self.embed_goal = nn.Linear(self.goal_space.shape[-1],
                                        self.goal_embed_dims)
        else:
            raise NotImplementedError

        self.blind = (self.rgb_resnet_uuid not in observation_spaces.spaces or
                      self.depth_resnet_uuid not in observation_spaces.spaces)
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[
                self.rgb_resnet_uuid].shape
            self.rgb_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0],
                          self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.depth_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0],
                          self.resnet_hid_out_dims[0], 1),
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
            return (2 * self.combine_hid_out_dims[-1] *
                    self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2])

    def get_object_type_encoding(
            self, observations: Dict[str,
                                     torch.FloatTensor]) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_rgb_resnet(self, observations):
        return self.rgb_resnet_compressor(observations[self.rgb_resnet_uuid])

    def compress_depth_resnet(self, observations):
        return self.depth_resnet_compressor(
            observations[self.depth_resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_embed_dims, 1,
                               1).expand(-1, -1, self.resnet_tensor_shape[-2],
                                         self.resnet_tensor_shape[-1])

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
        observations[self.depth_resnet_uuid] = depth.view(
            -1, *depth.shape[-3:])
        observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations)

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        rgb_embs = [
            self.compress_rgb_resnet(observations),
            self.distribute_target(observations),
        ]
        rgb_x = self.rgb_target_obs_combiner(torch.cat(
            rgb_embs,
            dim=1,
        ))
        depth_embs = [
            self.compress_depth_resnet(observations),
            self.distribute_target(observations),
        ]
        depth_x = self.depth_target_obs_combiner(torch.cat(
            depth_embs,
            dim=1,
        ))
        x = torch.cat([rgb_x, depth_x], dim=1)
        x = x.reshape(x.shape[0], -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class GroundedSAMTensorNavActorCritic(VisualNavActorCritic):

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
        depth_resnet_preprocessor_uuid: Optional[str] = None,
        rgb_grounded_sam_preprocessor_uuid: Optional[str] = None,
        depth_grounded_sam_preprocessor_uuid: Optional[str] = None,
        goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        vit_encoder_layers: Optional[int] = None,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
        )

        if (rgb_grounded_sam_preprocessor_uuid is None
                or depth_grounded_sam_preprocessor_uuid is None
            ) and rgb_resnet_preprocessor_uuid is None:
            grounded_sam_preprocessor_uuid = (
                rgb_grounded_sam_preprocessor_uuid
                if rgb_grounded_sam_preprocessor_uuid is not None else
                depth_grounded_sam_preprocessor_uuid)
            self.goal_visual_encoder = GroundedSAMTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                grounded_sam_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
                vit_encoder_layers=vit_encoder_layers,
            )
        elif (rgb_resnet_preprocessor_uuid is not None
              and rgb_grounded_sam_preprocessor_uuid
              is not None) and depth_grounded_sam_preprocessor_uuid is None:
            resnet_preprocessor_uuid = (
                rgb_resnet_preprocessor_uuid if rgb_resnet_preprocessor_uuid
                is not None else depth_resnet_preprocessor_uuid)
            grounded_sam_preprocessor_uuid = (
                rgb_grounded_sam_preprocessor_uuid
                if rgb_grounded_sam_preprocessor_uuid is not None else
                depth_grounded_sam_preprocessor_uuid)
            self.goal_visual_encoder = ResNetGroundedSAMTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                resnet_preprocessor_uuid,
                grounded_sam_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
                vit_encoder_layers=vit_encoder_layers,
            )
        else:
            self.goal_visual_encoder = ResnetDualGroundedSAMTensorGoalEncoder(  # type:ignore
                self.observation_space,
                goal_sensor_uuid,
                rgb_grounded_sam_preprocessor_uuid,
                depth_grounded_sam_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
                device=device,
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

    def forward_encoder(self,
                        observations: ObservationType) -> torch.FloatTensor:
        return self.goal_visual_encoder(observations)
