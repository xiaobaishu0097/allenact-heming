"""Defining imitation losses for actor critic type models."""

from collections import OrderedDict
from typing import Dict, cast, Optional, Union

import einops
import torch

import allenact.utils.spaces_utils as su
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import (
    Distr,
    CategoricalDistr,
    SequentialDistr,
    ConditionalDistr,
)
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.base_abstractions.sensor import AbstractExpertSensor


class HbSR(AbstractActorCriticLoss):
    """History-based State Regularization."""

    def __init__(self, end_action_id: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.end_action_id = end_action_id

    @staticmethod
    def group_loss(
        distribution: Union[CategoricalDistr, ConditionalDistr],
        expert_actions: torch.Tensor,
        expert_actions_masks: torch.Tensor,
    ):
        assert isinstance(distribution, CategoricalDistr) or (
            isinstance(distribution, ConditionalDistr)
            and isinstance(distribution.distr, CategoricalDistr)
        ), "This implementation only supports (groups of) `CategoricalDistr`"

        expert_successes = expert_actions_masks.sum()

        log_probs = distribution.log_prob(cast(torch.LongTensor, expert_actions))
        assert (
            log_probs.shape[: len(expert_actions_masks.shape)]
            == expert_actions_masks.shape
        )

        # Add dimensions to `expert_actions_masks` on the right to allow for masking
        # if necessary.
        len_diff = len(log_probs.shape) - len(expert_actions_masks.shape)
        assert len_diff >= 0
        expert_actions_masks = expert_actions_masks.view(
            *expert_actions_masks.shape, *((1,) * len_diff)
        )

        group_loss = -(expert_actions_masks * log_probs).sum() / torch.clamp(
            expert_successes, min=1
        )

        return group_loss, expert_successes

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[Distr],
        *args,
        **kwargs,
    ):
        memories = cast(torch.Tensor, batch["memories"])
        actions = cast(torch.Tensor, batch["actions"])

        losses = OrderedDict()
        total_loss = 0

        n_episodes = 0
        n_steps, n_samplers = memories.shape[:2]
        for i_sampler in range(n_samplers):
            sampler_memories = memories[:, i_sampler, ...]
            sampler_actions = actions[:, i_sampler]

            end_idx = (sampler_actions == self.end_action_id).nonzero(as_tuple=True)[0]
            for i_end in range(max(len(end_idx), 1)):
                if len(end_idx) == 0:
                    episode_start_id = 0
                    episode_end_id = n_steps
                else:
                    if i_end == 0:
                        episode_start_id = 0
                        episode_end_id = end_idx[i_end] + 1
                    elif i_end >= len(end_idx) - 1:
                        if end_idx[i_end - 1] + 1 == n_steps:
                            continue
                        episode_start_id = end_idx[i_end - 1] + 1
                        episode_end_id = n_steps
                    else:
                        episode_start_id = end_idx[i_end - 1] + 1
                        episode_end_id = end_idx[i_end] + 1

                if len(end_idx) > 0 and (
                    end_idx[i_end] == 0 or episode_end_id - episode_start_id < 3
                ):
                    continue

                sampler_episode_memories = einops.rearrange(
                    sampler_memories[episode_start_id:episode_end_id], "t 1 c -> t c"
                )
                assert sampler_episode_memories.shape[0] > 0, (
                    "Episode length must be greater than 0, "
                    f"but got {sampler_episode_memories.shape[0]}"
                )

                belief_state_corr_coef = torch.corrcoef(sampler_episode_memories)
                belief_state_corr_coef = torch.relu(
                    mask_diagonal_and_subdiagonal(belief_state_corr_coef)
                )
                total_loss += torch.sum(belief_state_corr_coef) / (
                    torch.count_nonzero(belief_state_corr_coef) + 1
                )
                n_episodes += 1

        total_loss /= n_episodes

        return (total_loss, {"expert_cross_entropy": total_loss.item(), **losses})


def mask_diagonal_and_subdiagonal(matrix):
    """
    Masks the diagonal and subdiagonal of a square matrix.

    Args:
    matrix (torch.Tensor): An n x n matrix.

    Returns:
    torch.Tensor: The masked matrix.
    """
    n = matrix.size(0)

    # Diagonal mask
    diagonal_mask = torch.eye(n, dtype=torch.bool, device=matrix.device)

    # Subdiagonal mask (shift the diagonal mask by one row)
    subdiagonal_mask = torch.roll(diagonal_mask, shifts=1, dims=0) | torch.roll(
        diagonal_mask, shifts=-1, dims=0
    )

    # Combine masks
    combined_mask = diagonal_mask | subdiagonal_mask

    # Invert mask for applying
    # mask_to_apply = ~combined_mask

    # Apply mask
    masked_matrix = matrix.masked_fill(combined_mask, 0)

    return masked_matrix
