"""Defining the PPO loss for actor critic type models."""

from typing import Dict, Optional, Callable, cast, Tuple

import torch

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput


class HBSR(AbstractActorCriticLoss):
    """Implementation of the Historical-based State Regularization loss.

    # Attributes

    clip_param : The clipping parameter to use.
    value_loss_coef : Weight of the value loss.
    entropy_coef : Weight of the entropy (encouraging) loss.
    use_clipped_value_loss : Whether or not to also clip the value loss.
    clip_decay : Callable for clip param decay factor (function of the current number of steps)
    entropy_method_name : Name of Distr's entropy method name. Default is `entropy`,
                          but we might use `conditional_entropy` for `SequentialDistr`
    show_ratios : If True, adds tracking for the PPO ratio (linear, clamped, and used) in each
                  epoch to be logged by the engine.
    normalize_advantage: Whether or not to use normalized advantage. Default is True.
    """

    def __init__(
        self,
        use_clipped_value_loss=True,
        clip_decay: Optional[Callable[[int], float]] = None,
        entropy_method_name: str = "entropy",
        normalize_advantage: bool = True,
        show_ratios: bool = False,
        *args,
        **kwargs
    ):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_decay = clip_decay if clip_decay is not None else (lambda x: 1.0)
        self.entropy_method_name = entropy_method_name
        self.show_ratios = show_ratios
        if normalize_advantage:
            self.adv_key = "norm_adv_targ"
        else:
            self.adv_key = "adv_targ"

    def mask_corrcoef_matrix(self, corrcoef_matrix: torch.Tensor) -> torch.Tensor:
        n_batch, n_dim, _ = corrcoef_matrix.shape

        for i in range(n_batch):
            corrcoef_matrix[i, torch.arange(n_dim-1), torch.arange(1, n_dim)] = 0
            corrcoef_matrix[i, torch.arange(n_dim), torch.arange(n_dim)] = 0

        return corrcoef_matrix

    def loss_per_step(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
    ) -> Dict[str, Tuple[torch.Tensor, Optional[float]]]:  # TODO tuple output

        belief_states = cast(torch.LongTensor, batch["belief_states"])
        belief_state_corrcoef = cast(
            torch.LongTensor, 
            torch.stack([
                torch.corrcoef(belief_states[:, i]) for i in range(belief_states.shape[1])
            ], dim=0)
        )

        belief_state_corrcoef = torch.stack([
            torch.triu(belief_state_corrcoef[i], diagonal=2) for i in range(belief_state_corrcoef.shape[0])
        ], dim=0)

        hsbr_loss = torch.sum(belief_state_corrcoef) / torch.count_nonzero(belief_state_corrcoef)

        # noinspection PyUnresolvedReferences
        return {
                "belief_states_regularization": (hsbr_loss, None),
        }

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        losses_per_step = self.loss_per_step(
            step_count=step_count, batch=batch, actor_critic_output=actor_critic_output,
        )
        losses = {
            key: (loss.mean(), weight)
            for (key, (loss, weight)) in losses_per_step.items()
        }

        total_loss = sum(
            loss * weight if weight is not None else loss
            for loss, weight in losses.values()
        )

        result = (
            total_loss,
            {
                "hsbr_total": cast(torch.Tensor, total_loss).item(),
                **{key: loss.item() for key, (loss, _) in losses.items()},
            },
            {"belief_states_regularization": total_loss.item(),}
            if self.show_ratios
            else {},
        )

        return result if self.show_ratios else result[:2]


HBSRConfig = dict()
