"""PPO variant with privileged supervision for depth-based ball perception."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO

from .model import DepthAuxCNNModel


class BallAuxiliaryPPO(PPO):
  """Add ball geometry and visibility losses to the Actor's PPO objective."""

  def __init__(
    self,
    *args: Any,
    auxiliary_target_group: str = "ball_aux_target",
    auxiliary_loss_coef: float = 1.0,
    auxiliary_position_coef: float = 1.0,
    auxiliary_visibility_coef: float = 0.2,
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, **kwargs)
    if self.rnd is not None or self.symmetry is not None:
      raise ValueError("BallAuxiliaryPPO V1 does not support RND or symmetry")
    if auxiliary_loss_coef < 0.0:
      raise ValueError("auxiliary_loss_coef must be non-negative")
    self.auxiliary_target_group = auxiliary_target_group
    self.auxiliary_loss_coef = auxiliary_loss_coef
    self.auxiliary_position_coef = auxiliary_position_coef
    self.auxiliary_visibility_coef = auxiliary_visibility_coef

  def _auxiliary_loss(
    self,
    actor: DepthAuxCNNModel,
    observations: Any,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = actor.auxiliary_prediction
    target = observations[self.auxiliary_target_group]
    visible = target[:, 3:4]

    position_error = F.smooth_l1_loss(
      prediction[:, :3], target[:, :3], reduction="none"
    ).mean(dim=1, keepdim=True)
    position_loss = (position_error * visible).sum() / visible.sum().clamp_min(1.0)
    visibility_loss = F.binary_cross_entropy_with_logits(prediction[:, 3:4], visible)
    total = (
      self.auxiliary_position_coef * position_loss
      + self.auxiliary_visibility_coef * visibility_loss
    )
    return total, position_loss, visibility_loss

  def update(self) -> dict[str, float]:
    """Run PPO updates jointly with the supervised perception objective."""
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise ValueError("BallAuxiliaryPPO V1 currently expects feed-forward models")

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_auxiliary_loss = 0.0
    mean_position_loss = 0.0
    mean_visibility_loss = 0.0
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )

    for batch in generator:
      assert batch.observations is not None
      assert batch.actions is not None
      assert batch.advantages is not None
      assert batch.old_actions_log_prob is not None
      assert batch.old_distribution_params is not None
      assert batch.values is not None
      assert batch.returns is not None

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )

      self.actor(batch.observations, stochastic_output=True)
      if not isinstance(self.actor, DepthAuxCNNModel):
        raise TypeError(f"Unexpected Actor type: {type(self.actor).__name__}")
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations)
      distribution_params = self.actor.output_distribution_params
      entropy = self.actor.output_entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, distribution_params
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            learning_rate = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(learning_rate, src=0)
            self.learning_rate = learning_rate.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))
      advantages = batch.advantages.squeeze(-1)
      surrogate = -advantages * ratio
      surrogate_clipped = -advantages * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      auxiliary_loss, position_loss, visibility_loss = self._auxiliary_loss(
        self.actor, batch.observations
      )
      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
        + self.auxiliary_loss_coef * auxiliary_loss
      )

      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_auxiliary_loss += auxiliary_loss.item()
      mean_position_loss += position_loss.item()
      mean_visibility_loss += visibility_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    return {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "ball_auxiliary": mean_auxiliary_loss / num_updates,
      "ball_position": mean_position_loss / num_updates,
      "ball_visibility": mean_visibility_loss / num_updates,
    }
