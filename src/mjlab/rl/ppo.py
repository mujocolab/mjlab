"""PPO variants for mjlab."""

from __future__ import annotations

from rsl_rl.algorithms import PPO


class CriticWarmupPPO(PPO):
  """PPO that freezes the actor for the first ``critic_warmup_updates`` updates.

  When RL fine-tuning a distilled policy, the critic starts from scratch and
  its early value estimates produce destructive policy gradients. Following
  arXiv:2505.11164, the policy is kept frozen until the critic has trained to
  a sufficient level. During warmup the actor's parameters (including the
  action std) receive no gradients; observation-normalizer statistics keep
  updating. The adaptive KL learning-rate schedule is inert during warmup
  (KL stays exactly zero), but a ``fixed`` schedule is recommended for
  conservative fine-tuning anyway.
  """

  def __init__(self, *args, critic_warmup_updates: int = 0, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.critic_warmup_updates = critic_warmup_updates
    self._num_updates = 0

  def update(self) -> dict[str, float]:
    warmup = self._num_updates < self.critic_warmup_updates
    if warmup:
      self._raw_actor.requires_grad_(False)
    try:
      loss_dict = super().update()
    finally:
      if warmup:
        self._raw_actor.requires_grad_(True)
    self._num_updates += 1
    loss_dict["critic_warmup"] = float(warmup)
    return loss_dict

  def save(self) -> dict:
    saved_dict = super().save()
    saved_dict["ppo_num_updates"] = self._num_updates
    return saved_dict

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    self._num_updates = loaded_dict.get("ppo_num_updates", self._num_updates)
    return super().load(loaded_dict, load_cfg, strict)
