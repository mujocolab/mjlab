"""Episode target commands for the Gym Reacher scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ReacherTargetCommand(CommandTerm):
  """Sample the goal; synchronize Gym's passive target marker joints on reset."""

  cfg: ReacherTargetCommandCfg

  def __init__(self, cfg: ReacherTargetCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.asset: Entity = env.scene[cfg.entity_name]
    joint_ids, _ = self.asset.find_joints(cfg.target_joint_names, preserve_order=True)
    self.joint_ids = torch.tensor(joint_ids, device=self.device)
    self.target_xy = torch.zeros(self.num_envs, 2, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_xy

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Equivalent to Gym's rejection sampling of a uniform point in this disk.
    radius = self.cfg.radius * torch.rand(len(env_ids), device=self.device).sqrt()
    angle = 2 * torch.pi * torch.rand(len(env_ids), device=self.device)
    target = torch.stack((radius * angle.cos(), radius * angle.sin()), dim=-1)
    self.target_xy[env_ids] = target
    # The marker is part of the original scene XML, not an arm joint. Its
    # passive slides encode world XY and have zero velocity for the episode.
    self.asset.write_joint_state_to_sim(
      target, torch.zeros_like(target), env_ids=env_ids, joint_ids=self.joint_ids
    )

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class ReacherTargetCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  target_joint_names: tuple[str, str] = ("target_x", "target_y")
  radius: float = 0.2

  def build(self, env: ManagerBasedRlEnv) -> ReacherTargetCommand:
    return ReacherTargetCommand(self, env)
