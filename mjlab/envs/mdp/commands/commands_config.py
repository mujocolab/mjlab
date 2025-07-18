from __future__ import annotations
from dataclasses import dataclass, MISSING

from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from mjlab.envs.mdp.commands.velocity_command import UniformVelocityCommand


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(CommandTermCfg):
  class_type: type[CommandTerm] = UniformVelocityCommand
  asset_name: str = MISSING
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float] = MISSING
    lin_vel_y: tuple[float, float] = MISSING
    ang_vel_z: tuple[float, float] = MISSING
    heading: tuple[float, float] | None = None

  ranges: Ranges = MISSING

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
