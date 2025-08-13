"""Useful methods for MPD terminations."""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensors.contact_sensor.contact_sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.entities.robots.robot import Robot


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum."""
  return env.episode_length_buf >= env.max_episode_length


def illegal_contact(
  env: ManagerBasedRlEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Terminate when the contact force on the sensor exceeds the force threshold."""
  contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
  net_contact_forces = contact_sensor.data.net_forces_w_history
  return torch.any(
    torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[
      0
    ]
    > threshold,
    dim=1,
  )


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
  asset: Robot = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  return torch.acos(-projected_gravity[:, 2]).abs() > limit_angle
