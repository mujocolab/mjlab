"""Useful methods for MPD terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum."""
  return env.episode_length_buf >= env.max_episode_length


# def illegal_contact(
#   env: ManagerBasedRlEnv, threshold: float, sensor_cfg: SceneEntityCfg
# ) -> torch.Tensor:
#   """Terminate when the contact force on the sensor exceeds the force threshold."""
#   contact_sensor = cast(ContactSensor, env.scene.sensors[sensor_cfg.name])
#   net_contact_forces = contact_sensor.data.net_forces_w_history

#   # Extract forces for specific body IDs.
#   body_forces = net_contact_forces[:, :, sensor_cfg.body_ids]

#   # Calculate force magnitudes and get maximum per timestep.
#   force_norms = torch.norm(body_forces, dim=-1)
#   max_forces_per_timestep = torch.max(force_norms, dim=1)[0]

#   # Check if any force exceeds threshold.
#   return torch.any(max_forces_per_timestep > threshold, dim=1)


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Terminate when the asset's orientation exceeds the limit angle."""
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  return torch.acos(-projected_gravity[:, 2]).abs() > limit_angle
