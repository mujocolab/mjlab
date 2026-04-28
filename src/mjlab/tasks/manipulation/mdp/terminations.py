from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def object_out_of_reach(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg,
  threshold: float,
) -> torch.Tensor:
  """Terminate when the object's horizontal distance from the robot base exceeds *threshold*."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  obj_xy = obj.data.root_link_pos_w[:, :2]
  base_xy = robot.data.root_link_pos_w[:, :2]
  return torch.linalg.vector_norm(obj_xy - base_xy, dim=-1) > threshold


def object_spinning_too_fast(
  env: ManagerBasedRlEnv,
  object_name: str,
  threshold: float,
) -> torch.Tensor:
  """Terminate when object angular speed exceeds *threshold* rad/s.

  Catches the pre-NaN regime where penetration impulses launch the object
  with extreme spin. Legitimate manipulation stays under ~20 rad/s; runaway
  spin before solver divergence is hundreds of rad/s, so a 25-30 rad/s
  threshold catches divergence early without false-positives on dynamic
  grasp motion.
  """
  obj: Entity = env.scene[object_name]
  ang_vel = obj.data.root_link_ang_vel_w
  return torch.linalg.vector_norm(ang_vel, dim=-1) > threshold
