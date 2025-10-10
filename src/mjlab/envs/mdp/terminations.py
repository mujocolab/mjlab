"""Useful methods for MDP terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum."""
  return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Terminate when the asset's orientation exceeds the limit angle."""
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  return torch.acos(-projected_gravity[:, 2]).abs() > limit_angle


def root_height_below_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the asset's root height is below the minimum height."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] < minimum_height


class illegal_contacts:
  """Terminate when the asset's selected contact sensors touche the ground."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self.sensor_names = list(cfg.params.get("sensor_names", []))
    asset: Entity = env.scene[self.asset_cfg.name]

    if not self.sensor_names:
      raise ValueError(
        "illegal_contacts requires 'sensor_names' (list of sensor names)."
      )

    self.threshold = torch.as_tensor(
      cfg.params.get("threshold", 1.0), device=env.device, dtype=torch.float32
    )

    for sensor_name in self.sensor_names:
      if sensor_name not in asset.data.sensor_data:
        raise KeyError(
          f"illegal_contacts: sensor '{sensor_name}' not found in sensor_data"
        )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_names: list[str],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    threshold: float = 1.0,
  ) -> torch.Tensor:
    del threshold  # Unused.
    del sensor_names  # Unused
    del asset_cfg  # Unused
    asset: Entity = env.scene[self.asset_cfg.name]
    terminate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for sensor_name in self.sensor_names:
      contact_force = asset.data.sensor_data[sensor_name]
      terminate |= (contact_force.abs() > self.threshold).any(dim=1)
    return terminate
