from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensors import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def feet_air_time(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_cfg: SceneEntityCfg,
  threshold: float,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
  first_contact = contact_sensor.compute_first_contact(env.step_dt)[
    :, sensor_cfg.body_ids
  ]
  last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
  reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
  reward *= (
    torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
  )
  return reward
