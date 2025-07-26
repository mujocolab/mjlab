"""Useful methods for MPD terminations."""

import torch

from mjlab.envs.manager_based_rl_env import ManagerBasedRLEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensors.contact_sensor.contact_sensor import ContactSensor


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum."""
  return env.episode_length_buf >= env.max_episode_length


def illegal_contact(
  env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Terminate when the contact force on the sensor exceeds the force threshold."""
  contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
  net_contact_forces = contact_sensor.data.net_forces_w_history
  # check if any contact force exceeds the threshold
  return torch.any(
    torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[
      0
    ]
    > threshold,
    dim=1,
  )
