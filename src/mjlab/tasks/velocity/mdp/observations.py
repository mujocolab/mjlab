from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def projected_gravity_imu(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Projected gravity using an IMU up-vector sensor."""
  sensor = env.scene[sensor_name]
  return -sensor.data


def gait_clock(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Return [cos(2*pi*phase), sin(2*pi*phase)] for velocity gait conditioning."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  command_term = env.command_manager.get_term(command_name)
  freq_base = getattr(command_term.cfg, "gait_freq_base", 0.5)
  freq_speed_scale = getattr(command_term.cfg, "gait_freq_speed_scale", 0.0)
  speed = torch.norm(command[:, :2], dim=1)
  freq = freq_base + freq_speed_scale * speed
  phase = torch.remainder(env.episode_length_buf.float() * env.step_dt * freq, 1.0)
  angle = phase * (2.0 * torch.pi)
  return torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
