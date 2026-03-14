from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ROBOT_JOINT_CFG = SceneEntityCfg("robot", joint_names=(".*",))


def actuator_work_l2_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_JOINT_CFG,
  use_joint_acc: bool = False,
  max_abs_power: float | None = None,
  max_abs_torque: float | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force[:, asset_cfg.joint_ids]
  if max_abs_torque is not None:
    tau = torch.clamp(tau, min=-max_abs_torque, max=max_abs_torque)
  motion_term = (
    asset.data.joint_acc[:, asset_cfg.joint_ids]
    if use_joint_acc
    else asset.data.joint_vel[:, asset_cfg.joint_ids]
  )
  mech_power = torch.sum(tau * motion_term, dim=-1)
  if max_abs_power is not None:
    mech_power = torch.clamp(mech_power, min=-max_abs_power, max=max_abs_power)
  work = torch.square(mech_power)
  if normalize and max_abs_power is not None and max_abs_power > 0.0:
    work = work / (max_abs_power**2)
  return work


class object_yaw_finite_diff_clipped:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.object_name = cfg.params["object_name"]
    self.history_steps = int(cfg.params.get("history_steps", 1))
    if self.history_steps < 1:
      raise ValueError(f"history_steps must be >= 1, got {self.history_steps}")
    self.prev_yaw = torch.zeros(env.num_envs, device=env.device)
    self._env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    self._delta_hist = torch.zeros((env.num_envs, self.history_steps), device=env.device)
    self._hist_ptr = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._hist_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._env = env
    self.reset(None)

  def _current_yaw(self) -> torch.Tensor:
    obj: Entity = self._env.scene[self.object_name]
    _, _, yaw = euler_xyz_from_quat(obj.data.root_link_quat_w)
    return yaw

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    obj: Entity = self._env.scene[self.object_name]
    yaw = self._current_yaw()
    self.prev_yaw[env_ids] = yaw[env_ids]
    self._delta_hist[env_ids] = 0.0
    self._hist_ptr[env_ids] = 0
    self._hist_count[env_ids] = 0
    roll, pitch, _ = euler_xyz_from_quat(obj.data.root_link_quat_w)
    self._init_pos_w[env_ids] = obj.data.root_link_pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    clip_min: float = -0.25,
    clip_max: float = 0.25,
    history_steps: int | None = None,
    negate_yaw_rate: bool = True,
    drift_position_threshold: float = 0.02,
    drift_tilt_threshold: float = 0.35,
    drift_inside_factor: float = 1.0,
    drift_outside_factor: float = 0.1,
    drift_mode: str = "step",
    drift_decay_k_pos: float = 30.0,
    drift_decay_k_tilt: float = 3.0,
    drift_min_factor: float = 0.1,
  ) -> torch.Tensor:
    del object_name
    if history_steps is not None and history_steps != self.history_steps:
      raise ValueError(
        f"history_steps={history_steps} does not match initialized value {self.history_steps}."
      )

    yaw = self._current_yaw()
    reset_mask = env.episode_length_buf <= 1
    if torch.any(reset_mask):
      self.reset(reset_mask.nonzero(as_tuple=False).squeeze(-1))

    delta_yaw = wrap_to_pi(yaw - self.prev_yaw)
    self.prev_yaw[:] = yaw

    hist_ptr = self._hist_ptr
    self._delta_hist[self._env_ids, hist_ptr] = delta_yaw
    self._hist_ptr = (hist_ptr + 1) % self.history_steps
    self._hist_count = torch.clamp(self._hist_count + 1, max=self.history_steps)

    hist_len = torch.clamp(self._hist_count, min=1).to(delta_yaw.dtype)
    avg_delta_yaw = self._delta_hist.sum(dim=-1) / hist_len

    yaw_rate = avg_delta_yaw / max(env.step_dt, 1e-6)
    if negate_yaw_rate:
      yaw_rate = -yaw_rate
    yaw_reward = torch.clamp(yaw_rate, min=clip_min, max=clip_max)

    obj: Entity = self._env.scene[self.object_name]
    roll, pitch, _ = euler_xyz_from_quat(obj.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(obj.data.root_link_pos_w - self._init_pos_w, dim=-1)
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )

    if drift_mode == "step":
      in_bounds = (pos_error <= drift_position_threshold) & (
        tilt_error <= drift_tilt_threshold
      )
      drift_factor = torch.where(
        in_bounds,
        torch.full_like(yaw_reward, drift_inside_factor),
        torch.full_like(yaw_reward, drift_outside_factor),
      )
    elif drift_mode == "exp":
      pos_excess = torch.clamp(pos_error - drift_position_threshold, min=0.0)
      tilt_excess = torch.clamp(tilt_error - drift_tilt_threshold, min=0.0)
      drift_factor = torch.exp(
        -drift_decay_k_pos * pos_excess - drift_decay_k_tilt * tilt_excess
      )
      drift_factor = torch.clamp(drift_factor, min=drift_min_factor, max=1.0)
    else:
      raise ValueError(
        f"Unknown drift_mode '{drift_mode}'. Expected one of: 'step', 'exp'."
      )

    drift_factor = torch.where(self._has_init, drift_factor, torch.ones_like(drift_factor))
    return yaw_reward * drift_factor


def object_linvel_l1(
  env: ManagerBasedRlEnv,
  object_name: str,
  clip_max: float | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  value = torch.norm(obj.data.root_link_lin_vel_w, p=1, dim=-1)
  if clip_max is not None:
    value = torch.clamp(value, max=clip_max)
  if normalize and clip_max is not None and clip_max > 0.0:
    value = value / clip_max
  return value


def object_fallen(
  env: ManagerBasedRlEnv,
  object_name: str,
  minimum_height: float,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  return (obj.data.root_link_pos_w[:, 2] < minimum_height).float()


def joint_torque_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_JOINT_CFG,
  max_abs_torque: float | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force[:, asset_cfg.joint_ids]
  if max_abs_torque is not None:
    tau = torch.clamp(tau, min=-max_abs_torque, max=max_abs_torque)
  value = torch.sum(torch.square(tau), dim=-1)
  if normalize:
    if max_abs_torque is not None and max_abs_torque > 0.0:
      denom = tau.shape[1] * (max_abs_torque**2)
    else:
      denom = max(tau.shape[1], 1)
    value = value / denom
  return value


class pose_diff_l2_from_reset:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self.asset: Entity = env.scene[self.asset_cfg.name]
    num_joints = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids].shape[1]
    self.init_joint_pos = torch.zeros(
      (env.num_envs, num_joints),
      device=env.device,
      dtype=self.asset.data.joint_pos.dtype,
    )
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    self.init_joint_pos[env_ids] = self.asset.data.joint_pos[env_ids][
      :, self.asset_cfg.joint_ids
    ]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    average_per_joint: bool = True,
    joint_tolerance: float = 0.0,
    clip_max: float | None = None,
    normalize: bool = False,
  ) -> torch.Tensor:
    del env, asset_cfg, normalize
    q = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids]
    joint_diff = torch.abs(q - self.init_joint_pos)
    if joint_tolerance > 0.0:
      joint_diff = torch.clamp(joint_diff - joint_tolerance, min=0.0)
    value = torch.sum(torch.square(joint_diff), dim=-1)
    if average_per_joint:
      value = value / max(q.shape[1], 1)
    if clip_max is not None:
      value = torch.clamp(value, max=clip_max)
    return value
