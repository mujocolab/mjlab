from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.time_step_total = self.joint_pos.shape[0]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._body_pos_w[:, self._body_indexes]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._body_quat_w[:, self._body_indexes]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._body_lin_vel_w[:, self._body_indexes]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.asset_name]
    self.robot_ref_body_index = self.robot.body_names.index(self.cfg.reference_body)
    self.motion_ref_body_index = self.cfg.body_names.index(self.cfg.reference_body)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionLoader(
      self.cfg.motion_file, self.body_indexes, device=self.device
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.metrics["error_ref_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_ref_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_ref_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_ref_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

    self._model_viz: mujoco.MjModel = copy.deepcopy(env.sim.mj_model)
    self._model_viz.geom_rgba[:, 1] = np.clip(
      self._model_viz.geom_rgba[:, 1] * 1.5, 0.0, 1.0
    )
    self._data_viz = mujoco.MjData(self._model_viz)
    self._vopt = mujoco.MjvOption()
    self._vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    self._pert = mujoco.MjvPerturb()
    self._catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]

  @property
  def ref_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_ref_body_index]
      + self._env.scene.env_origins
    )

  @property
  def ref_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_ref_body_index]

  @property
  def ref_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_ref_body_index]

  @property
  def ref_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_ref_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_ref_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_ref_body_index]

  @property
  def robot_ref_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_ref_body_index]

  @property
  def robot_ref_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_ref_body_index]

  @property
  def robot_ref_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_ref_body_index]

  def _update_metrics(self):
    self.metrics["error_ref_pos"] = torch.norm(
      self.ref_pos_w - self.robot_ref_pos_w, dim=-1
    )
    self.metrics["error_ref_rot"] = quat_error_magnitude(
      self.ref_quat_w, self.robot_ref_quat_w
    )
    self.metrics["error_ref_lin_vel"] = torch.norm(
      self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1
    )
    self.metrics["error_ref_ang_vel"] = torch.norm(
      self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.start_from_beginning:
      phase = torch.zeros(len(env_ids), device=self.device)
    else:
      phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
    self.time_steps[env_ids] = (phase * (self.motion.time_step_total - 1)).long()
    self._random_state_initialization(env_ids)

  def _random_state_initialization(self, env_ids: torch.Tensor):
    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        quat_apply_inverse(root_ori[env_ids], root_ang_vel[env_ids]),
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self):
    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    ref_pos_w_repeat = self.ref_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
    ref_quat_w_repeat = self.ref_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_ref_pos_w_repeat = self.robot_ref_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_ref_quat_w_repeat = self.robot_ref_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = ref_pos_w_repeat - robot_ref_pos_w_repeat
    delta_pos_w[..., :2] = 0.0
    delta_ori_w = yaw_quat(
      quat_mul(robot_ref_quat_w_repeat, quat_inv(ref_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = (
      robot_ref_pos_w_repeat
      + delta_pos_w
      + quat_apply(delta_ori_w, self.body_pos_w - ref_pos_w_repeat)
    )

  def _debug_vis_impl(self, scn: mujoco.MjvScene) -> None:
    for i in range(self.num_envs):
      entity: Entity = self._env.scene[self.cfg.asset_name]
      indexing = entity.indexing

      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      free_joint_pos_adr = free_joint_q_adr[:3]
      free_joint_ori_adr = free_joint_q_adr[3:7]
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      self._data_viz.qpos[free_joint_pos_adr] = (
        self.body_pos_w[i, 0].cpu().numpy().copy()
      )
      self._data_viz.qpos[free_joint_ori_adr] = (
        self.body_quat_w[i, 0].cpu().numpy().copy()
      )
      self._data_viz.qpos[joint_q_adr] = self.joint_pos[i].cpu().numpy().copy()

      mujoco.mj_forward(self._model_viz, self._data_viz)
      mujoco.mjv_addGeoms(
        self._model_viz,
        self._data_viz,
        self._vopt,
        self._pert,
        self._catmask.value,
        scn,
      )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str
  reference_body: str
  body_names: list[str]
  asset_name: str
  class_type: type[CommandTerm] = MotionCommand
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  start_from_beginning: bool = False
