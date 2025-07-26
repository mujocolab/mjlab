from __future__ import annotations

from mjlab.managers.command_manager import CommandTerm
import torch
from typing import (
  TYPE_CHECKING,
  Sequence,
)
import mujoco
import numpy as np
from mjlab.entities.robots.robot import Robot
from mjlab.utils.math import wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv
  from mjlab.envs.mdp.commands.commands_config import UniformVelocityCommandCfg


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
    super().__init__(cfg, env)

    self.robot: Robot = env.scene.entities[cfg.asset_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
    return self.vel_command_b

  def _update_metrics(self):
    # time for which the command was executed
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    # logs data
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_com_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_com_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: Sequence[int]):
    # sample velocity commands
    r = torch.empty(len(env_ids), device=self.device)
    # -- linear velocity - x direction
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    # -- linear velocity - y direction
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    # -- ang vel yaw - rotation around z
    self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    # heading target
    if self.cfg.heading_command:
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      # update heading envs
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    # update standing envs
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

  def _update_command(self):
    if self.cfg.heading_command:
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      # compute angular velocity
      heading_error = wrap_to_pi(
        self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
      )
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * heading_error,
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    # Enforce standing (i.e., zero velocity command) for standing envs
    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0

  # Visualization.

  def _debug_vis_impl(self, scn):
    base_pos_w = self.robot.data.root_link_pos_w.clone()  # (num_envs, 3)
    base_quat_w = self.robot.data.root_link_quat_w.clone()
    base_mat_w = matrix_from_quat(base_quat_w)

    base_pos_ws = base_pos_w.cpu().numpy()
    base_mat_ws = base_mat_w.cpu().numpy()
    cmds = self.command.cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()

    for batch in range(self.num_envs):
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      def local_to_world(vec: np.ndarray) -> np.ndarray:
        return base_pos_w + base_mat_w @ vec

      def make_arrow(
        from_local: np.ndarray, to_local: np.ndarray
      ) -> tuple[np.ndarray, np.ndarray]:
        return local_to_world(from_local), local_to_world(to_local)

      def add_arrow(from_w, to_w, rgba, width=0.015, size=(0.005, 0.02, 0.02)):
        scn.ngeom += 1
        geom = scn.geoms[scn.ngeom - 1]
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR

        mujoco.mjv_initGeom(
          geom=geom,
          type=mujoco.mjtGeom.mjGEOM_ARROW.value,
          size=np.array(size, dtype=np.float32),
          pos=np.zeros(3),
          mat=np.zeros(9),
          rgba=np.asarray(rgba, dtype=np.float32),
        )

        mujoco.mjv_connector(
          geom=geom,
          type=mujoco.mjtGeom.mjGEOM_ARROW.value,
          width=width,
          from_=from_w,
          to=to_w,
        )

      scale = 0.75
      z_offset = 0.2
      cmd_lin_from = np.array([0, 0, z_offset]) * scale
      cmd_lin_to = cmd_lin_from + np.array([cmd[0], cmd[1], 0]) * scale
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = cmd_ang_from + np.array([0, 0, cmd[2]]) * scale
      add_arrow(*make_arrow(cmd_lin_from, cmd_lin_to), rgba=[0.2, 0.2, 0.6, 0.6])
      add_arrow(*make_arrow(cmd_ang_from, cmd_ang_to), rgba=[0.2, 0.6, 0.2, 0.6])

      # Actual velocities.
      act_lin_from = np.array([0, 0, z_offset]) * scale
      act_lin_to = act_lin_from + np.array([lin_vel_b[0], lin_vel_b[1], 0]) * scale
      act_ang_from = act_lin_from
      act_ang_to = act_ang_from + np.array([0, 0, ang_vel_b[2]]) * scale
      add_arrow(*make_arrow(act_lin_from, act_lin_to), rgba=[0.0, 0.6, 1.0, 0.7])
      add_arrow(*make_arrow(act_ang_from, act_ang_to), rgba=[0.0, 1.0, 0.4, 0.7])
