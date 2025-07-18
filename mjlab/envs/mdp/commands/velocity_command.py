from __future__ import annotations

from mjlab.managers.command_manager import CommandTerm
import torch
from typing import (
  TYPE_CHECKING,
  Sequence,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv
  from mjlab.envs.mdp.commands.commands_config import UniformVelocityCommandCfg


class UniformVelocityCommand(CommandTerm):
  cfg: UniformVelocityCommandCfg

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
    super().__init__(cfg, env)

    # data.root_lin_vel_b
    # data.root_ang_vel_b
    # heading_w

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

  # @property
  # def _root_lin_vel_b(self) -> torch.Tensor:
  #   data = self._env.sim.data
  # root_body_mat_w = data.xmat[:, 1]  # (num_envs, 3, 3)
  # root_body_inv_mat_w = root_body_mat_w.transpose(-2, -1)  # (num_envs, 3, 3)
  # lin_vel_w = data.sensordata[]

  # @property
  # def _root_ang_vel_b(self) -> torch.Tensor:
  #   pass

  def _update_metrics(self):
    # # time for which the command was executed
    # max_command_time = self.cfg.resampling_time_range[1]
    # max_command_step = max_command_time / self._env.step_dt
    # # logs data
    # self.metrics["error_vel_xy"] += (
    #     torch.norm(self.vel_command_b[:, :2] - self._root_lin_vel_b[:, :2],
    #                dim=-1) / max_command_step
    # )
    # self.metrics["error_vel_yaw"] += (
    #     torch.abs(self.vel_command_b[:, 2] - self.root_ang_vel_b[:,
    #                                          2]) / max_command_step
    # )
    pass

  def _resample_command(self, env_ids: Sequence[int]):
    # # sample velocity commands
    # r = torch.empty(len(env_ids), device=self.device)
    # # -- linear velocity - x direction
    # self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    # # -- linear velocity - y direction
    # self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    # # -- ang vel yaw - rotation around z
    # self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
    # # heading target
    # if self.cfg.heading_command:
    #   self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
    #   # update heading envs
    #   self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
    # # update standing envs
    # self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    pass

  def _update_command(self):
    # if self.cfg.heading_command:
    #   env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
    #   # compute angular velocity
    #   heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
    #   self.vel_command_b[env_ids, 2] = torch.clip(
    #     self.cfg.heading_control_stiffness * heading_error,
    #     min=self.cfg.ranges.ang_vel_z[0],
    #     max=self.cfg.ranges.ang_vel_z[1],
    #   )
    # # Enforce standing (i.e., zero velocity command) for standing envs
    # standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    # self.vel_command_b[standing_env_ids, :] = 0.0
    pass
