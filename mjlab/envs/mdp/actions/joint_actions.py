from __future__ import annotations
from typing import Sequence, TYPE_CHECKING

from mjlab.managers.action_manager import ActionTerm
import torch
from mjlab.entities.robots.robot import Robot

if TYPE_CHECKING:
  from mjlab.envs.mdp.actions import actions_config
  from mjlab.envs.manager_based_env import ManagerBasedEnv


class JointAction(ActionTerm):
  """Base class for joint actions."""

  cfg: actions_config.JointActionCfg
  _asset: Robot

  def __init__(self, cfg: actions_config.JointActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

    # self._joint_ids, self._joint_names = self._asset.find_joints(
    #   self.cfg.joint_names, preserve_order=self.cfg.preserve_order
    # )
    self._actuator_ids, self._actuator_names = self._asset.find_actuators(
      self.cfg.actuator_names,
      preserve_order=self.cfg.preserve_order,
    )
    self._num_joints = len(self._actuator_ids)
    self._action_dim = len(self._actuator_ids)

    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)
    self._scale = cfg.scale
    self._offset = cfg.offset

  # Properties.

  @property
  def action_dim(self) -> int:
    return self._action_dim

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale + self._offset

  def reset(self, env_ids: Sequence[int] | None = None) -> None:
    self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
  def __init__(self, cfg: actions_config.JointPositionActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

    # TODO: Check that the actuators are PD actuators.

    if cfg.use_default_offset:
      self._offset = self._asset.data.default_joint_pos[:, self._actuator_ids].clone()

  def apply_actions(self):
    self._env.sim.set_ctrl(self._processed_actions, ctrl_ids=self._actuator_ids)
    # from ipdb import set_trace; set_trace()
    # self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
