from __future__ import annotations
from typing import Sequence, TYPE_CHECKING

from mjlab.managers.action_manager import ActionTerm
from mjlab.utils.string import filter_exp
import torch
from mjlab.utils.mujoco import is_position_actuator
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

    from ipdb import set_trace

    set_trace()
    joint_names = filter_exp(self.cfg.joint_names, self._asset.joint_names)
    # joint_ids = self._asset.

    # joint_ids, joint_names = self.
    # self._entity_cfg = SceneEntityCfg(name=cfg.asset_name, joint_names=cfg.joint_names)
    # self._entity_cfg.resolve(env.sim.mj_model)
    # self._joint_ids = self._entity_cfg.joint_ids
    # self._joint_names = self._entity_cfg.joint_names
    self._num_joints = len(joint_ids)

    self._actuator_ids = []
    for aid in self._entity_cfg.actuator_ids:
      if not is_position_actuator(env.sim.mj_model.actuator(aid)):
        raise ValueError
      self._actuator_ids.append(aid)
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
    # TODO: clip

  def reset(self, env_ids: Sequence[int] | None = None) -> None:
    self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
  def __init__(self, cfg: actions_config.JointPositionActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

  def apply_actions(self):
    self._env.sim.set_ctrl(self._processed_actions, ctrl_ids=self._actuator_ids)
