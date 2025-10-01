# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm
from mjlab.third_party.isaaclab.isaaclab.utils.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv
  from mjlab.envs.mdp.actions import actions_config


class JointAction(ActionTerm):
  """Base class for joint actions."""

  _asset: Entity

  def __init__(self, cfg: actions_config.JointActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

    actuator_ids, self._actuator_names = self._asset.find_actuators(
      cfg.actuator_names, preserve_order=cfg.preserve_order
    )
    self._actuator_ids = torch.tensor(
      actuator_ids, device=self.device, dtype=torch.long
    )

    self._num_joints = len(self._actuator_ids)
    self._action_dim = len(self._actuator_ids)

    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    if isinstance(cfg.scale, (float, int)):
      self._scale = float(cfg.scale)
    elif isinstance(cfg.scale, dict):
      self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.scale, self._actuator_names
      )
      self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError(
        f"Unsupported scale type: {type(cfg.scale)}."
        " Supported types are float and dict."
      )

    if isinstance(cfg.offset, (float, int)):
      self._offset = float(cfg.offset)
    elif isinstance(cfg.offset, dict):
      self._offset = torch.zeros_like(self._raw_actions)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.offset, self._actuator_names
      )
      self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError(
        f"Unsupported offset type: {type(cfg.offset)}."
        " Supported types are float and dict."
      )

  # Properties.

  @property
  def scale(self) -> torch.Tensor | float:
    return self._scale

  @property
  def offset(self) -> torch.Tensor | float:
    return self._offset

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def action_dim(self) -> int:
    return self._action_dim

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale + self._offset

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
  def __init__(self, cfg: actions_config.JointPositionActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._asset.data.default_joint_pos[:, self._actuator_ids].clone()

  def apply_actions(self):
    self._asset.write_joint_position_target_to_sim(
      self._processed_actions, self._actuator_ids
    )
