import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.utils.string import resolve_expr
from mjlab.entities import entity
from mjlab.entities.robots.robot_config import RobotCfg
from mjlab.utils.spec_editor.spec_editor import (
  ActuatorEditor,
  SensorEditor,
  CollisionEditor,
)
from mjlab.utils.spec import get_non_root_joints
from mjlab.entities.robots.robot_data import RobotData
from mjlab.entities.indexing import EntityIndexing


class Robot(entity.Entity):
  cfg: RobotCfg

  def __init__(self, robot_cfg: RobotCfg):
    super().__init__(robot_cfg)

    self._non_root_joints = get_non_root_joints(self._spec)
    self._joint_names = [j.name for j in self._non_root_joints]
    self._body_names = [b.name for b in self.spec.bodies if b.name != "world"]
    
    self._modify_joint_range()

  # Public methods.

  @property
  def joint_names(self) -> list[str]:
    return self._joint_names

  @property
  def body_names(self) -> list[str]:
    return self._body_names

  def initialize(
    self, indexing: EntityIndexing, data: mjwarp.Data, device: str
  ) -> None:
    self._data = RobotData(indexing=indexing, data=data, device=device)

    default_root_state = (
      tuple(self.cfg.init_state.pos)
      + tuple(self.cfg.init_state.rot)
      + tuple(self.cfg.init_state.lin_vel)
      + tuple(self.cfg.init_state.ang_vel)
    )
    default_root_state = torch.tensor(
      default_root_state, dtype=torch.float, device=device
    )
    self._data.default_root_state = default_root_state.repeat(data.nworld, 1)

    self._data.default_joint_pos = torch.tensor(
      resolve_expr(self.cfg.init_state.joint_pos, self.joint_names),
      device=device,
    ).repeat(data.nworld, 1)
    self._data.default_joint_vel = torch.tensor(
      resolve_expr(self.cfg.init_state.joint_vel, self.joint_names),
      device=device,
    ).repeat(data.nworld, 1)

    self._data.joint_names = self.joint_names
    self._data.body_names = self.body_names

  def update(self, dt: float) -> None:
    self._data.update(dt)

  def reset(self):
    pass

  @property
  def data(self) -> RobotData:
    return self._data

  # Private methods.

  def _configure_spec(self) -> None:
    super()._configure_spec()
    ActuatorEditor(self.cfg.actuators).edit_spec(self._spec)
    for sens in self.cfg.sensors:
      SensorEditor(sens).edit_spec(self._spec)
    for col in self.cfg.collisions:
      CollisionEditor(col).edit_spec(self._spec)

  def _modify_joint_range(self) -> None:
    ranges = [j.range for j in self._non_root_joints]
    lowers = np.array([r[0] for r in ranges])
    uppers = np.array([r[1] for r in ranges])
    c = (lowers + uppers) / 2
    r = uppers - lowers
    soft_lowers = c - 0.5 * r * self.cfg.soft_joint_pos_limit_factor
    soft_uppers = c + 0.5 * r * self.cfg.soft_joint_pos_limit_factor
    for i, j in enumerate(self._non_root_joints):
      j.range[0] = soft_lowers[i]
      j.range[1] = soft_uppers[i]
