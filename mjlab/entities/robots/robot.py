import mujoco
import numpy as np

from mjlab.core import entity
from mjlab.entities.robots.robot_config import RobotCfg
from mjlab.entities.robots import editors
from mjlab.entities.common import editors as common_editors
from mjlab.utils.spec import get_non_root_joints


class Robot(entity.Entity):
  def __init__(self, robot_cfg: RobotCfg):
    self._cfg = robot_cfg
    spec = mujoco.MjSpec.from_file(str(robot_cfg.xml_path), assets=robot_cfg.asset_fn())
    super().__init__(spec)

    self._non_root_joints = get_non_root_joints(self._spec)
    self._modify_joint_range()

    self._configure_keyframes()
    self._configure_actuators()
    self._configure_sensors()
    self._configure_collisions()

  # @property
  # def spec(self) -> mujoco.MjSpec:
  #   return self._spec

  # Private methods.

  def _configure_keyframes(self) -> None:
    for key in self._cfg.keyframes:
      editors.KeyframeEditor(key).edit_spec(self._spec)

  def _configure_actuators(self) -> None:
    editors.ActuatorEditor(self._cfg.actuators).edit_spec(self._spec)

  def _configure_sensors(self) -> None:
    for sens in self._cfg.sensors:
      editors.SensorEditor(sens).edit_spec(self._spec)

  def _configure_collisions(self) -> None:
    for col in self._cfg.collisions:
      common_editors.CollisionEditor(col).edit_spec(self._spec)

  def _modify_joint_range(self) -> None:
    ranges = [j.range for j in self._non_root_joints]
    lowers = np.array([r[0] for r in ranges])
    uppers = np.array([r[1] for r in ranges])
    c = (lowers + uppers) / 2
    r = uppers - lowers
    soft_lowers = c - 0.5 * r * self._cfg.soft_joint_pos_limit_factor
    soft_uppers = c + 0.5 * r * self._cfg.soft_joint_pos_limit_factor
    for i, j in enumerate(self._non_root_joints):
      j.range[0] = soft_lowers[i]
      j.range[1] = soft_uppers[i]
