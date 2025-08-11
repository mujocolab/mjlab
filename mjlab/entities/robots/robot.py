from typing import Sequence

import mujoco_warp as mjwarp
import torch
import mujoco

from mjlab.utils.string import resolve_expr
from mjlab.entities import entity
from mjlab.entities.robots.robot_config import RobotCfg
from mjlab.utils.spec_editor.spec_editor import (
  ActuatorEditor,
  CollisionEditor,
  KeyframeEditor,
)
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names
from mjlab.utils.spec import get_non_root_joints
from mjlab.utils import string as string_utils
from mjlab.entities.robots.robot_data import RobotData
from mjlab.entities.indexing import EntityIndexing
import warp as wp


class Robot(entity.Entity):
  cfg: RobotCfg

  def __init__(self, robot_cfg: RobotCfg):
    super().__init__(robot_cfg)

    # Joints.
    self._non_root_joints = get_non_root_joints(self._spec)
    self._joint_names = [j.name for j in self._non_root_joints]
    self.num_joints = len(self._joint_names)

    # Bodies.
    self._body_names = [b.name for b in self.spec.bodies if b.name != "world"]
    self.num_bodies = len(self._body_names)

    # Geoms.
    self._geom_names = [g.name for g in self.spec.geoms]
    self.num_geoms = len(self._geom_names)

    # Sites.
    self._site_names = [s.name for s in self._spec.sites]
    self.num_sites = len(self._site_names)

    # Actuators.
    self._actuator_names = [a.name for a in self._spec.actuators]
    self.actuator_to_joint = {}
    for actuator in self._spec.actuators:
      if actuator.trntype != mujoco.mjtTrn.mjTRN_JOINT:
        continue
      self.actuator_to_joint[actuator.name] = actuator.target
    self.joint_actuators = list(self.actuator_to_joint.values())

    # Sensors.
    self._sensor_names = [s.name for s in self._spec.sensors]
    self.num_sensors = len(self._sensor_names)

  # Attributes.

  @property
  def data(self) -> RobotData:
    return self._data

  @property
  def joint_names(self) -> list[str]:
    return self._joint_names

  @property
  def body_names(self) -> list[str]:
    return self._body_names

  @property
  def geom_names(self) -> list[str]:
    return self._geom_names

  @property
  def site_names(self) -> list[str]:
    return self._site_names

  @property
  def sensor_names(self) -> list[str]:
    return self._sensor_names

  @property
  def actuator_names(self) -> list[str]:
    return self._actuator_names

  # Find methods.

  def find_bodies(
    self, name_keys: str | Sequence[str], preserve_order: bool = False
  ) -> tuple[list[int], list[str]]:
    return resolve_matching_names(name_keys, self.body_names, preserve_order)

  def find_joints(
    self,
    name_keys: str | Sequence[str],
    joint_subset: list[str] | None = None,
    preserve_order: bool = False,
  ) -> tuple[list[int], list[str]]:
    if joint_subset is None:
      joint_subset = self.joint_names
    return resolve_matching_names(name_keys, joint_subset, preserve_order)

  def find_actuators(
    self,
    name_keys: str | Sequence[str],
    actuator_subset: list[str] | None = None,
    preserve_order: bool = False,
  ):
    if actuator_subset is None:
      actuator_subset = self.actuator_names
    return resolve_matching_names(name_keys, actuator_subset, preserve_order)

  def find_geoms(
    self,
    name_keys: str | Sequence[str],
    geom_subset: list[str] | None = None,
    preserve_order: bool = False,
  ):
    if geom_subset is None:
      geom_subset = self.geom_names
    return resolve_matching_names(name_keys, geom_subset, preserve_order)

  def find_sensors(
    self,
    name_keys: str | Sequence[str],
    sensor_subset: list[str] | None = None,
    preserve_order: bool = False,
  ):
    if sensor_subset is None:
      sensor_subset = self.sensor_names
    return resolve_matching_names(name_keys, sensor_subset, preserve_order)

  def find_sites(
    self,
    name_keys: str | Sequence[str],
    site_subset: list[str] | None = None,
    preserve_order: bool = False,
  ):
    if site_subset is None:
      site_subset = self.site_names
    return resolve_matching_names(name_keys, site_subset, preserve_order)

  # ABC implementations.

  def initialize(
    self,
    indexing: EntityIndexing,
    model: mujoco.MjModel,
    data: mjwarp.Data,
    device: str,
    wp_model,
  ) -> None:
    del model

    self._data = RobotData(indexing=indexing, data=data, device=device)

    self._data.body_names = self.body_names
    self._data.geom_names = self.geom_names
    self._data.site_names = self.site_names
    self._data.sensor_names = self.sensor_names
    self._data.joint_names = self.joint_names

    # Default root state.
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

    # Default joint state (pos, vel).
    self._data.default_joint_pos = torch.tensor(
      resolve_expr(self.cfg.init_state.joint_pos, self.joint_names), device=device
    )[None].repeat(data.nworld, 1)
    self._data.default_joint_vel = torch.tensor(
      resolve_expr(self.cfg.init_state.joint_vel, self.joint_names), device=device
    )[None].repeat(data.nworld, 1)
    dof_limits = torch.tensor(
      [j.range.tolist() for j in self._non_root_joints],
      dtype=torch.float,
      device=device,
    )
    self._data.default_joint_pos_limits = dof_limits[None].repeat(data.nworld, 1, 1)
    self._data.joint_pos_limits = self._data.default_joint_pos_limits.clone()
    joint_pos_mean = (
      self._data.joint_pos_limits[..., 0] + self._data.joint_pos_limits[..., 1]
    ) / 2
    joint_pos_range = (
      self._data.joint_pos_limits[..., 1] - self._data.joint_pos_limits[..., 0]
    )
    soft_limit_factor = self.cfg.soft_joint_pos_limit_factor
    self._data.soft_joint_pos_limits = torch.zeros(
      data.nworld, self.num_joints, 2, device=device
    )
    self._data.soft_joint_pos_limits[..., 0] = (
      joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor
    )
    self._data.soft_joint_pos_limits[..., 1] = (
      joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor
    )
    act_ids = resolve_matching_names(self._actuator_names, self.joint_actuators, True)[
      0
    ]
    self._data.default_joint_stiffness = wp.to_torch(wp_model.actuator_gainprm)[
      : data.nworld, act_ids, 0
    ]
    self._data.default_joint_damping = (
      -1.0 * wp.to_torch(wp_model.actuator_biasprm)[: data.nworld, act_ids, 2]
    )
    self._data.joint_stiffness = self._data.default_joint_stiffness.clone()
    self._data.joint_damping = self._data.default_joint_damping.clone()
    if self.cfg.joint_pos_weight is not None:
      weight = string_utils.resolve_expr(
        self.cfg.joint_pos_weight, self.joint_names, 1.0
      )
    else:
      weight = [1.0] * len(self.joint_names)
    self._data.joint_pos_weight = torch.tensor(weight, device=device).repeat(
      data.nworld, 1
    )

  def update(self, dt: float) -> None:
    self._data.update(dt)

  def reset(self, env_ids: Sequence[int] | None = None):
    if env_ids is None:
      env_ids = slice(None)
    # TODO(kevin): This should only be resetting this entity's attributes.
    self._data.data.qacc_warmstart[env_ids] = 0.0
    self._data.data.xfrc_applied[env_ids] = 0.0
    self._data.data.qfrc_applied[env_ids] = 0.0
    self._data.data.act[env_ids] = 0.0

  def write_data_to_sim(self) -> None:
    pass

  # Write methods.

  def write_root_state_to_sim(
    self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None
  ):
    self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    self.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

  def write_root_link_pose_to_sim(
    self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None
  ):
    assert root_pose.shape[-1] == 7
    if env_ids is None:
      env_ids = slice(None)
    if env_ids != slice(None):
      env_ids = env_ids[:, None]
    q_slice = self.data.indexing.free_joint_q_adr
    self._data.data.qpos[env_ids, q_slice] = root_pose

  def write_root_link_velocity_to_sim(
    self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None
  ):
    assert root_velocity.shape[-1] == 6
    if env_ids is None:
      env_ids = slice(None)
    if env_ids != slice(None):
      env_ids = env_ids[:, None]
    v_slice = self.data.indexing.free_joint_v_adr
    self._data.data.qvel[env_ids, v_slice] = root_velocity

    # TODO(kevin): Is this required?
    self._data.data.qacc[env_ids, v_slice] = 0.0
    self._data.data.qacc_warmstart[env_ids, v_slice] = 0.0

  def write_joint_state_to_sim(
    self,
    position: torch.Tensor,
    velocity: torch.Tensor,
    joint_ids: Sequence[int] | slice | None = None,
    env_ids: Sequence[int] | slice | None = None,
  ):
    self.write_joint_position_to_sim(position, joint_ids=joint_ids, env_ids=env_ids)
    self.write_joint_velocity_to_sim(velocity, joint_ids=joint_ids, env_ids=env_ids)

  def write_joint_position_to_sim(
    self,
    position: torch.Tensor,
    joint_ids: Sequence[int] | slice | None = None,
    env_ids: Sequence[int] | slice | None = None,
  ):
    if env_ids is None:
      env_ids = slice(None)
    if joint_ids is None:
      joint_ids = slice(None)
    if env_ids != slice(None):
      env_ids = env_ids[:, None]
    q_slice = self._data.indexing.joint_q_adr[joint_ids]
    self._data.data.qpos[env_ids, q_slice] = position

  def write_joint_velocity_to_sim(
    self,
    velocity: torch.Tensor,
    joint_ids: Sequence[int] | slice | None = None,
    env_ids: Sequence[int] | slice | None = None,
  ):
    if env_ids is None:
      env_ids = slice(None)
    if joint_ids is None:
      joint_ids = slice(None)
    if env_ids != slice(None):
      env_ids = env_ids[:, None]
    v_slice = self._data.indexing.joint_v_adr[joint_ids]
    self._data.data.qvel[env_ids, v_slice] = velocity

  def write_joint_stiffness_to_sim(
    self,
    stiffness: torch.Tensor | float,
    joint_ids: Sequence[int] | slice | None = None,
    env_ids: Sequence[int] | None = None,
  ) -> None:
    if env_ids is None:
      env_ids = slice(None)
    if joint_ids is None:
      joint_ids = slice(None)
    if env_ids != slice(None):
      env_ids = env_ids[:, None]
    self._data.joint_stiffness[env_ids, joint_ids] = stiffness

  def write_joint_damping_to_sim(
    self,
    damping: torch.Tensor | float,
    joint_ids: Sequence[int] | slice | None = None,
    env_ids: Sequence[int] | None = None,
  ) -> None:
    if env_ids is None:
      env_ids = slice(None)
    if joint_ids is None:
      joint_ids = slice(None)
    if env_ids != slice(None):
      env_ids = env_ids[:, None]
    self._data.joint_damping[env_ids, joint_ids] = damping

  # Private methods.

  def _configure_spec(self) -> None:
    super()._configure_spec()
    editor = ActuatorEditor(self.cfg.actuators)
    editor.edit_spec(self._spec)
    self._actuator_joint_names = editor.jnt_names
    for col in self.cfg.collisions:
      CollisionEditor(col).edit_spec(self._spec)
    KeyframeEditor(self.cfg.init_state).edit_spec(self._spec)
