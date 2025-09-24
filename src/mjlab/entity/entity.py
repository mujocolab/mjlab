from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.entity.data import EntityData
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names
from mjlab.utils.mujoco import dof_width, qpos_width
from mjlab.utils.spec_editor.spec_editor import (
  ActuatorEditor,
  CameraEditor,
  CollisionEditor,
  LightEditor,
  MaterialEditor,
  SensorEditor,
  TextureEditor,
)
from mjlab.utils.spec_editor.spec_editor_config import (
  ActuatorCfg,
  CameraCfg,
  CollisionCfg,
  LightCfg,
  MaterialCfg,
  SensorCfg,
  TextureCfg,
)
from mjlab.utils.string import resolve_expr


@dataclass(frozen=True)
class EntityIndexing:
  """Maps entity elements to global indices and addresses in the simulation."""

  # Elements.
  bodies: tuple[mujoco.MjsBody, ...]
  joints: tuple[mujoco.MjsJoint, ...]
  geoms: tuple[mujoco.MjsGeom, ...]
  sites: tuple[mujoco.MjsSite, ...]
  actuators: tuple[mujoco.MjsActuator, ...] | None

  # Indices.
  body_ids: torch.Tensor
  geom_ids: torch.Tensor
  site_ids: torch.Tensor
  ctrl_ids: torch.Tensor
  joint_ids: torch.Tensor

  # Addresses.
  joint_q_adr: torch.Tensor
  joint_v_adr: torch.Tensor
  free_joint_q_adr: torch.Tensor
  free_joint_v_adr: torch.Tensor

  @property
  def root_body_id(self) -> int:
    return self.bodies[0].id


@dataclass
class EntityCfg:
  @dataclass
  class InitialStateCfg:
    # Root position and orientation.
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # Root linear and angular velocity (only for floating base entities).
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Articulation (only for articulated entities).
    joint_pos: dict[str, float] = field(default_factory=lambda: {".*": 0.0})
    joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})

  init_state: InitialStateCfg = field(default_factory=InitialStateCfg)
  spec_fn: Callable[[], mujoco.MjSpec] = field(
    default_factory=lambda: (lambda: mujoco.MjSpec())
  )
  articulation: EntityArticulationInfoCfg | None = None

  # Editors.
  lights: tuple[LightCfg, ...] = field(default_factory=tuple)
  cameras: tuple[CameraCfg, ...] = field(default_factory=tuple)
  textures: tuple[TextureCfg, ...] = field(default_factory=tuple)
  materials: tuple[MaterialCfg, ...] = field(default_factory=tuple)
  sensors: tuple[SensorCfg, ...] = field(default_factory=tuple)
  collisions: tuple[CollisionCfg, ...] = field(default_factory=tuple)

  # Misc.
  debug_vis: bool = False


@dataclass
class EntityArticulationInfoCfg:
  actuators: tuple[ActuatorCfg, ...] = field(default_factory=tuple)
  soft_joint_pos_limit_factor: float = 1.0


class Entity:
  """An entity represents a physical object in the simulation.

  Entity Type Matrix
  ==================
  MuJoCo entities can be categorized along two dimensions:

  1. Base Type:
    - Fixed Base: Entity is welded to the world (no freejoint)
    - Floating Base: Entity has 6 DOF movement (has freejoint)

  2. Articulation:
    - Non-articulated: No joints other than freejoint
    - Articulated: Has joints in kinematic tree (may or may not be actuated)

  Supported Combinations:
  ----------------------
  | Type                      | Example                    | is_fixed_base | is_articulated | is_actuated |
  |---------------------------|----------------------------|---------------|----------------|-------------|
  | Fixed Non-articulated     | Table, wall, ground plane  | True          | False          | False       |
  | Fixed Articulated         | Robot arm, door on hinges  | True          | True           | True/False  |
  | Floating Non-articulated  | Box, ball, mug             | False         | False          | False       |
  | Floating Articulated      | Humanoid, quadruped        | False         | True           | True/False  |
  """

  def __init__(self, cfg: EntityCfg) -> None:
    self.cfg = cfg
    self._spec = cfg.spec_fn()

    # Identify free joint and articulated joints.
    all_joints = self._spec.joints
    self._free_joint = None
    self._non_free_joints = tuple(all_joints)
    if all_joints and all_joints[0].type == mujoco.mjtJoint.mjJNT_FREE:
      self._free_joint = all_joints[0]
      self._non_free_joints = tuple(all_joints[1:])

    self._apply_spec_editors()
    self._add_initial_state_keyframe()
    # TODO: Should init_state.pos/rot be applied to root body if fixed base?

  def _apply_spec_editors(self) -> None:
    editors = [
      (self.cfg.lights, LightEditor),
      (self.cfg.cameras, CameraEditor),
      (self.cfg.textures, TextureEditor),
      (self.cfg.materials, MaterialEditor),
      (self.cfg.sensors, SensorEditor),
      (self.cfg.collisions, CollisionEditor),
    ]

    for configs, editor_class in editors:
      for config in configs:
        editor_class(config).edit_spec(self._spec)

    if self.cfg.articulation:
      ActuatorEditor(self.cfg.articulation.actuators).edit_spec(self._spec)

  def _add_initial_state_keyframe(self) -> None:
    qpos_components = []

    if self._free_joint is not None:
      qpos_components.extend([self.cfg.init_state.pos, self.cfg.init_state.rot])

    joint_pos = None
    if self._non_free_joints:
      joint_pos = resolve_expr(self.cfg.init_state.joint_pos, self.joint_names)
      qpos_components.append(joint_pos)

    key_qpos = np.hstack(qpos_components) if qpos_components else np.array([])
    key = self._spec.add_key(name="init_state", qpos=key_qpos)

    if self.is_actuated and joint_pos is not None:
      key.ctrl = joint_pos

  # Attributes.

  @property
  def is_fixed_base(self) -> bool:
    """Entity is welded to the world."""
    return self._free_joint is None

  @property
  def is_articulated(self) -> bool:
    """Entity is articulated (has fixed or actuated joints)."""
    return len(self._non_free_joints) > 0

  @property
  def is_actuated(self) -> bool:
    """Entity has actuated joints."""
    return self.cfg.articulation is not None and self.num_actuators > 0

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  @property
  def data(self) -> EntityData:
    return self._data

  @property
  def joint_names(self) -> list[str]:
    return [j.name.split("/")[-1] for j in self._non_free_joints]

  @property
  def tendon_names(self) -> list[str]:
    return [t.name.split("/")[-1] for t in self._spec.tendons]

  @property
  def body_names(self) -> list[str]:
    return [b.name.split("/")[-1] for b in self.spec.bodies[1:]]

  @property
  def geom_names(self) -> list[str]:
    return [g.name.split("/")[-1] for g in self.spec.geoms]

  @property
  def site_names(self) -> list[str]:
    return [s.name.split("/")[-1] for s in self.spec.sites]

  @property
  def sensor_names(self) -> list[str]:
    return [s.name.split("/")[-1] for s in self.spec.sensors]

  @property
  def actuator_names(self) -> list[str]:
    return [a.name.split("/")[-1] for a in self.spec.actuators]

  @property
  def num_joints(self) -> int:
    return len(self.joint_names)

  @property
  def num_tendons(self) -> int:
    return len(self.tendon_names)

  @property
  def num_bodies(self) -> int:
    return len(self.body_names)

  @property
  def num_geoms(self) -> int:
    return len(self.geom_names)

  @property
  def num_sites(self) -> int:
    return len(self.site_names)

  @property
  def num_sensors(self) -> int:
    return len(self.sensor_names)

  @property
  def num_actuators(self) -> int:
    return len(self.actuator_names)

  # Methods.

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

  def find_tendons(
    self,
    name_keys: str | Sequence[str],
    tendon_subset: list[str] | None = None,
    preserve_order: bool = False,
  ) -> tuple[list[int], list[str]]:
    if tendon_subset is None:
      tendon_subset = self.tendon_names
    return resolve_matching_names(name_keys, tendon_subset, preserve_order)

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

  def compile(self) -> mujoco.MjModel:
    """Compile the underlying MjSpec into an MjModel."""
    return self.spec.compile()

  def write_xml(self, xml_path: Path) -> None:
    """Write the MjSpec to disk."""
    with open(xml_path, "w") as f:
      f.write(self.spec.to_xml())

  def to_zip(self, path: Path) -> None:
    """Write the MjSpec to a zip file."""
    with path.open("wb") as f:
      mujoco.MjSpec.to_zip(self.spec, f)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    indexing = self._compute_indexing(mj_model, device)
    self.indexing = indexing
    nworld = data.nworld

    # Root state - only for movable entities.
    if not self.is_fixed_base:
      default_root_state = (
        tuple(self.cfg.init_state.pos)
        + tuple(self.cfg.init_state.rot)
        + tuple(self.cfg.init_state.lin_vel)
        + tuple(self.cfg.init_state.ang_vel)
      )
      default_root_state = torch.tensor(
        default_root_state, dtype=torch.float, device=device
      )
      default_root_state = default_root_state.repeat(nworld, 1)
    else:
      # Static entities have no root state.
      default_root_state = torch.empty(nworld, 0, dtype=torch.float, device=device)

    # Joint state - only for articulated entities.
    if self.is_articulated:
      default_joint_pos = torch.tensor(
        resolve_expr(self.cfg.init_state.joint_pos, self.joint_names), device=device
      )[None].repeat(nworld, 1)
      default_joint_vel = torch.tensor(
        resolve_expr(self.cfg.init_state.joint_vel, self.joint_names), device=device
      )[None].repeat(nworld, 1)

      if self.is_actuated:
        default_joint_stiffness = model.actuator_gainprm[:, self.indexing.ctrl_ids, 0]
        default_joint_damping = -model.actuator_biasprm[:, self.indexing.ctrl_ids, 2]
      else:
        default_joint_stiffness = torch.empty(
          nworld, 0, dtype=torch.float, device=device
        )
        default_joint_damping = torch.empty(nworld, 0, dtype=torch.float, device=device)

      # Joint limits and control parameters.
      joint_ids_global = [j.id for j in self._non_free_joints]
      dof_limits = model.jnt_range[:, joint_ids_global]
      default_joint_pos_limits = dof_limits.clone()
      joint_pos_limits = default_joint_pos_limits.clone()
      joint_pos_mean = (joint_pos_limits[..., 0] + joint_pos_limits[..., 1]) / 2
      joint_pos_range = joint_pos_limits[..., 1] - joint_pos_limits[..., 0]

      # Get soft limit factor from config.
      if self.cfg.articulation:
        soft_limit_factor = self.cfg.articulation.soft_joint_pos_limit_factor
      else:
        soft_limit_factor = 1.0

      soft_joint_pos_limits = torch.zeros(nworld, self.num_joints, 2, device=device)
      soft_joint_pos_limits[..., 0] = (
        joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor
      )
      soft_joint_pos_limits[..., 1] = (
        joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor
      )
    else:
      # Non-articulated entities - create empty tensors.
      default_joint_pos = torch.empty(nworld, 0, dtype=torch.float, device=device)
      default_joint_vel = torch.empty(nworld, 0, dtype=torch.float, device=device)
      default_joint_pos_limits = torch.empty(
        nworld, 0, 2, dtype=torch.float, device=device
      )
      joint_pos_limits = torch.empty(nworld, 0, 2, dtype=torch.float, device=device)
      soft_joint_pos_limits = torch.empty(
        nworld, 0, 2, dtype=torch.float, device=device
      )
      default_joint_stiffness = torch.empty(nworld, 0, dtype=torch.float, device=device)
      default_joint_damping = torch.empty(nworld, 0, dtype=torch.float, device=device)

    self._data = EntityData(
      indexing=indexing,
      data=data,
      model=model,
      device=device,
      default_root_state=default_root_state,
      default_joint_pos=default_joint_pos,
      default_joint_vel=default_joint_vel,
      default_joint_stiffness=default_joint_stiffness,
      default_joint_damping=default_joint_damping,
      default_joint_pos_limits=default_joint_pos_limits,
      joint_pos_limits=joint_pos_limits,
      soft_joint_pos_limits=soft_joint_pos_limits,
      gravity_vec_w=torch.tensor([0.0, 0.0, -1.0], device=device).repeat(nworld, 1),
      forward_vec_b=torch.tensor([1.0, 0.0, 0.0], device=device).repeat(nworld, 1),
      is_fixed_base=self.is_fixed_base,
      is_articulated=self.is_articulated,
      is_actuated=self.is_actuated,
    )

  def update(self, dt: float) -> None:
    self._data.update(dt)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.clear_state(env_ids)

  def write_data_to_sim(self) -> None:
    pass

  def clear_state(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._data.clear_state(env_ids)

  def write_root_state_to_sim(
    self, root_state: torch.Tensor, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    """Set the root state into the simulation.

    The root state consists of position (3), orientation as a (w, x, y, z)
    quaternion (4), linear velocity (3), and angular velocity (3), for a total
    of 13 values. All of the quantities are in the world frame.

    Args:
      root_state: Tensor of shape (N, 13) where N is the number of environments.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_root_state(root_state, env_ids)

  def write_root_link_pose_to_sim(
    self,
    root_pose: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
  ):
    """Set the root pose into the simulation. Like `write_root_state_to_sim()`
    but only sets position and orientation.

    Args:
      root_pose: Tensor of shape (N, 7) where N is the number of environments.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_root_pose(root_pose, env_ids)

  def write_root_link_velocity_to_sim(
    self,
    root_velocity: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
  ):
    """Set the root velocity into the simulation. Like `write_root_state_to_sim()`
    but only sets linear and angular velocity.

    Args:
      root_velocity: Tensor of shape (N, 6) where N is the number of environments.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_root_velocity(root_velocity, env_ids)

  def write_joint_state_to_sim(
    self,
    position: torch.Tensor,
    velocity: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ):
    """Set the joint state into the simulation.

    The joint state consists of joint positions and velocities. It does not include
    the root state.

    Args:
      position: Tensor of shape (N, num_joints) where N is the number of environments.
      velocity: Tensor of shape (N, num_joints) where N is the number of environments.
      joint_ids: Optional tensor or slice specifying which joints to set. If None,
        all joints are set.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_joint_state(position, velocity, joint_ids, env_ids)

  def write_joint_position_to_sim(
    self,
    position: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ):
    """Set the joint positions into the simulation. Like `write_joint_state_to_sim()`
    but only sets joint positions.

    Args:
      position: Tensor of shape (N, num_joints) where N is the number of environments.
      joint_ids: Optional tensor or slice specifying which joints to set. If None,
        all joints are set.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_joint_position(position, joint_ids, env_ids)

  def write_joint_velocity_to_sim(
    self,
    velocity: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ):
    """Set the joint velocities into the simulation. Like `write_joint_state_to_sim()`
    but only sets joint velocities.

    Args:
      velocity: Tensor of shape (N, num_joints) where N is the number of environments.
      joint_ids: Optional tensor or slice specifying which joints to set. If None,
        all joints are set.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_joint_velocity(velocity, joint_ids, env_ids)

  def write_joint_position_target_to_sim(
    self,
    position_target: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    """Set the joint position targets for PD control.

    Args:
      position_target: Tensor of shape (N, num_joints) where N is the number of
        environments.
      joint_ids: Optional tensor or slice specifying which joints to set. If None,
        all joints are set.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_ctrl(position_target, joint_ids, env_ids)

  def write_external_wrench_to_sim(
    self,
    forces: torch.Tensor,
    torques: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
    body_ids: Sequence[int] | slice | None = None,
  ) -> None:
    """Apply external wrenches to bodies in the simulation.

    Underneath the hood, this sets the `xfrc_applied` field in the MuJoCo data
    structure. The wrenches are specified in the world frame and persist until
    the next call to this function or until the simulation is reset.

    Args:
      forces: Tensor of shape (N, num_bodies, 3) where N is the number of
        environments.
      torques: Tensor of shape (N, num_bodies, 3) where N is the number of
        environments.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
      body_ids: Optional list of body indices or slice specifying which bodies to
        apply the wrenches to. If None, wrenches are applied to all bodies.
    """
    self._data.write_external_wrench(forces, torques, body_ids, env_ids)

  ##
  # Private methods.
  ##

  def _compute_indexing(self, model: mujoco.MjModel, device: str) -> EntityIndexing:
    bodies = tuple([b for b in self.spec.bodies[1:]])
    joints = self._non_free_joints
    geoms = tuple(self.spec.geoms)
    sites = tuple(self.spec.sites)

    body_ids = torch.tensor([b.id for b in bodies], dtype=torch.int, device=device)
    geom_ids = torch.tensor([g.id for g in geoms], dtype=torch.int, device=device)
    site_ids = torch.tensor([s.id for s in sites], dtype=torch.int, device=device)
    joint_ids = torch.tensor([j.id for j in joints], dtype=torch.int, device=device)

    if self.is_actuated:
      actuators = tuple(self.spec.actuators)
      ctrl_ids = torch.tensor([a.id for a in actuators], dtype=torch.int, device=device)
    else:
      actuators = None
      ctrl_ids = torch.empty(0, dtype=torch.int, device=device)

    joint_q_adr = []
    joint_v_adr = []
    free_joint_q_adr = []
    free_joint_v_adr = []
    for joint in self.spec.joints:
      jnt = model.joint(joint.name)
      jnt_type = jnt.type[0]
      vadr = jnt.dofadr[0]
      qadr = jnt.qposadr[0]
      if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        free_joint_v_adr.extend(range(vadr, vadr + 6))
        free_joint_q_adr.extend(range(qadr, qadr + 7))
      else:
        joint_v_adr.extend(range(vadr, vadr + dof_width(jnt_type)))
        joint_q_adr.extend(range(qadr, qadr + qpos_width(jnt_type)))
    joint_q_adr = torch.tensor(joint_q_adr, dtype=torch.int, device=device)
    joint_v_adr = torch.tensor(joint_v_adr, dtype=torch.int, device=device)
    free_joint_v_adr = torch.tensor(free_joint_v_adr, dtype=torch.int, device=device)
    free_joint_q_adr = torch.tensor(free_joint_q_adr, dtype=torch.int, device=device)

    return EntityIndexing(
      bodies=bodies,
      joints=joints,
      geoms=geoms,
      sites=sites,
      actuators=actuators,
      body_ids=body_ids,
      geom_ids=geom_ids,
      site_ids=site_ids,
      ctrl_ids=ctrl_ids,
      joint_ids=joint_ids,
      joint_q_adr=joint_q_adr,
      joint_v_adr=joint_v_adr,
      free_joint_q_adr=free_joint_q_adr,
      free_joint_v_adr=free_joint_v_adr,
    )
