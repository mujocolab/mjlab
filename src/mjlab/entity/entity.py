from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab import actuator
from mjlab.entity.data import EntityData
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names
from mjlab.utils import spec_config as spec_cfg
from mjlab.utils.mujoco import dof_width, qpos_width
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
  mocap_id: int | None

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
  lights: tuple[spec_cfg.LightCfg, ...] = field(default_factory=tuple)
  cameras: tuple[spec_cfg.CameraCfg, ...] = field(default_factory=tuple)
  textures: tuple[spec_cfg.TextureCfg, ...] = field(default_factory=tuple)
  materials: tuple[spec_cfg.MaterialCfg, ...] = field(default_factory=tuple)
  collisions: tuple[spec_cfg.CollisionCfg, ...] = field(default_factory=tuple)

  # Misc.
  debug_vis: bool = False


@dataclass
class EntityArticulationInfoCfg:
  actuators: tuple[actuator.ActuatorCfg, ...] = field(default_factory=tuple)
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

  Fixed non-articulated entities can optionally be mocap bodies, whereby their
  position and orientation can be set directly each timestep rather than being
  determined by physics. This property can be useful for creating props with
  adjustable position and orientation.

  Supported Combinations:
  ----------------------
  | Type                      | Example             | is_fixed_base | is_articulated | is_actuated |
  |---------------------------|---------------------|---------------|----------------|-------------|
  | Fixed Non-articulated     | Table, wall         | True          | False          | False       |
  | Fixed Articulated         | Robot arm, door     | True          | True           | True/False  |
  | Floating Non-articulated  | Box, ball, mug      | False         | False          | False       |
  | Floating Articulated      | Humanoid, quadruped | False         | True           | True/False  |
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
    self._actuators: list[actuator.Actuator] = []

    self._apply_spec_editors()
    self._add_actuators()
    self._add_initial_state_keyframe()

  def _apply_spec_editors(self) -> None:
    for cfg_list in [
      self.cfg.lights,
      self.cfg.cameras,
      self.cfg.textures,
      self.cfg.materials,
      self.cfg.collisions,
    ]:
      for cfg in cfg_list:
        cfg.edit_spec(self._spec)

  def _add_actuators(self) -> None:
    if self.cfg.articulation is None:
      return

    for actuator_cfg in self.cfg.articulation.actuators:
      joint_ids, joint_names = self.find_joints(actuator_cfg.joint_names_expr)
      if len(joint_names) == 0:
        raise ValueError(
          "No joints found for actuator with expressions: "
          f"{actuator_cfg.joint_names_expr}"
        )
      actuator_instance = actuator_cfg.build(self, joint_ids, joint_names)
      actuator_instance.edit_spec(self._spec, joint_names)
      self._actuators.append(actuator_instance)

  def _add_initial_state_keyframe(self) -> None:
    qpos_components = []

    if self._free_joint is not None:
      qpos_components.extend([self.cfg.init_state.pos, self.cfg.init_state.rot])

    joint_pos = None
    if self._non_free_joints:
      joint_pos = resolve_expr(self.cfg.init_state.joint_pos, self.joint_names, 0.0)
      qpos_components.append(joint_pos)

    key_qpos = np.hstack(qpos_components) if qpos_components else np.array([])
    key = self._spec.add_key(name="init_state", qpos=key_qpos)

    if self.is_actuated and joint_pos is not None:
      name_to_pos = {name: joint_pos[i] for i, name in enumerate(self.joint_names)}
      ctrl = []
      for act in self._spec.actuators:
        joint_name = act.target
        ctrl.append(name_to_pos.get(joint_name, 0.0))
      key.ctrl = np.array(ctrl)

    if self.is_fixed_base:
      self.root_body.pos[:] = self.cfg.init_state.pos
      self.root_body.quat[:] = self.cfg.init_state.rot

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
    return len(self._actuators) > 0

  @property
  def is_mocap(self) -> bool:
    """Entity root body is a mocap body (only for fixed-base entities)."""
    return bool(self.root_body.mocap) if self.is_fixed_base else False

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  @property
  def data(self) -> EntityData:
    return self._data

  @property
  def actuators(self) -> list[actuator.Actuator]:
    return self._actuators

  @property
  def joint_names(self) -> tuple[str, ...]:
    return tuple(j.name.split("/")[-1] for j in self._non_free_joints)

  @property
  def body_names(self) -> tuple[str, ...]:
    return tuple(b.name.split("/")[-1] for b in self.spec.bodies[1:])

  @property
  def geom_names(self) -> tuple[str, ...]:
    return tuple(g.name.split("/")[-1] for g in self.spec.geoms)

  @property
  def site_names(self) -> tuple[str, ...]:
    return tuple(s.name.split("/")[-1] for s in self.spec.sites)

  @property
  def actuator_names(self) -> tuple[str, ...]:
    return tuple(a.name.split("/")[-1] for a in self.spec.actuators)

  @property
  def num_joints(self) -> int:
    return len(self.joint_names)

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
  def num_actuators(self) -> int:
    return len(self.actuator_names)

  @property
  def root_body(self) -> mujoco.MjsBody:
    return self.spec.bodies[1]

  # Methods.

  def find_bodies(
    self, name_keys: str | Sequence[str], preserve_order: bool = False
  ) -> tuple[list[int], list[str]]:
    return resolve_matching_names(name_keys, self.body_names, preserve_order)

  def find_joints(
    self,
    name_keys: str | Sequence[str],
    joint_subset: Sequence[str] | None = None,
    preserve_order: bool = False,
  ) -> tuple[list[int], list[str]]:
    if joint_subset is None:
      joint_subset = self.joint_names
    return resolve_matching_names(name_keys, joint_subset, preserve_order)

  def find_actuators(
    self,
    name_keys: str | Sequence[str],
    actuator_subset: Sequence[str] | None = None,
    preserve_order: bool = False,
  ):
    if actuator_subset is None:
      actuator_subset = self.actuator_names
    return resolve_matching_names(name_keys, actuator_subset, preserve_order)

  def find_geoms(
    self,
    name_keys: str | Sequence[str],
    geom_subset: Sequence[str] | None = None,
    preserve_order: bool = False,
  ):
    if geom_subset is None:
      geom_subset = self.geom_names
    return resolve_matching_names(name_keys, geom_subset, preserve_order)

  def find_sites(
    self,
    name_keys: str | Sequence[str],
    site_subset: Sequence[str] | None = None,
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

    for act in self._actuators:
      act.initialize(mj_model, model, data, device)

    # Root state.
    root_state_components = [self.cfg.init_state.pos, self.cfg.init_state.rot]
    if not self.is_fixed_base:
      root_state_components.extend(
        [self.cfg.init_state.lin_vel, self.cfg.init_state.ang_vel]
      )
    default_root_state = torch.tensor(
      sum((tuple(c) for c in root_state_components), ()),
      dtype=torch.float,
      device=device,
    ).repeat(nworld, 1)

    # Joint state.
    if self.is_articulated:
      default_joint_pos = torch.tensor(
        resolve_expr(self.cfg.init_state.joint_pos, self.joint_names, 0.0),
        device=device,
      )[None].repeat(nworld, 1)
      default_joint_vel = torch.tensor(
        resolve_expr(self.cfg.init_state.joint_vel, self.joint_names, 0.0),
        device=device,
      )[None].repeat(nworld, 1)

      # Joint limits.
      joint_ids_global = torch.tensor(
        [j.id for j in self._non_free_joints], device=device
      )
      dof_limits = model.jnt_range[:, joint_ids_global]
      default_joint_pos_limits = dof_limits.clone()
      joint_pos_limits = default_joint_pos_limits.clone()
      joint_pos_mean = (joint_pos_limits[..., 0] + joint_pos_limits[..., 1]) / 2
      joint_pos_range = joint_pos_limits[..., 1] - joint_pos_limits[..., 0]

      # Soft limits.
      soft_limit_factor = (
        self.cfg.articulation.soft_joint_pos_limit_factor
        if self.cfg.articulation
        else 1.0
      )
      soft_joint_pos_limits = torch.stack(
        [
          joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,
          joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,
        ],
        dim=-1,
      )
    else:
      empty_shape = (nworld, 0)
      default_joint_pos = torch.empty(*empty_shape, dtype=torch.float, device=device)
      default_joint_vel = torch.empty(*empty_shape, dtype=torch.float, device=device)
      default_joint_pos_limits = torch.empty(
        *empty_shape, 2, dtype=torch.float, device=device
      )
      joint_pos_limits = torch.empty(*empty_shape, 2, dtype=torch.float, device=device)
      soft_joint_pos_limits = torch.empty(
        *empty_shape, 2, dtype=torch.float, device=device
      )

    if self.is_actuated:
      joint_pos_target = torch.zeros(
        (nworld, self.num_joints), dtype=torch.float, device=device
      )
      joint_vel_target = torch.zeros(
        (nworld, self.num_joints), dtype=torch.float, device=device
      )
      joint_effort_target = torch.zeros(
        (nworld, self.num_joints), dtype=torch.float, device=device
      )
    else:
      joint_pos_target = torch.empty(nworld, 0, dtype=torch.float, device=device)
      joint_vel_target = torch.empty(nworld, 0, dtype=torch.float, device=device)
      joint_effort_target = torch.empty(nworld, 0, dtype=torch.float, device=device)

    self._data = EntityData(
      indexing=indexing,
      data=data,
      model=model,
      device=device,
      default_root_state=default_root_state,
      default_joint_pos=default_joint_pos,
      default_joint_vel=default_joint_vel,
      default_joint_pos_limits=default_joint_pos_limits,
      joint_pos_limits=joint_pos_limits,
      soft_joint_pos_limits=soft_joint_pos_limits,
      gravity_vec_w=torch.tensor([0.0, 0.0, -1.0], device=device).repeat(nworld, 1),
      forward_vec_b=torch.tensor([1.0, 0.0, 0.0], device=device).repeat(nworld, 1),
      is_fixed_base=self.is_fixed_base,
      is_articulated=self.is_articulated,
      is_actuated=self.is_actuated,
      joint_pos_target=joint_pos_target,
      joint_vel_target=joint_vel_target,
      joint_effort_target=joint_effort_target,
    )

  def update(self, dt: float) -> None:
    for act in self._actuators:
      act.update(dt)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.clear_state(env_ids)

    for act in self._actuators:
      act.reset(env_ids)

  def write_data_to_sim(self) -> None:
    self._apply_actuator_controls()

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
        Contains linear velocity (3) and angular velocity (3), both in world frame.
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

  def set_joint_position_target(
    self,
    position: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    """Set joint position targets.

    Args:
      position: Target joint poisitions with shape (N, num_joints).
      joint_ids: Optional joint indices to set. If None, set all joints.
      env_ids: Optional environment indices. If None, set all environments.
    """
    if env_ids is None:
      env_ids = slice(None)
    if joint_ids is None:
      joint_ids = slice(None)
    self._data.joint_pos_target[env_ids, joint_ids] = position

  def set_joint_velocity_target(
    self,
    velocity: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    """Set joint velocity targets.

    Args:
      velocity: Target joint velocities with shape (N, num_joints).
      joint_ids: Optional joint indices to set. If None, set all joints.
      env_ids: Optional environment indices. If None, set all environments.
    """
    if env_ids is None:
      env_ids = slice(None)
    if joint_ids is None:
      joint_ids = slice(None)
    self._data.joint_vel_target[env_ids, joint_ids] = velocity

  def set_joint_effort_target(
    self,
    effort: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    """Set joint effort targets.

    Args:
      effort: Target joint efforts with shape (N, num_joints).
      joint_ids: Optional joint indices to set. If None, set all joints.
      env_ids: Optional environment indices. If None, set all environments.
    """
    if env_ids is None:
      env_ids = slice(None)
    if joint_ids is None:
      joint_ids = slice(None)
    self._data.joint_effort_target[env_ids, joint_ids] = effort

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

  def write_mocap_pose_to_sim(
    self,
    mocap_pose: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    """Set the pose of a mocap body into the simulation.

    Args:
      mocap_pose: Tensor of shape (N, 7) where N is the number of environments.
        Format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    self._data.write_mocap_pose(mocap_pose, env_ids)

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

    if self.is_fixed_base and self.is_mocap:
      mocap_id = int(model.body_mocapid[self.root_body.id])
    else:
      mocap_id = None

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
      mocap_id=mocap_id,
      joint_q_adr=joint_q_adr,
      joint_v_adr=joint_v_adr,
      free_joint_q_adr=free_joint_q_adr,
      free_joint_v_adr=free_joint_v_adr,
    )

  def _apply_actuator_controls(self) -> None:
    for act in self._actuators:
      command = actuator.ActuatorCmd(
        position_target=self._data.joint_pos_target[:, act.joint_ids],
        velocity_target=self._data.joint_vel_target[:, act.joint_ids],
        effort_target=self._data.joint_effort_target[:, act.joint_ids],
        joint_pos=self._data.joint_pos[:, act.joint_ids],
        joint_vel=self._data.joint_vel[:, act.joint_ids],
      )
      self._data.write_ctrl(act.compute(command), act.ctrl_ids)
