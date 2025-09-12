from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity.config import EntityCfg
from mjlab.entity.data import EntityData
from mjlab.entity.indexing import EntityIndexing
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names
from mjlab.utils import string as string_utils
from mjlab.utils.mujoco import dof_width, qpos_width
from mjlab.utils.spec import is_joint_limited
from mjlab.utils.spec_editor.spec_editor import (
  ActuatorEditor,
  CameraEditor,
  CollisionEditor,
  KeyframeEditor,
  LightEditor,
  MaterialEditor,
  SensorEditor,
  TextureEditor,
)
from mjlab.utils.string import resolve_expr


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

  Notes:
  - Only one freejoint is allowed per entity
  - Actuators defined in XML are removed; use ActuatorCfg instead
  - Joint counts exclude the freejoint (only articulation joints)
  """

  def __init__(self, cfg: EntityCfg) -> None:
    self.cfg = cfg
    self._spec = cfg.spec_fn()

    self._give_names_to_missing_elems()
    self._validate_and_extract_joints()
    self._handle_xml_actuators()
    self._configure_spec()
    self._store_element_names()
    self._build_actuator_mapping()

  # Attributes.

  @property
  def is_fixed_base(self) -> bool:
    """Entity is welded to the world."""
    return self._root_joint is None

  @property
  def is_articulated(self) -> bool:
    """Entity is articulated (has fixed or actuated joints)."""
    return len(self._non_root_joints) > 0

  @property
  def is_actuated(self) -> bool:
    """Entity has actuated joints."""
    return self.cfg.articulation is not None and len(self.joint_actuators) > 0

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  @property
  def data(self) -> EntityData:
    return self._data

  @property
  def joint_names(self) -> list[str]:
    return self._joint_names

  @property
  def tendon_names(self) -> list[str]:
    return self._tendon_names

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

  @property
  def num_joints(self) -> int:
    return len(self._joint_names)

  @property
  def num_tendons(self) -> int:
    return len(self._tendon_names)

  @property
  def num_bodies(self) -> int:
    return len(self._body_names)

  @property
  def num_geoms(self) -> int:
    return len(self._geom_names)

  @property
  def num_sites(self) -> int:
    return len(self._site_names)

  @property
  def num_sensors(self) -> int:
    return len(self._sensor_names)

  @property
  def num_actuators(self) -> int:
    return len(self._actuator_names)

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

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    indexing = self._compute_indexing(mj_model, device)
    self.indexing = indexing

    # Global ID mappings for bodies, joints, actuators.
    local_body_ids = range(self.num_bodies)
    self._body_ids_global = torch.tensor(
      [indexing.body_local2global[lid] for lid in local_body_ids],
      dtype=torch.int,
      device=device,
    )

    # Joint mappings - only for articulated entities.
    if self.is_articulated:
      local_joint_ids = range(self.num_joints)
      self._joint_ids_global = torch.tensor(
        [indexing.joint_local2global[lid] for lid in local_joint_ids],
        dtype=torch.int,
        device=device,
      )
    else:
      self._joint_ids_global = torch.empty(0, dtype=torch.int, device=device)

    # Actuator mappings - only for articulated entities with actuated joints.
    if self.is_articulated and self.is_actuated:
      local_act_ids = resolve_matching_names(
        self._actuator_names, self.joint_actuators, True
      )[0]
      self._actuator_ids_global = torch.tensor(
        [indexing.actuator_local2global[lid] for lid in local_act_ids],
        dtype=torch.int,
        device=device,
      )
    else:
      self._actuator_ids_global = torch.empty(0, dtype=torch.int, device=device)

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

      # Joint limits and control parameters.
      dof_limits = model.jnt_range[:nworld, self._joint_ids_global]
      default_joint_pos_limits = dof_limits.clone()
      joint_pos_limits = default_joint_pos_limits.clone()
      joint_pos_mean = (joint_pos_limits[..., 0] + joint_pos_limits[..., 1]) / 2
      joint_pos_range = joint_pos_limits[..., 1] - joint_pos_limits[..., 0]

      # Get soft limit factor from config, with fallback.
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

      # Joint control parameters - only if we have actuators.
      if len(self._actuator_ids_global) > 0:
        default_joint_stiffness = model.actuator_gainprm[
          :nworld, self._actuator_ids_global, 0
        ]
        default_joint_damping = (
          -1.0 * model.actuator_biasprm[:nworld, self._actuator_ids_global, 2]
        )
        joint_stiffness = default_joint_stiffness.clone()
        joint_damping = default_joint_damping.clone()
      else:
        # No actuators - create empty tensors.
        default_joint_stiffness = torch.empty(
          nworld, 0, dtype=torch.float, device=device
        )
        default_joint_damping = torch.empty(nworld, 0, dtype=torch.float, device=device)
        joint_stiffness = torch.empty(nworld, 0, dtype=torch.float, device=device)
        joint_damping = torch.empty(nworld, 0, dtype=torch.float, device=device)

      # Joint position weights.
      if self.cfg.articulation and self.cfg.articulation.joint_pos_weight is not None:
        weight = string_utils.resolve_expr(
          self.cfg.articulation.joint_pos_weight, self.joint_names, 1.0
        )
      else:
        weight = [1.0] * len(self.joint_names)
      joint_pos_weight = torch.tensor(weight, device=device).repeat(nworld, 1)

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
      joint_pos_weight = torch.empty(nworld, 0, dtype=torch.float, device=device)
      default_joint_stiffness = torch.empty(nworld, 0, dtype=torch.float, device=device)
      default_joint_damping = torch.empty(nworld, 0, dtype=torch.float, device=device)
      joint_stiffness = torch.empty(nworld, 0, dtype=torch.float, device=device)
      joint_damping = torch.empty(nworld, 0, dtype=torch.float, device=device)

    # Universal constants - all entities need these.
    GRAVITY_VEC_W = torch.tensor([0.0, 0.0, -1.0], device=device).repeat(nworld, 1)
    FORWARD_VEC_B = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(nworld, 1)

    self._data = EntityData(
      indexing=indexing,
      data=data,
      device=device,
      body_names=self.body_names,
      geom_names=self.geom_names,
      site_names=self.site_names,
      sensor_names=self.sensor_names,
      joint_names=self.joint_names,
      default_root_state=default_root_state,
      default_joint_pos=default_joint_pos,
      default_joint_vel=default_joint_vel,
      default_joint_pos_limits=default_joint_pos_limits,
      joint_pos_limits=joint_pos_limits,
      soft_joint_pos_limits=soft_joint_pos_limits,
      joint_pos_weight=joint_pos_weight,
      default_joint_stiffness=default_joint_stiffness,
      default_joint_damping=default_joint_damping,
      joint_stiffness=joint_stiffness,
      joint_damping=joint_damping,
      FORWARD_VEC_B=FORWARD_VEC_B,
      GRAVITY_VEC_W=GRAVITY_VEC_W,
    )

  def update(self, dt: float) -> None:
    self._data.update(dt)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.clear_state(env_ids)

  def write_data_to_sim(self) -> None:
    pass

  def clear_state(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, torch.Tensor):
      env_ids = env_ids[:, None]
    v_slice = self.data.indexing.free_joint_v_adr
    # Reset external wrenches on bodies and DoFs.
    self._data.data.qfrc_applied[env_ids, v_slice] = 0.0
    self._data.data.xfrc_applied[env_ids, self._body_ids_global] = 0.0
    # TODO(kevin): Reset data.act if it exists.
    # TODO(kevin): Is this needed?
    self._data.data.qacc_warmstart[env_ids, v_slice] = 0.0

  def write_root_state_to_sim(
    self,
    root_state: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
  ):
    """Set the root state into the simulation.

    The root state consists of position (3), orientation as a (w, x, y, z)
    quaternion (4), linear velocity (3), and angular velocity (3), for a total
    of 13 values. All of the quantities are in the world frame.

    Args:
      root_state: Tensor of shape (N, 13) where N is the number of environments.
      env_ids: Optional tensor or slice specifying which environments to set. If
        None, all environments are set.
    """
    if self.is_fixed_base:
      raise ValueError("Cannot write root state for fixed-base entity.")
    assert root_state.shape[-1] == (3 + 4 + 3 + 3)
    self.write_root_link_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    self.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

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
    if self.is_fixed_base:
      raise ValueError("Cannot write root pose for fixed-base entity.")
    assert root_pose.shape[-1] == (3 + 4)
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, torch.Tensor):
      env_ids = env_ids[:, None]
    q_slice = self.data.indexing.free_joint_q_adr
    self._data.data.qpos[env_ids, q_slice] = root_pose

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
    if self.is_fixed_base:
      raise ValueError("Cannot write root velocity for fixed-base entity.")
    assert root_velocity.shape[-1] == (3 + 3)
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, torch.Tensor):
      env_ids = env_ids[:, None]
    v_slice = self.data.indexing.free_joint_v_adr
    self._data.data.qvel[env_ids, v_slice] = root_velocity

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
    if not self.is_articulated:
      raise ValueError("Cannot write joint state for non-articulated entity.")
    self.write_joint_position_to_sim(position, joint_ids=joint_ids, env_ids=env_ids)
    self.write_joint_velocity_to_sim(velocity, joint_ids=joint_ids, env_ids=env_ids)

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
    if not self.is_articulated:
      raise ValueError("Cannot write joint position for non-articulated entity.")
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, torch.Tensor):
      env_ids = env_ids[:, None]
    if joint_ids is None:
      joint_ids = slice(None)
    q_slice = self._data.indexing.joint_q_adr[joint_ids]
    self._data.data.qpos[env_ids, q_slice] = position

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
    if not self.is_articulated:
      raise ValueError("Cannot write joint velocity for non-articulated entity.")
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, torch.Tensor):
      env_ids = env_ids[:, None]
    if joint_ids is None:
      joint_ids = slice(None)
    v_slice = self._data.indexing.joint_v_adr[joint_ids]
    self._data.data.qvel[env_ids, v_slice] = velocity

  def write_joint_stiffness_to_sim(
    self,
    stiffness: torch.Tensor | float,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    if not self.is_articulated:
      raise ValueError("Cannot write joint stiffness for non-articulated entity.")
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, torch.Tensor):
      env_ids = env_ids[:, None]
    if joint_ids is None:
      joint_ids = slice(None)
    self._data.joint_stiffness[env_ids, joint_ids] = stiffness

  def write_joint_damping_to_sim(
    self,
    damping: torch.Tensor | float,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    if not self.is_articulated:
      raise ValueError("Cannot write joint damping for non-articulated entity.")
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, torch.Tensor):
      env_ids = env_ids[:, None]
    if joint_ids is None:
      joint_ids = slice(None)
    self._data.joint_damping[env_ids, joint_ids] = damping

  def set_external_force_and_torque(
    self,
    forces: torch.Tensor,
    torques: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
    body_ids: Sequence[int] | slice | None = None,
  ) -> None:
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, torch.Tensor):
      env_ids = env_ids[:, None]
    if body_ids is None:
      body_ids = slice(None)
    self._data.set_external_wrench(env_ids, body_ids, forces, torques)

  ##
  # Private methods.
  ##

  def _configure_spec(self) -> None:
    for light in self.cfg.lights:
      LightEditor(light).edit_spec(self._spec)
    for camera in self.cfg.cameras:
      CameraEditor(camera).edit_spec(self._spec)
    for tex in self.cfg.textures:
      TextureEditor(tex).edit_spec(self._spec)
    for mat in self.cfg.materials:
      MaterialEditor(mat).edit_spec(self._spec)
    for sns in self.cfg.sensors:
      SensorEditor(sns).edit_spec(self._spec)
    for col in self.cfg.collisions:
      CollisionEditor(col).edit_spec(self._spec)
    if self.cfg.articulation:
      ActuatorEditor(self.cfg.articulation.actuators).edit_spec(self._spec)

    if self._root_joint:
      KeyframeEditor(self.cfg.init_state).edit_spec(self._spec)

  def _give_names_to_missing_elems(self) -> None:
    """Ensure all important elements of the spec have names to simplify attachment."""

    def _incremental_rename(
      elem_list: Sequence[mujoco.MjsElement], elem_type: str
    ) -> None:
      counter: int = 0
      for elem in elem_list:
        if not elem.name:  # type: ignore
          elem.name = f"{elem_type}_{counter}"  # type: ignore
          counter += 1

    # TODO(kevin): Rethink this.
    _incremental_rename(self._spec.bodies, "body")
    _incremental_rename(self._spec.geoms, "geom")
    _incremental_rename(self._spec.sites, "site")
    _incremental_rename(self._spec.sensors, "sensor")

  def _compute_indexing(self, model: mujoco.MjModel, device: str) -> EntityIndexing:
    ##
    # BODY
    ##
    body_ids = []
    body_root_ids = []
    body_iquats = []
    body_local2global = {}
    for local_id, body_ in enumerate(self.spec.bodies[1:]):
      body_name = body_.name
      body = model.body(body_name)
      body_ids.append(body.id)
      body_root_ids.extend(body.rootid)
      body_iquats.append(body.iquat.tolist())
      body_local2global[local_id] = body.id
    body_ids = torch.tensor(body_ids, dtype=torch.int, device=device)
    body_root_ids = torch.tensor(body_root_ids, dtype=torch.int, device=device)
    body_iquats = torch.tensor(body_iquats, dtype=torch.float, device=device)

    if self.cfg.articulation:
      # Find the root body by looking for the free joint and then getting its body ID.
      root_body_id = None
      for joint in self.spec.joints:
        jnt = model.joint(joint.name)
        if jnt.type[0] == mujoco.mjtJoint.mjJNT_FREE:
          # NOTE: `jnt.bodyid` is currently returning an
          # array when it should only be returning a single
          # value. So instead, we use model.jnt_bodyid.
          root_body_id = int(model.jnt_bodyid[jnt.id])
      if root_body_id is None:
        raise ValueError("Entity has no root body.")
    else:
      # Assume that the first body after world is the root. Note in MuJoCo, world is
      # always body ID 0.
      root_body_id = 1
    root_body_iquat = torch.tensor(
      model.body_iquat[root_body_id], dtype=torch.float, device=device
    )

    ##
    # GEOM
    ##
    geom_ids = []
    geom_body_ids = []
    geom_local2global = {}
    for local_id, geom_ in enumerate(self.spec.geoms):
      geom_name = geom_.name
      geom = model.geom(geom_name)
      geom_ids.append(geom.id)
      geom_body_ids.append(geom.bodyid[0])
      geom_local2global[local_id] = geom.id
    geom_ids = torch.tensor(geom_ids, dtype=torch.int, device=device)
    geom_body_ids = torch.tensor(geom_body_ids, dtype=torch.int, device=device)

    ##
    # SITE
    ##
    site_ids = []
    site_body_ids = []
    site_local2global = {}
    for local_id, site_ in enumerate(self.spec.sites):
      site_name = site_.name
      site = model.site(site_name)
      site_ids.append(site.id)
      site_body_ids.append(site.bodyid[0])
      site_local2global[local_id] = site.id
    site_ids = torch.tensor(site_ids, dtype=torch.int, device=device)
    site_body_ids = torch.tensor(site_body_ids, dtype=torch.int, device=device)

    ##
    # ACTUATOR
    ##
    ctrl_ids = []
    actuator_local2global = {}
    for local_id, actuator in enumerate(self.spec.actuators):
      act = model.actuator(actuator.name)
      ctrl_ids.append(act.id)
      actuator_local2global[local_id] = act.id

    ##
    # SENSOR
    ##
    sensor_adr = {}
    for sensor in self.spec.sensors:
      sensor_name = sensor.name
      sns = model.sensor(sensor_name)
      dim = sns.dim[0]
      start_adr = sns.adr[0]
      sensor_adr[sensor_name] = torch.arange(
        start_adr, start_adr + dim, dtype=torch.int, device=device
      )

    ##
    # JOINT
    ##
    joint_q_adr = []
    joint_v_adr = []
    free_joint_q_adr = []
    free_joint_v_adr = []
    joint_local2global = {}
    for local_id, joint in enumerate(self.spec.joints):
      jnt = model.joint(joint.name)
      jnt_type = jnt.type[0]
      vadr = jnt.dofadr[0]
      qadr = jnt.qposadr[0]
      if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        free_joint_v_adr.extend(range(vadr, vadr + 6))
        free_joint_q_adr.extend(range(qadr, qadr + 7))
      else:
        vdim = dof_width(jnt_type)
        joint_v_adr.extend(range(vadr, vadr + vdim))
        qdim = qpos_width(jnt_type)
        joint_q_adr.extend(range(qadr, qadr + qdim))
        # -1 because for accessing jnt_* fields in model, we want to skip the
        # freejoint.
        joint_local2global[local_id - 1] = jnt.id

    return EntityIndexing(
      root_body_id=root_body_id,
      body_ids=body_ids,
      body_root_ids=body_root_ids,
      geom_ids=geom_ids,
      geom_body_ids=geom_body_ids,
      site_ids=site_ids,
      site_body_ids=site_body_ids,
      ctrl_ids=torch.tensor(ctrl_ids, dtype=torch.int, device=device),
      root_body_iquat=root_body_iquat,
      body_iquats=body_iquats,
      sensor_adr=sensor_adr,
      joint_q_adr=torch.tensor(joint_q_adr, dtype=torch.int, device=device),
      joint_v_adr=torch.tensor(joint_v_adr, dtype=torch.int, device=device),
      free_joint_v_adr=torch.tensor(free_joint_v_adr, dtype=torch.int, device=device),
      free_joint_q_adr=torch.tensor(free_joint_q_adr, dtype=torch.int, device=device),
      body_local2global=body_local2global,
      geom_local2global=geom_local2global,
      site_local2global=site_local2global,
      actuator_local2global=actuator_local2global,
      joint_local2global=joint_local2global,
    )

  def _validate_and_extract_joints(self) -> None:
    """Validate joint configuration and extract root/non-root joints.

    Raises:
      ValueError: If entity has invalid joint configuration.
    """
    freejoints = []
    other_joints = []

    for joint in self._spec.joints:
      if joint.type == mujoco.mjtJoint.mjJNT_FREE:
        freejoints.append(joint)
      else:
        other_joints.append(joint)

    # Validation: Only one freejoint allowed.
    if len(freejoints) > 1:
      joint_names = [j.name for j in freejoints]
      raise ValueError(
        f"Entity has multiple freejoints: {joint_names}. "
        "Only one freejoint is allowed per entity. "
        "Consider splitting into separate entities or using a different joint type."
      )

    self._root_joint: mujoco.MjsJoint | None = freejoints[0] if freejoints else None
    self._non_root_joints: tuple[mujoco.MjsJoint, ...] = tuple(other_joints)

    # Additional validation for articulated entities.
    # Check that articulated joints have reasonable ranges. Even robots with unlimited
    # joint ranges should use joint limits with a very large range.
    if self._non_root_joints:
      for joint in self._non_root_joints:
        if joint.type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
          if not is_joint_limited(joint):
            import warnings

            warnings.warn(
              f"Joint '{joint.name}' has no range limits defined.",
              UserWarning,
              stacklevel=2,
            )

  # TODO(kevin): Remove this once we support XML actuators.
  def _handle_xml_actuators(self) -> None:
    """Handle actuators defined in XML (remove them with warning)."""
    if len(self._spec.actuators) > 0:
      actuator_names = [a.name for a in self._spec.actuators]
      print(
        f"WARNING: Entity has {len(self._spec.actuators)} XML actuator(s): {actuator_names}. "
        "These will be removed. Use ActuatorCfg in EntityArticulationInfoCfg instead."
      )
      for actuator in self._spec.actuators:
        self._spec.delete(actuator)

  def _store_element_names(self) -> None:
    self._joint_names = [j.name for j in self._non_root_joints]
    self._tendon_names = [t.name for t in self._spec.tendons]
    self._body_names = [b.name for b in self.spec.bodies if b.name != "world"]
    self._geom_names = [g.name for g in self.spec.geoms]
    self._site_names = [s.name for s in self._spec.sites]
    self._sensor_names = [s.name for s in self._spec.sensors]
    self._actuator_names = [a.name for a in self._spec.actuators]

  def _build_actuator_mapping(self) -> None:
    """Build mapping between actuators and joints."""
    self.actuator_to_joint = {}
    self.joint_actuators = []

    if self.cfg.articulation:
      for actuator in self._spec.actuators:
        if actuator.trntype != mujoco.mjtTrn.mjTRN_JOINT:
          continue
        self.actuator_to_joint[actuator.name] = actuator.target
      self.joint_actuators = list(self.actuator_to_joint.values())

      # Validation: Check for joints without actuators if actuation is configured
      if self.cfg.articulation.actuators:
        unactuated_joints = set(self._joint_names) - set(self.joint_actuators)
        if unactuated_joints:
          import warnings

          warnings.warn(
            f"Joints without actuators: {unactuated_joints}. "
            "These joints will be passive.",
            UserWarning,
            stacklevel=2,
          )
