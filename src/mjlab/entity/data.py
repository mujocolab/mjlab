from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import mujoco_warp as mjwarp
import torch

from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_from_matrix,
  quat_mul,
)
from mjlab.utils.torch import compose_index

if TYPE_CHECKING:
  from mjlab.entity.entity import EntityIndexing


def compute_velocity_from_cvel(
  pos: torch.Tensor,
  subtree_com: torch.Tensor,
  cvel: torch.Tensor,
) -> torch.Tensor:
  """Convert cvel quantities to world-frame velocities."""
  lin_vel_c = cvel[..., 3:6]
  ang_vel_c = cvel[..., 0:3]
  offset = subtree_com - pos
  lin_vel_w = lin_vel_c - torch.cross(ang_vel_c, offset, dim=-1)
  ang_vel_w = ang_vel_c
  return torch.cat([lin_vel_w, ang_vel_w], dim=-1)


def _index(
  env_ids: torch.Tensor | slice, ids: torch.Tensor | slice
) -> tuple[torch.Tensor | slice, torch.Tensor | slice]:
  """Make ``[env_ids, ids]`` select the outer product when both are tensors."""
  if isinstance(env_ids, torch.Tensor) and isinstance(ids, torch.Tensor):
    return env_ids[:, None], ids
  return env_ids, ids


@dataclass
class EntityData:
  """Data container for an entity.

  Note: Write methods (write_*) modify state directly. Read properties (e.g.,
  root_link_pose_w) require sim.forward() to be current. If you write then read,
  call sim.forward() in between. Event order matters when mixing reads and writes.
  All inputs/outputs use world frame.

  Properties derived from kinematic quantities (``xpos``, ``xquat``, ``cvel``, ...)
  are cached until the next simulation kernel launch (``sim.step()``,
  ``sim.forward()``, ``sim.reset()``), so reading e.g. ``root_link_quat_w`` from
  several observation and reward terms in the same step costs one computation.
  Treat the returned tensors as read-only: mutating them in place would be visible
  to every other reader in the same step. Properties that read ``qpos``/``qvel``
  directly (``joint_pos``, ``joint_vel``, ...) are views into the simulation
  arrays and always reflect the latest writes.
  """

  indexing: EntityIndexing
  data: mjwarp.Data
  model: mjwarp.Model
  device: str

  default_root_state: torch.Tensor
  default_joint_pos: torch.Tensor
  default_joint_vel: torch.Tensor

  default_joint_pos_limits: torch.Tensor
  joint_pos_limits: torch.Tensor
  soft_joint_pos_limits: torch.Tensor

  gravity_vec_w: torch.Tensor
  forward_vec_b: torch.Tensor

  is_fixed_base: bool
  is_articulated: bool
  is_actuated: bool

  joint_pos_target: torch.Tensor
  joint_vel_target: torch.Tensor
  joint_effort_target: torch.Tensor

  tendon_len_target: torch.Tensor
  tendon_vel_target: torch.Tensor
  tendon_effort_target: torch.Tensor

  site_effort_target: torch.Tensor

  encoder_bias: torch.Tensor

  # Cache of derived kinematic quantities, valid for ``data.version`` equal to
  # ``_cache_version``. See ``_derived_cache``.
  _cache: dict[str, torch.Tensor] = field(
    default_factory=dict, init=False, repr=False, compare=False
  )
  _cache_version: int = field(default=-1, init=False, repr=False, compare=False)

  # State dimensions.
  POS_DIM = 3
  QUAT_DIM = 4
  LIN_VEL_DIM = 3
  ANG_VEL_DIM = 3
  ROOT_POSE_DIM = POS_DIM + QUAT_DIM  # 7
  ROOT_VEL_DIM = LIN_VEL_DIM + ANG_VEL_DIM  # 6
  ROOT_STATE_DIM = ROOT_POSE_DIM + ROOT_VEL_DIM  # 13

  def write_root_state(
    self, root_state: torch.Tensor, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    if self.is_fixed_base:
      raise ValueError("Cannot write root state for fixed-base entity.")
    assert root_state.shape[-1] == self.ROOT_STATE_DIM

    self.write_root_pose(root_state[:, : self.ROOT_POSE_DIM], env_ids)
    self.write_root_velocity(root_state[:, self.ROOT_POSE_DIM :], env_ids)

  def write_root_pose(
    self, pose: torch.Tensor, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    if self.is_fixed_base:
      raise ValueError("Cannot write root pose for fixed-base entity.")
    assert pose.shape[-1] == self.ROOT_POSE_DIM

    env_ids = self._resolve_env_ids(env_ids)
    self.data.qpos[_index(env_ids, self.indexing.free_joint_q_sel)] = pose

  def write_root_velocity(
    self, velocity: torch.Tensor, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    if self.is_fixed_base:
      raise ValueError("Cannot write root velocity for fixed-base entity.")
    assert velocity.shape[-1] == self.ROOT_VEL_DIM

    env_ids = self._resolve_env_ids(env_ids)
    quat_w = self.data.qpos[_index(env_ids, self._free_joint_quat_index())]
    ang_vel_b = quat_apply_inverse(quat_w, velocity[:, 3:])
    velocity_qvel = torch.cat([velocity[:, :3], ang_vel_b], dim=-1)
    self.data.qvel[_index(env_ids, self.indexing.free_joint_v_sel)] = velocity_qvel

  def write_root_velocity_b(
    self, velocity_b: torch.Tensor, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    """Write the root link velocity given in the root link's body frame.

    The orientation is read from qpos, so this is safe to call during a reset
    before forward() has refreshed derived kinematics.
    """
    if self.is_fixed_base:
      raise ValueError("Cannot write root velocity for fixed-base entity.")
    assert velocity_b.shape[-1] == self.ROOT_VEL_DIM

    env_ids = self._resolve_env_ids(env_ids)
    quat_w = self.data.qpos[_index(env_ids, self._free_joint_quat_index())]
    lin_vel_w = quat_apply(quat_w, velocity_b[:, :3])
    # Free joint qvel holds world-frame linear and body-frame angular velocity.
    velocity_qvel = torch.cat([lin_vel_w, velocity_b[:, 3:]], dim=-1)
    self.data.qvel[_index(env_ids, self.indexing.free_joint_v_sel)] = velocity_qvel

  def write_root_com_velocity(
    self, velocity: torch.Tensor, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    if self.is_fixed_base:
      raise ValueError("Cannot write root COM velocity for fixed-base entity.")
    assert velocity.shape[-1] == self.ROOT_VEL_DIM

    env_ids = env_ids if env_ids is not None else slice(None)
    com_offset_b = self.model.body_ipos[:, self.indexing.root_body_id]
    quat_w = self.data.qpos[:, self._free_joint_quat_index()][env_ids]
    com_offset_w = quat_apply(quat_w, com_offset_b[env_ids])
    lin_vel_com = velocity[:, :3]
    ang_vel_w = velocity[:, 3:]
    lin_vel_link = lin_vel_com - torch.cross(ang_vel_w, com_offset_w, dim=-1)
    link_velocity = torch.cat([lin_vel_link, ang_vel_w], dim=-1)
    self.write_root_velocity(link_velocity, env_ids)

  def write_joint_state(
    self,
    position: torch.Tensor,
    velocity: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    if not self.is_articulated:
      raise ValueError("Cannot write joint state for non-articulated entity.")

    self.write_joint_position(position, joint_ids, env_ids)
    self.write_joint_velocity(velocity, joint_ids, env_ids)

  def write_joint_position(
    self,
    position: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    if not self.is_articulated:
      raise ValueError("Cannot write joint position for non-articulated entity.")

    env_ids = self._resolve_env_ids(env_ids)
    q_index = compose_index(
      self.indexing.joint_q_sel, self.indexing.joint_q_adr, joint_ids
    )
    self.data.qpos[_index(env_ids, q_index)] = position

  def write_joint_velocity(
    self,
    velocity: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    if not self.is_articulated:
      raise ValueError("Cannot write joint velocity for non-articulated entity.")

    env_ids = self._resolve_env_ids(env_ids)
    v_index = compose_index(
      self.indexing.joint_v_sel, self.indexing.joint_v_adr, joint_ids
    )
    self.data.qvel[_index(env_ids, v_index)] = velocity

  def write_external_wrench(
    self,
    force: torch.Tensor | None,
    torque: torch.Tensor | None,
    body_ids: Sequence[int] | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    env_ids = self._resolve_env_ids(env_ids)
    if body_ids is None or isinstance(body_ids, (slice, torch.Tensor)):
      global_body_ids = compose_index(
        self.indexing.body_sel, self.indexing.body_ids, body_ids
      )
    else:
      global_body_ids = self.indexing.body_ids[body_ids]
    env_index, body_index = _index(env_ids, global_body_ids)
    if force is not None:
      self.data.xfrc_applied[env_index, body_index, 0:3] = force
    if torque is not None:
      self.data.xfrc_applied[env_index, body_index, 3:6] = torque

  def write_ctrl(
    self,
    ctrl: torch.Tensor,
    ctrl_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    if not self.is_actuated:
      raise ValueError("Cannot write control for non-actuated entity.")

    env_ids = self._resolve_env_ids(env_ids)
    global_ctrl_ids = compose_index(
      self.indexing.ctrl_sel, self.indexing.ctrl_ids, ctrl_ids
    )
    self.data.ctrl[_index(env_ids, global_ctrl_ids)] = ctrl

  def write_mocap_pose(
    self, pose: torch.Tensor, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    if self.indexing.mocap_id is None:
      raise ValueError("Cannot write mocap pose for non-mocap entity.")
    assert pose.shape[-1] == self.ROOT_POSE_DIM

    env_ids = self._resolve_env_ids(env_ids)
    self.data.mocap_pos[env_ids, self.indexing.mocap_id] = pose[:, 0:3]
    self.data.mocap_quat[env_ids, self.indexing.mocap_id] = pose[:, 3:7]

  def clear_state(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self.is_actuated:
      env_ids = self._resolve_env_ids(env_ids)
      for target in (
        self.joint_pos_target,
        self.joint_vel_target,
        self.joint_effort_target,
        self.tendon_len_target,
        self.tendon_vel_target,
        self.tendon_effort_target,
        self.site_effort_target,
      ):
        # Targets for absent transmission types are (num_envs, 0): skip the launch.
        if target.shape[1] > 0:
          target[env_ids] = 0.0

  def _resolve_env_ids(
    self, env_ids: torch.Tensor | slice | None
  ) -> torch.Tensor | slice:
    """Convert env_ids to consistent indexing format."""
    if env_ids is None:
      return slice(None)
    return env_ids

  def _free_joint_quat_index(self) -> torch.Tensor | slice:
    """Index of the free joint's quaternion within qpos."""
    return compose_index(
      self.indexing.free_joint_q_sel, self.indexing.free_joint_q_adr, slice(3, 7)
    )

  def _joint_dof_field(self, field_name: str) -> torch.Tensor:
    """Return a generalized-force field sliced to this entity's joint DoFs."""
    field = getattr(self.data, field_name)
    return field[:, self.indexing.joint_v_sel]

  def _derived_cache(self) -> dict[str, torch.Tensor]:
    """Return the cache for quantities derived from the current kinematic state.

    The cache is keyed on ``data.version``, which :class:`~mjlab.sim.sim.Simulation`
    bumps after every kernel launch that modifies the simulation arrays. If the
    data object exposes no version (e.g. a raw ``mjwarp.Data``), nothing is cached.
    """
    version = getattr(self.data, "version", None)
    if version is None:
      return {}
    if version != self._cache_version:
      self._cache.clear()
      self._cache_version = version
    return self._cache

  # Root properties

  @property
  def root_link_pose_w(self) -> torch.Tensor:
    """Root link pose in world frame. Shape (num_envs, 7)."""
    cache = self._derived_cache()
    if (out := cache.get("root_link_pose_w")) is None:
      pos_w = self.data.xpos[:, self.indexing.root_body_id]  # (num_envs, 3)
      quat_w = self.data.xquat[:, self.indexing.root_body_id]  # (num_envs, 4)
      out = cache["root_link_pose_w"] = torch.cat([pos_w, quat_w], dim=-1)
    return out

  @property
  def root_link_vel_w(self) -> torch.Tensor:
    """Root link velocity in world frame. Shape (num_envs, 6)."""
    # NOTE: Equivalently, can read this from qvel[:6] but the angular part
    # will be in body frame and needs to be rotated to world frame.
    # Note also that an extra forward() call might be required to make
    # both values equal.
    cache = self._derived_cache()
    if (out := cache.get("root_link_vel_w")) is None:
      pos = self.data.xpos[:, self.indexing.root_body_id]  # (num_envs, 3)
      subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
      cvel = self.data.cvel[:, self.indexing.root_body_id]  # (num_envs, 6)
      out = cache["root_link_vel_w"] = compute_velocity_from_cvel(
        pos, subtree_com, cvel
      )
    return out

  @property
  def root_com_pose_w(self) -> torch.Tensor:
    """Root center-of-mass pose in world frame. Shape (num_envs, 7)."""
    cache = self._derived_cache()
    if (out := cache.get("root_com_pose_w")) is None:
      pos_w = self.data.xipos[:, self.indexing.root_body_id]
      quat = self.data.xquat[:, self.indexing.root_body_id]
      body_iquat = self.model.body_iquat[:, self.indexing.root_body_id]
      assert body_iquat is not None
      quat_w = quat_mul(quat, body_iquat.squeeze(1))
      out = cache["root_com_pose_w"] = torch.cat([pos_w, quat_w], dim=-1)
    return out

  @property
  def root_com_vel_w(self) -> torch.Tensor:
    """Root center-of-mass velocity in world frame. Shape (num_envs, 6)."""
    # NOTE: Equivalent sensor is framelinvel/frameangvel with objtype="body".
    cache = self._derived_cache()
    if (out := cache.get("root_com_vel_w")) is None:
      pos = self.data.xipos[:, self.indexing.root_body_id]  # (num_envs, 3)
      subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
      cvel = self.data.cvel[:, self.indexing.root_body_id]  # (num_envs, 6)
      out = cache["root_com_vel_w"] = compute_velocity_from_cvel(pos, subtree_com, cvel)
    return out

  # Body properties

  @property
  def body_link_pose_w(self) -> torch.Tensor:
    """Body link pose in world frame. Shape (num_envs, num_bodies, 7)."""
    cache = self._derived_cache()
    if (out := cache.get("body_link_pose_w")) is None:
      pos_w = self.data.xpos[:, self.indexing.body_sel]
      quat_w = self.data.xquat[:, self.indexing.body_sel]
      out = cache["body_link_pose_w"] = torch.cat([pos_w, quat_w], dim=-1)
    return out

  @property
  def body_link_vel_w(self) -> torch.Tensor:
    """Body link velocity in world frame. Shape (num_envs, num_bodies, 6)."""
    # NOTE: Equivalent sensor is framelinvel/frameangvel with objtype="xbody".
    cache = self._derived_cache()
    if (out := cache.get("body_link_vel_w")) is None:
      pos = self.data.xpos[:, self.indexing.body_sel]  # (num_envs, num_bodies, 3)
      subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
      cvel = self.data.cvel[:, self.indexing.body_sel]
      out = cache["body_link_vel_w"] = compute_velocity_from_cvel(
        pos, subtree_com.unsqueeze(1), cvel
      )
    return out

  @property
  def body_com_pose_w(self) -> torch.Tensor:
    """Body center-of-mass pose in world frame. Shape (num_envs, num_bodies, 7)."""
    cache = self._derived_cache()
    if (out := cache.get("body_com_pose_w")) is None:
      pos_w = self.data.xipos[:, self.indexing.body_sel]
      quat = self.data.xquat[:, self.indexing.body_sel]
      body_iquat = self.model.body_iquat[:, self.indexing.body_sel]
      quat_w = quat_mul(quat, body_iquat)
      out = cache["body_com_pose_w"] = torch.cat([pos_w, quat_w], dim=-1)
    return out

  @property
  def body_com_vel_w(self) -> torch.Tensor:
    """Body center-of-mass velocity in world frame. Shape (num_envs, num_bodies, 6)."""
    # NOTE: Equivalent sensor is framelinvel/frameangvel with objtype="body".
    cache = self._derived_cache()
    if (out := cache.get("body_com_vel_w")) is None:
      pos = self.data.xipos[:, self.indexing.body_sel]
      subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
      cvel = self.data.cvel[:, self.indexing.body_sel]
      out = cache["body_com_vel_w"] = compute_velocity_from_cvel(
        pos, subtree_com.unsqueeze(1), cvel
      )
    return out

  @property
  def body_external_wrench(self) -> torch.Tensor:
    """Body external wrench in world frame. Shape (num_envs, num_bodies, 6)."""
    return self.data.xfrc_applied[:, self.indexing.body_sel]

  # Geom properties

  @property
  def geom_pose_w(self) -> torch.Tensor:
    """Geom pose in world frame. Shape (num_envs, num_geoms, 7)."""
    cache = self._derived_cache()
    if (out := cache.get("geom_pose_w")) is None:
      pos_w = self.data.geom_xpos[:, self.indexing.geom_sel]
      xmat = self.data.geom_xmat[:, self.indexing.geom_sel]
      quat_w = quat_from_matrix(xmat)
      out = cache["geom_pose_w"] = torch.cat([pos_w, quat_w], dim=-1)
    return out

  @property
  def geom_vel_w(self) -> torch.Tensor:
    """Geom velocity in world frame. Shape (num_envs, num_geoms, 6)."""
    cache = self._derived_cache()
    if (out := cache.get("geom_vel_w")) is None:
      pos = self.data.geom_xpos[:, self.indexing.geom_sel]
      body_ids = self.model.geom_bodyid[self.indexing.geom_sel]  # (num_geoms,)
      subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
      cvel = self.data.cvel[:, body_ids]
      out = cache["geom_vel_w"] = compute_velocity_from_cvel(
        pos, subtree_com.unsqueeze(1), cvel
      )
    return out

  # Site properties

  @property
  def site_pose_w(self) -> torch.Tensor:
    """Site pose in world frame. Shape (num_envs, num_sites, 7)."""
    cache = self._derived_cache()
    if (out := cache.get("site_pose_w")) is None:
      pos_w = self.data.site_xpos[:, self.indexing.site_sel]
      mat_w = self.data.site_xmat[:, self.indexing.site_sel]
      quat_w = quat_from_matrix(mat_w)
      out = cache["site_pose_w"] = torch.cat([pos_w, quat_w], dim=-1)
    return out

  @property
  def site_vel_w(self) -> torch.Tensor:
    """Site velocity in world frame. Shape (num_envs, num_sites, 6)."""
    cache = self._derived_cache()
    if (out := cache.get("site_vel_w")) is None:
      pos = self.data.site_xpos[:, self.indexing.site_sel]
      body_ids = self.model.site_bodyid[self.indexing.site_sel]  # (num_sites,)
      subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
      cvel = self.data.cvel[:, body_ids]
      out = cache["site_vel_w"] = compute_velocity_from_cvel(
        pos, subtree_com.unsqueeze(1), cvel
      )
    return out

  # Joint properties

  @property
  def joint_pos(self) -> torch.Tensor:
    """Joint positions. Shape (num_envs, num_joints)."""
    return self.data.qpos[:, self.indexing.joint_q_sel]

  @property
  def joint_pos_biased(self) -> torch.Tensor:
    """Joint positions with encoder bias applied. Shape (num_envs, num_joints)."""
    return self.joint_pos + self.encoder_bias

  @property
  def joint_vel(self) -> torch.Tensor:
    """Joint velocities. Shape (num_envs, nv)."""
    return self.data.qvel[:, self.indexing.joint_v_sel]

  @property
  def joint_acc(self) -> torch.Tensor:
    """Joint accelerations. Shape (num_envs, nv)."""
    return self.data.qacc[:, self.indexing.joint_v_sel]

  # Tendon properties

  @property
  def tendon_len(self) -> torch.Tensor:
    """Tendon lengths. Shape (num_envs, num_tendons)."""
    return self.data.ten_length[:, self.indexing.tendon_sel]

  @property
  def tendon_vel(self) -> torch.Tensor:
    """Tendon velocities. Shape (num_envs, num_tendons)."""
    return self.data.ten_velocity[:, self.indexing.tendon_sel]

  # Generalized forces

  @property
  def joint_torques(self) -> torch.Tensor:
    """Joint torques. Shape (num_envs, nv)."""
    raise NotImplementedError(
      "Joint torques are ambiguous. Use 'qfrc_actuator' for actuator forces "
      "in joint space, or 'qfrc_external' for body wrench contributions."
    )

  @property
  def actuator_force(self) -> torch.Tensor:
    """Scalar actuator output in actuation space. Shape (num_envs, nu).

    This is not the same as joint-space generalized force. Use ``qfrc_actuator`` for
    the actuator contribution projected into DoF space.
    """
    return self.data.actuator_force[:, self.indexing.ctrl_sel]

  @property
  def qfrc_actuator(self) -> torch.Tensor:
    """Forces produced by all actuators, mapped into joint space.

    For motors this is the commanded torque times the gear ratio. For position and
    velocity actuators this is the force computed by the internal PD law. When
    ``actuatorgravcomp`` is enabled on a joint, the gravity compensation force is
    included here.
    """
    return self._joint_dof_field("qfrc_actuator")

  @property
  def qfrc_external(self) -> torch.Tensor:
    """Forces on joints due to Cartesian wrenches applied to bodies.

    When a force or torque is applied to a body via ``xfrc_applied``, this property
    gives the equivalent joint forces (the Jacobian transpose mapping).
    """
    # MuJoCo folds J^T * xfrc_applied into qfrc_smooth without storing it.
    # Recover via the qfrc_smooth identity:
    #   qfrc_smooth = qfrc_actuator + qfrc_passive - qfrc_bias
    #                 + qfrc_applied + J^T * xfrc_applied
    f = self._joint_dof_field
    return (
      f("qfrc_smooth")
      - f("qfrc_actuator")
      - f("qfrc_applied")
      - f("qfrc_passive")
      + f("qfrc_bias")
    )

  # Pose and velocity component accessors.

  @property
  def root_link_pos_w(self) -> torch.Tensor:
    """Root link position in world frame. Shape (num_envs, 3)."""
    return self.root_link_pose_w[:, 0:3]

  @property
  def root_link_quat_w(self) -> torch.Tensor:
    """Root link quaternion in world frame. Shape (num_envs, 4)."""
    return self.root_link_pose_w[:, 3:7]

  @property
  def root_link_lin_vel_w(self) -> torch.Tensor:
    """Root link linear velocity in world frame. Shape (num_envs, 3)."""
    return self.root_link_vel_w[:, 0:3]

  @property
  def root_link_ang_vel_w(self) -> torch.Tensor:
    """Root link angular velocity in world frame. Shape (num_envs, 3)."""
    return self.data.cvel[:, self.indexing.root_body_id, 0:3]

  @property
  def root_com_pos_w(self) -> torch.Tensor:
    """Root COM position in world frame. Shape (num_envs, 3)."""
    return self.root_com_pose_w[:, 0:3]

  @property
  def root_com_quat_w(self) -> torch.Tensor:
    """Root COM quaternion in world frame. Shape (num_envs, 4)."""
    return self.root_com_pose_w[:, 3:7]

  @property
  def root_com_lin_vel_w(self) -> torch.Tensor:
    """Root COM linear velocity in world frame. Shape (num_envs, 3)."""
    return self.root_com_vel_w[:, 0:3]

  @property
  def root_com_ang_vel_w(self) -> torch.Tensor:
    """Root COM angular velocity in world frame. Shape (num_envs, 3)."""
    # Angular velocity is the same for link and COM frames.
    return self.data.cvel[:, self.indexing.root_body_id, 0:3]

  @property
  def body_link_pos_w(self) -> torch.Tensor:
    """Body link positions in world frame. Shape (num_envs, num_bodies, 3)."""
    return self.body_link_pose_w[..., 0:3]

  @property
  def body_link_quat_w(self) -> torch.Tensor:
    """Body link quaternions in world frame. Shape (num_envs, num_bodies, 4)."""
    return self.body_link_pose_w[..., 3:7]

  @property
  def body_link_lin_vel_w(self) -> torch.Tensor:
    """Body link linear velocities in world frame. Shape (num_envs, num_bodies, 3)."""
    return self.body_link_vel_w[..., 0:3]

  @property
  def body_link_ang_vel_w(self) -> torch.Tensor:
    """Body link angular velocities in world frame. Shape (num_envs, num_bodies, 3)."""
    return self.data.cvel[:, self.indexing.body_sel, 0:3]

  @property
  def body_com_pos_w(self) -> torch.Tensor:
    """Body COM positions in world frame. Shape (num_envs, num_bodies, 3)."""
    return self.body_com_pose_w[..., 0:3]

  @property
  def body_com_quat_w(self) -> torch.Tensor:
    """Body COM quaternions in world frame. Shape (num_envs, num_bodies, 4)."""
    return self.body_com_pose_w[..., 3:7]

  @property
  def body_com_lin_vel_w(self) -> torch.Tensor:
    """Body COM linear velocities in world frame. Shape (num_envs, num_bodies, 3)."""
    return self.body_com_vel_w[..., 0:3]

  @property
  def body_com_ang_vel_w(self) -> torch.Tensor:
    """Body COM angular velocities in world frame. Shape (num_envs, num_bodies, 3)."""
    # Angular velocity is the same for link and COM frames.
    return self.data.cvel[:, self.indexing.body_sel, 0:3]

  @property
  def body_external_force(self) -> torch.Tensor:
    """Body external forces in world frame. Shape (num_envs, num_bodies, 3)."""
    return self.body_external_wrench[..., 0:3]

  @property
  def body_external_torque(self) -> torch.Tensor:
    """Body external torques in world frame. Shape (num_envs, num_bodies, 3)."""
    return self.body_external_wrench[..., 3:6]

  @property
  def geom_pos_w(self) -> torch.Tensor:
    """Geom positions in world frame. Shape (num_envs, num_geoms, 3)."""
    return self.geom_pose_w[..., 0:3]

  @property
  def geom_quat_w(self) -> torch.Tensor:
    """Geom quaternions in world frame. Shape (num_envs, num_geoms, 4)."""
    return self.geom_pose_w[..., 3:7]

  @property
  def geom_lin_vel_w(self) -> torch.Tensor:
    """Geom linear velocities in world frame. Shape (num_envs, num_geoms, 3)."""
    return self.geom_vel_w[..., 0:3]

  @property
  def geom_ang_vel_w(self) -> torch.Tensor:
    """Geom angular velocities in world frame. Shape (num_envs, num_geoms, 3)."""
    body_ids = self.model.geom_bodyid[self.indexing.geom_sel]
    return self.data.cvel[:, body_ids, 0:3]

  @property
  def site_pos_w(self) -> torch.Tensor:
    """Site positions in world frame. Shape (num_envs, num_sites, 3)."""
    return self.site_pose_w[..., 0:3]

  @property
  def site_quat_w(self) -> torch.Tensor:
    """Site quaternions in world frame. Shape (num_envs, num_sites, 4)."""
    return self.site_pose_w[..., 3:7]

  @property
  def site_lin_vel_w(self) -> torch.Tensor:
    """Site linear velocities in world frame. Shape (num_envs, num_sites, 3)."""
    return self.site_vel_w[..., 0:3]

  @property
  def site_ang_vel_w(self) -> torch.Tensor:
    """Site angular velocities in world frame. Shape (num_envs, num_sites, 3)."""
    body_ids = self.model.site_bodyid[self.indexing.site_sel]
    return self.data.cvel[:, body_ids, 0:3]

  # Derived properties.

  @property
  def projected_gravity_b(self) -> torch.Tensor:
    """Gravity vector projected into body frame. Shape (num_envs, 3)."""
    cache = self._derived_cache()
    if (out := cache.get("projected_gravity_b")) is None:
      out = cache["projected_gravity_b"] = quat_apply_inverse(
        self.root_link_quat_w, self.gravity_vec_w
      )
    return out

  @property
  def heading_w(self) -> torch.Tensor:
    """Heading angle in world frame. Shape (num_envs,)."""
    cache = self._derived_cache()
    if (out := cache.get("heading_w")) is None:
      forward_w = quat_apply(self.root_link_quat_w, self.forward_vec_b)
      out = cache["heading_w"] = torch.atan2(forward_w[:, 1], forward_w[:, 0])
    return out

  @property
  def root_link_lin_vel_b(self) -> torch.Tensor:
    """Root link linear velocity in body frame. Shape (num_envs, 3)."""
    cache = self._derived_cache()
    if (out := cache.get("root_link_lin_vel_b")) is None:
      out = cache["root_link_lin_vel_b"] = quat_apply_inverse(
        self.root_link_quat_w, self.root_link_lin_vel_w
      )
    return out

  @property
  def root_link_ang_vel_b(self) -> torch.Tensor:
    """Root link angular velocity in body frame. Shape (num_envs, 3)."""
    cache = self._derived_cache()
    if (out := cache.get("root_link_ang_vel_b")) is None:
      out = cache["root_link_ang_vel_b"] = quat_apply_inverse(
        self.root_link_quat_w, self.root_link_ang_vel_w
      )
    return out

  @property
  def root_com_lin_vel_b(self) -> torch.Tensor:
    """Root COM linear velocity in body frame. Shape (num_envs, 3)."""
    cache = self._derived_cache()
    if (out := cache.get("root_com_lin_vel_b")) is None:
      out = cache["root_com_lin_vel_b"] = quat_apply_inverse(
        self.root_link_quat_w, self.root_com_lin_vel_w
      )
    return out

  @property
  def root_com_ang_vel_b(self) -> torch.Tensor:
    """Root COM angular velocity in body frame. Shape (num_envs, 3)."""
    cache = self._derived_cache()
    if (out := cache.get("root_com_ang_vel_b")) is None:
      out = cache["root_com_ang_vel_b"] = quat_apply_inverse(
        self.root_link_quat_w, self.root_com_ang_vel_w
      )
    return out
