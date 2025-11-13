from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import mujoco_warp as mjwarp
import torch

from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  quat_apply,
  quat_apply_inverse,
  quat_from_matrix,
  quat_mul,
)

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


@dataclass
class EntityData:
  """Data container for an entity.

  Note: Write methods (write_*) modify state directly. Read properties (e.g.,
  root_link_pose_w) require sim.forward() to be current. If you write then read,
  call sim.forward() in between. Event order matters when mixing reads and writes.
  All inputs/outputs use world frame.
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
    self.data.qpos[env_ids, self.indexing.free_joint_q_adr] = pose

  def write_root_velocity(
    self, velocity: torch.Tensor, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    if self.is_fixed_base:
      raise ValueError("Cannot write root velocity for fixed-base entity.")
    assert velocity.shape[-1] == self.ROOT_VEL_DIM

    env_ids = self._resolve_env_ids(env_ids)
    quat_w = self.data.qpos[env_ids, self.indexing.free_joint_q_adr[3:7]]
    ang_vel_b = quat_apply_inverse(quat_w, velocity[:, 3:])
    velocity_qvel = torch.cat([velocity[:, :3], ang_vel_b], dim=-1)
    self.data.qvel[env_ids, self.indexing.free_joint_v_adr] = velocity_qvel

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
    joint_ids = joint_ids if joint_ids is not None else slice(None)
    q_slice = self.indexing.joint_q_adr[joint_ids]
    self.data.qpos[env_ids, q_slice] = position

  def write_joint_velocity(
    self,
    velocity: torch.Tensor,
    joint_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    if not self.is_articulated:
      raise ValueError("Cannot write joint velocity for non-articulated entity.")

    env_ids = self._resolve_env_ids(env_ids)
    joint_ids = joint_ids if joint_ids is not None else slice(None)
    v_slice = self.indexing.joint_v_adr[joint_ids]
    self.data.qvel[env_ids, v_slice] = velocity

  def write_external_wrench(
    self,
    force: torch.Tensor | None,
    torque: torch.Tensor | None,
    body_ids: Sequence[int] | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    env_ids = self._resolve_env_ids(env_ids)
    local_body_ids = body_ids if body_ids is not None else slice(None)
    global_body_ids = self.indexing.body_ids[local_body_ids]
    if force is not None:
      self.data.xfrc_applied[env_ids, global_body_ids, 0:3] = force
    if torque is not None:
      self.data.xfrc_applied[env_ids, global_body_ids, 3:6] = torque

  def write_ctrl(
    self,
    ctrl: torch.Tensor,
    ctrl_ids: torch.Tensor | slice | None = None,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    if not self.is_actuated:
      raise ValueError("Cannot write control for non-actuated entity.")

    env_ids = self._resolve_env_ids(env_ids)
    local_ctrl_ids = ctrl_ids if ctrl_ids is not None else slice(None)
    global_ctrl_ids = self.indexing.ctrl_ids[local_ctrl_ids]
    self.data.ctrl[env_ids, global_ctrl_ids] = ctrl

  def write_mocap_pose(
    self, pose: torch.Tensor, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    if self.indexing.mocap_id is None:
      raise ValueError("Cannot write mocap pose for non-mocap entity.")
    assert pose.shape[-1] == self.ROOT_POSE_DIM

    env_ids = self._resolve_env_ids(env_ids)
    self.data.mocap_pos[env_ids, self.indexing.mocap_id] = pose[:, 0:3].unsqueeze(1)
    self.data.mocap_quat[env_ids, self.indexing.mocap_id] = pose[:, 3:7].unsqueeze(1)

  def clear_state(self, env_ids: torch.Tensor | slice | None = None) -> None:
    env_ids = self._resolve_env_ids(env_ids)
    v_slice = self.indexing.free_joint_v_adr
    self.data.qfrc_applied[env_ids, v_slice] = 0.0
    self.data.xfrc_applied[env_ids, self.indexing.body_ids] = 0.0

    if self.is_actuated:
      self.joint_pos_target[env_ids] = 0.0
      self.joint_vel_target[env_ids] = 0.0
      self.joint_effort_target[env_ids] = 0.0

  def _resolve_env_ids(
    self, env_ids: torch.Tensor | slice | None
  ) -> torch.Tensor | slice:
    """Convert env_ids to consistent indexing format."""
    if env_ids is None:
      return slice(None)
    if isinstance(env_ids, torch.Tensor):
      return env_ids[:, None]
    return env_ids

  # Root properties

  @property
  def root_link_pose_w(self) -> torch.Tensor:
    """Root link pose in simulation world frame. Shape (num_envs, 7)."""
    pos_w = self.data.xpos[:, self.indexing.root_body_id]  # (num_envs, 3)
    quat_w = self.data.xquat[:, self.indexing.root_body_id]  # (num_envs, 4)
    return torch.cat([pos_w, quat_w], dim=-1)  # (num_envs, 7)

  @property
  def root_link_vel_w(self) -> torch.Tensor:
    """Root link velocity in simulation world frame. Shape (num_envs, 6)."""
    # NOTE: Equivalently, can read this from qvel[:6] but the angular part
    # will be in body frame and needs to be rotated to world frame.
    # Note also that an extra forward() call might be required to make
    # both values equal.
    pos = self.data.xpos[:, self.indexing.root_body_id]  # (num_envs, 3)
    subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
    cvel = self.data.cvel[:, self.indexing.root_body_id]  # (num_envs, 6)
    return compute_velocity_from_cvel(pos, subtree_com, cvel)  # (num_envs, 6)

  @property
  def root_com_pose_w(self) -> torch.Tensor:
    """Root center-of-mass pose in simulation world frame. Shape (num_envs, 7)."""
    pos_w = self.data.xipos[:, self.indexing.root_body_id]
    quat = self.data.xquat[:, self.indexing.root_body_id]
    body_iquat = self.model.body_iquat[:, self.indexing.root_body_id]
    assert body_iquat is not None
    quat_w = quat_mul(quat, body_iquat[None])
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def root_com_vel_w(self) -> torch.Tensor:
    """Root center-of-mass velocity in world frame. Shape (num_envs, 6)."""
    # NOTE: Equivalent sensor is framelinvel/frameangvel with objtype="body".
    pos = self.data.xipos[:, self.indexing.root_body_id]  # (num_envs, 3)
    subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
    cvel = self.data.cvel[:, self.indexing.root_body_id]  # (num_envs, 6)
    return compute_velocity_from_cvel(pos, subtree_com, cvel)  # (num_envs, 6)

  # Body properties

  @property
  def body_link_pose_w(self) -> torch.Tensor:
    """Body link pose in simulation world frame. Shape (num_envs, num_bodies, 7)."""
    pos_w = self.data.xpos[:, self.indexing.body_ids]
    quat_w = self.data.xquat[:, self.indexing.body_ids]
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def body_link_vel_w(self) -> torch.Tensor:
    """Body link velocity in simulation world frame. Shape (num_envs, num_bodies, 6)."""
    # NOTE: Equivalent sensor is framelinvel/frameangvel with objtype="xbody".
    pos = self.data.xpos[:, self.indexing.body_ids]  # (num_envs, num_bodies, 3)
    subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
    cvel = self.data.cvel[:, self.indexing.body_ids]
    return compute_velocity_from_cvel(pos, subtree_com.unsqueeze(1), cvel)

  @property
  def body_com_pose_w(self) -> torch.Tensor:
    """Body center-of-mass pose in simulation world frame. Shape (num_envs, num_bodies, 7)."""
    pos_w = self.data.xipos[:, self.indexing.body_ids]
    quat = self.data.xquat[:, self.indexing.body_ids]
    body_iquat = self.model.body_iquat[:, self.indexing.body_ids]
    quat_w = quat_mul(quat, body_iquat)
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def body_com_vel_w(self) -> torch.Tensor:
    """Body center-of-mass velocity in simulation world frame. Shape (num_envs, num_bodies, 6)."""
    # NOTE: Equivalent sensor is framelinvel/frameangvel with objtype="body".
    pos = self.data.xipos[:, self.indexing.body_ids]
    subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
    cvel = self.data.cvel[:, self.indexing.body_ids]
    return compute_velocity_from_cvel(pos, subtree_com.unsqueeze(1), cvel)

  @property
  def body_external_wrench(self) -> torch.Tensor:
    """Body external wrench in world frame. Shape (num_envs, num_bodies, 6)."""
    return self.data.xfrc_applied[:, self.indexing.body_ids]

  # Geom properties

  @property
  def geom_pose_w(self) -> torch.Tensor:
    """Geom pose in simulation world frame. Shape (num_envs, num_geoms, 7)."""
    pos_w = self.data.geom_xpos[:, self.indexing.geom_ids]
    xmat = self.data.geom_xmat[:, self.indexing.geom_ids]
    quat_w = quat_from_matrix(xmat)
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def geom_vel_w(self) -> torch.Tensor:
    """Geom velocity in simulation world frame. Shape (num_envs, num_geoms, 6)."""
    pos = self.data.geom_xpos[:, self.indexing.geom_ids]
    body_ids = self.model.geom_bodyid[self.indexing.geom_ids]  # (num_geoms,)
    subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
    cvel = self.data.cvel[:, body_ids]
    return compute_velocity_from_cvel(pos, subtree_com.unsqueeze(1), cvel)

  # Site properties

  @property
  def site_pose_w(self) -> torch.Tensor:
    """Site pose in simulation world frame. Shape (num_envs, num_sites, 7)."""
    pos_w = self.data.site_xpos[:, self.indexing.site_ids]
    mat_w = self.data.site_xmat[:, self.indexing.site_ids]
    quat_w = quat_from_matrix(mat_w)
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def site_vel_w(self) -> torch.Tensor:
    """Site velocity in simulation world frame. Shape (num_envs, num_sites, 6)."""
    pos = self.data.site_xpos[:, self.indexing.site_ids]
    body_ids = self.model.site_bodyid[self.indexing.site_ids]  # (num_sites,)
    subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
    cvel = self.data.cvel[:, body_ids]
    return compute_velocity_from_cvel(pos, subtree_com.unsqueeze(1), cvel)

  # Joint properties

  @property
  def joint_pos(self) -> torch.Tensor:
    """Joint positions. Shape (num_envs, nv)"""
    return self.data.qpos[:, self.indexing.joint_q_adr]

  @property
  def joint_vel(self) -> torch.Tensor:
    """Joint velocities. Shape (num_envs, nv)."""
    return self.data.qvel[:, self.indexing.joint_v_adr]

  @property
  def joint_acc(self) -> torch.Tensor:
    """Joint accelerations. Shape (num_envs, nv)."""
    return self.data.qacc[:, self.indexing.joint_v_adr]

  @property
  def joint_torques(self) -> torch.Tensor:
    """Joint torques. Shape (num_envs, nv)."""
    raise NotImplementedError(
      "Joint torques are not currently available. "
      "Consider using 'actuator_force' property for actuation forces, "
      "or 'generalized_force' property for generalized forces applied to the DoFs."
    )

  @property
  def actuator_force(self) -> torch.Tensor:
    """Scalar actuation force in actuation space. Shape (num_envs, nu)."""
    return self.data.actuator_force[:, self.indexing.ctrl_ids]

  @property
  def generalized_force(self) -> torch.Tensor:
    """Generalized forces applied to the DoFs. Shape (num_envs, nv)."""
    return self.data.qfrc_applied[:, self.indexing.free_joint_v_adr]

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
    return self.root_link_vel_w[:, 3:6]

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
    return self.root_com_vel_w[:, 3:6]

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
    return self.body_link_vel_w[..., 3:6]

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
    return self.body_com_vel_w[..., 3:6]

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
    return self.geom_vel_w[..., 3:6]

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
    return self.site_vel_w[..., 3:6]

  # Derived properties.

  @property
  def projected_gravity_b(self) -> torch.Tensor:
    """Gravity vector projected into body frame. Shape (num_envs, 3)."""
    return quat_apply_inverse(self.root_link_quat_w, self.gravity_vec_w)

  @property
  def heading_w(self) -> torch.Tensor:
    """Heading angle in world frame. Shape (num_envs,)."""
    forward_w = quat_apply(self.root_link_quat_w, self.forward_vec_b)
    return torch.atan2(forward_w[:, 1], forward_w[:, 0])

  @property
  def root_link_lin_vel_b(self) -> torch.Tensor:
    """Root link linear velocity in body frame. Shape (num_envs, 3)."""
    return quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

  @property
  def root_link_ang_vel_b(self) -> torch.Tensor:
    """Root link angular velocity in body frame. Shape (num_envs, 3)."""
    return quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

  @property
  def root_com_lin_vel_b(self) -> torch.Tensor:
    """Root COM linear velocity in body frame. Shape (num_envs, 3)."""
    return quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

  @property
  def root_com_ang_vel_b(self) -> torch.Tensor:
    """Root COM angular velocity in body frame. Shape (num_envs, 3)."""
    return quat_apply_inverse(self.root_link_quat_w, self.root_com_ang_vel_w)
