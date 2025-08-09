import torch
from mjlab.entities.indexing import EntityIndexing
import mujoco_warp as mjwarp
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  quat_apply,
  quat_apply_inverse,
  quat_mul,
  quat_from_matrix,
)


class RobotData:
  """Robot data interface for MuJoCo simulation data with batched environments."""

  body_names: list[str] | None = None
  geom_names: list[str] | None = None
  site_names: list[str] | None = None
  sensor_names: list[str] | None = None
  joint_names: list[str] | None = None

  default_root_state: torch.Tensor | None = None
  default_joint_pos: torch.Tensor | None = None
  default_joint_vel: torch.Tensor | None = None

  default_joint_pos_limits: torch.Tensor | None = None
  joint_pos_limits: torch.Tensor | None = None
  soft_joint_pos_limits: torch.Tensor | None = None
  joint_pos_weight: torch.Tensor | None = None

  default_joint_stiffness: torch.Tensor = None
  default_joint_damping: torch.Tensor = None
  joint_stiffness: torch.Tensor | None = None
  joint_damping: torch.Tensor | None = None

  def __init__(self, indexing: EntityIndexing, data: mjwarp.Data, device: str):
    self.indexing = indexing
    self.data = data
    self.device = device
    self._sim_timestamp = 0.0

    self.GRAVITY_VEC_W = torch.tensor([0.0, 0.0, -1.0], device=device).repeat(
      data.nworld, 1
    )
    self.FORWARD_VEC_B = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(
      data.nworld, 1
    )

  def update(self, dt: float) -> None:
    self._sim_timestamp += dt

  def _compute_velocity_from_cvel(
    self, pos: torch.Tensor, subtree_com: torch.Tensor, cvel: torch.Tensor
  ) -> torch.Tensor:
    lin_vel_c = cvel[..., 3:6]
    ang_vel_c = cvel[..., 0:3]
    offset = subtree_com - pos
    lin_vel_w = lin_vel_c - torch.cross(ang_vel_c, offset, dim=-1)
    ang_vel_w = ang_vel_c
    return torch.cat([lin_vel_w, ang_vel_w], dim=-1)

  # Root properties

  @property
  def root_link_pose_w(self) -> torch.Tensor:
    """Root link pose in simulation world frame. Shape (num_envs, 7)."""
    pos_w = self.data.xpos[:, self.indexing.root_body_id].clone()  # (num_envs, 3)
    quat_w = self.data.xquat[:, self.indexing.root_body_id].clone()  # (num_envs, 4)
    return torch.cat([pos_w, quat_w], dim=-1)  # (num_envs, 7)

  @property
  def root_link_vel_w(self) -> torch.Tensor:
    """Root link velocity in simulation world frame. Shape (num_envs, 6)."""
    # NOTE: Equivalently, can read this from qvel[:6] but the angular part
    # will be in body frame and needs to be rotated to world frame.
    # Note also that an extra forward() call might be required to make
    # both values equal.
    pos = self.data.xpos[:, self.indexing.root_body_id].clone()  # (num_envs, 3)
    subtree_com = self.data.subtree_com[
      :, self.indexing.body_root_ids[0]
    ].clone()  # (num_envs, 3)
    cvel = self.data.cvel[:, self.indexing.root_body_id].clone()  # (num_envs, 6)
    return self._compute_velocity_from_cvel(pos, subtree_com, cvel)  # (num_envs, 6)

  @property
  def root_com_pose_w(self) -> torch.Tensor:
    """Root center-of-mass pose in simulation world frame. Shape (num_envs, 7)."""
    pos_w = self.data.xipos[:, self.indexing.root_body_id].clone()
    quat = self.data.xquat[:, self.indexing.root_body_id].clone()
    body_iquat = self.indexing.root_body_iquat
    assert body_iquat is not None
    quat_w = quat_mul(quat, body_iquat[None])
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def root_com_vel_w(self) -> torch.Tensor:
    """Root center-of-mass velocity in world frame. Shape (num_envs, 6)."""
    # NOTE: Equivalent sensor is framelinvel/frameangvel with objtype="body".
    pos = self.data.xipos[:, self.indexing.root_body_id].clone()  # (num_envs, 3)
    subtree_com = self.data.subtree_com[
      :, self.indexing.body_root_ids[0]
    ].clone()  # (num_envs, 3)
    cvel = self.data.cvel[:, self.indexing.root_body_id].clone()  # (num_envs, 6)
    return self._compute_velocity_from_cvel(pos, subtree_com, cvel)  # (num_envs, 6)

  # Body properties

  @property
  def body_link_pose_w(self) -> torch.Tensor:
    """Body link pose in simulation world frame. Shape (num_envs, num_bodies, 7)."""
    pos_w = self.data.xpos[:, self.indexing.body_ids].clone()
    quat_w = self.data.xquat[:, self.indexing.body_ids].clone()
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def body_link_vel_w(self) -> torch.Tensor:
    """Body link velocity in simulation world frame. Shape (num_envs, num_bodies, 6)."""
    # NOTE: Equivalent sensor is framelinvel/frameangvel with objtype="xbody".
    pos = self.data.xpos[:, self.indexing.body_ids].clone()  # (num_envs, num_bodies, 3)
    subtree_com = self.data.subtree_com[
      :, self.indexing.body_root_ids
    ].clone()  # (num_envs, num_bodies, 3)
    cvel = self.data.cvel[
      :, self.indexing.body_ids
    ].clone()  # (num_envs, num_bodies, 6)
    return self._compute_velocity_from_cvel(
      pos, subtree_com, cvel
    )  # (num_envs, num_bodies, 6)

  @property
  def body_com_pose_w(self) -> torch.Tensor:
    """Body center-of-mass pose in simulation world frame. Shape (num_envs, num_bodies, 7)."""
    pos_w = self.data.xipos[:, self.indexing.body_ids].clone()
    quat = self.data.xquat[:, self.indexing.body_ids].clone()
    quat_w = quat_mul(quat, self.indexing.body_iquats[None])
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def body_com_vel_w(self) -> torch.Tensor:
    """Body center-of-mass velocity in simulation world frame. Shape (num_envs, num_bodies, 6)."""
    # NOTE: Equivalent sensor is framelinvel/frameangvel with objtype="body".
    pos = self.data.xipos[
      :, self.indexing.body_ids
    ].clone()  # (num_envs, num_bodies, 3)
    subtree_com = self.data.subtree_com[
      :, self.indexing.body_root_ids
    ].clone()  # (num_envs, num_bodies, 3)
    cvel = self.data.cvel[
      :, self.indexing.body_ids
    ].clone()  # (num_envs, num_bodies, 6)
    return self._compute_velocity_from_cvel(
      pos, subtree_com, cvel
    )  # (num_envs, num_bodies, 6)

  # Geom properties

  @property
  def geom_pose_w(self) -> torch.Tensor:
    """Geom pose in simulation world frame. Shape (num_envs, num_geoms, 7)."""
    pos_w = self.data.geom_xpos[:, self.indexing.geom_ids].clone()
    xmat = self.data.geom_xmat[:, self.indexing.geom_ids].clone()
    quat_w = quat_from_matrix(xmat)
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def geom_vel_w(self) -> torch.Tensor:
    """Geom velocity in simulation world frame. Shape (num_envs, num_geoms, 6)."""
    pos = self.data.geom_xpos[:, self.indexing.geom_ids].clone()
    body_ids = self.indexing.geom_body_ids
    root_body_ids = self.indexing.body_root_ids[body_ids - 1]
    subtree_com = self.data.subtree_com[:, root_body_ids].clone()
    cvel = self.data.cvel[:, body_ids].clone()

    return self._compute_velocity_from_cvel(pos, subtree_com, cvel)

  # TODO: Add site properties.

  # Joint properties

  @property
  def joint_pos(self) -> torch.Tensor:
    """Joint positions. Shape (num_envs, nv)"""
    return self.data.qpos[:, self.indexing.joint_q_adr].clone()

  @property
  def joint_vel(self) -> torch.Tensor:
    """Joint velocities. Shape (num_envs, nv)."""
    return self.data.qvel[:, self.indexing.joint_v_adr].clone()

  @property
  def joint_acc(self) -> torch.Tensor:
    """Joint accelerations. Shape (num_envs, nv)."""
    return self.data.qacc[:, self.indexing.joint_v_adr].clone()

  @property
  def joint_torques(self) -> torch.Tensor:
    """Joint torques. Shape (num_envs, nv)."""
    # TODO: Implement this. I think I need a sensor?
    raise NotImplementedError

  @property
  def actuator_force(self) -> torch.Tensor:
    """Scalar actuation force in actuation space. Shape (num_envs, nu)."""
    return self.data.actuator_force.clone()

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

  # Derived properties.

  @property
  def projected_gravity_b(self) -> torch.Tensor:
    """Gravity vector projected into body frame. Shape (num_envs, 3)."""
    return quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

  @property
  def heading_w(self) -> torch.Tensor:
    """Robot heading angle in world frame. Shape (num_envs,)."""
    forward_w = quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
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
