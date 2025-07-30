import torch
from mjlab.entities.indexing import EntityIndexing
import mujoco_warp as mjwarp
from mjlab.utils import math as math_utils


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
    """Transform body velocities from MuJoCo's c-frame to world frame coordinates.

    MuJoCo internally computes velocities with respect to "c-frames" - coordinate frames
    located at the center of mass of each kinematic subtree but oriented like the world
    frame. This improves numerical precision for bodies far from the global origin.
    This function transforms these c-frame velocities to world frame velocities.

    The transformation uses rigid body kinematics: for a body with angular velocity ω,
    the linear velocity at any point is related to the linear velocity at the center
    of mass by: v_point = v_com + ω x (r_point - r_com)

    Args:
      pos (torch.Tensor): Body positions in world coordinates.
        Shape: (..., 3)
      subtree_com (torch.Tensor): Center of mass positions of kinematic subtrees
        (c-frame origins) in world coordinates. Shape: (..., 3)
      cvel (torch.Tensor): Body velocities in c-frame coordinates, with angular
        velocity in first 3 components and linear velocity in last 3 components.
        Shape: (..., 6) where [..., 0:3] = angular velocity, [..., 3:6] = linear velocity

    Returns:
      torch.Tensor: Body velocities in world frame coordinates.
        Shape: (..., 6) where [..., 0:3] = linear velocity, [..., 3:6] = angular velocity
        Note: Output format differs from input (linear first, then angular)

    Mathematical Details:
      - offset = pos - subtree_com  (vector from c-frame origin to body position)
      - lin_vel_world = lin_vel_c - offset x ang_vel_c
      - ang_vel_world = ang_vel_c  (unchanged due to same frame orientation)

      The subtraction (rather than addition) in the cross product accounts for
      transforming FROM the center of mass TO the body position.
    """
    offset = pos - subtree_com
    lin_vel_c = cvel[..., 3:6]
    ang_vel_c = cvel[..., 0:3]

    lin_vel_w = lin_vel_c - torch.cross(offset, ang_vel_c, dim=-1)
    ang_vel_w = ang_vel_c

    return torch.cat([lin_vel_w, ang_vel_w], dim=-1)

  def _get_pose_components(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([pos, quat], dim=-1)

  # Root properties

  @property
  def root_link_pose_w(self) -> torch.Tensor:
    """Root link pose in simulation world frame. Shape (num_envs, 7)."""
    pos_w = self.data.xpos[:, self.indexing.root_body_id].clone()
    quat_w = self.data.xquat[:, self.indexing.root_body_id].clone()
    return self._get_pose_components(pos_w, quat_w)

  @property
  def root_link_vel_w(self) -> torch.Tensor:
    """Root link velocity in simulation world frame. Shape (num_envs, 6)."""
    pos = self.data.xpos[:, self.indexing.root_body_id].clone()
    subtree_com = self.data.subtree_com[:, self.indexing.root_body_id].clone()
    cvel = self.data.cvel[:, self.indexing.root_body_id].clone()

    return self._compute_velocity_from_cvel(pos, subtree_com, cvel)

  @property
  def root_com_pose_w(self) -> torch.Tensor:
    """Root center-of-mass pose in simulation world frame. Shape (num_envs, 7)."""
    pos_w = self.data.xipos[:, self.indexing.root_body_id].clone()
    quat = self.data.xquat[:, self.indexing.root_body_id].clone()
    body_iquat = self.indexing.root_body_iquat
    assert body_iquat is not None
    quat_w = math_utils.quat_mul(quat, body_iquat[None])
    return self._get_pose_components(pos_w, quat_w)

  @property
  def root_com_vel_w(self) -> torch.Tensor:
    """Root center-of-mass velocity in world frame. Shape (num_envs, 6)."""
    pos = self.data.xipos[:, self.indexing.root_body_id].clone()
    subtree_com = self.data.subtree_com[:, self.indexing.root_body_id].clone()
    cvel = self.data.cvel[:, self.indexing.root_body_id].clone()

    return self._compute_velocity_from_cvel(pos, subtree_com, cvel)

  # Body properties

  @property
  def body_link_pose_w(self) -> torch.Tensor:
    """Body link pose in simulation world frame. Shape (num_envs, num_bodies, 7)."""
    pos_w = self.data.xpos[:, self.indexing.body_ids].clone()
    quat_w = self.data.xquat[:, self.indexing.body_ids].clone()
    return self._get_pose_components(pos_w, quat_w)

  @property
  def body_link_vel_w(self) -> torch.Tensor:
    """Body link velocity in simulation world frame. Shape (num_envs, num_bodies, 6)."""
    pos = self.data.xpos[:, self.indexing.body_ids].clone()
    subtree_com = self.data.subtree_com[:, self.indexing.body_root_ids].clone()
    cvel = self.data.cvel[:, self.indexing.body_ids].clone()

    return self._compute_velocity_from_cvel(pos, subtree_com, cvel)

  @property
  def body_com_pose_w(self) -> torch.Tensor:
    """Body center-of-mass pose in simulation world frame. Shape (num_envs, num_bodies, 7)."""
    pos_w = self.data.xipos[:, self.indexing.body_ids].clone()
    quat = self.data.xquat[:, self.indexing.body_ids].clone()
    quat_w = math_utils.quat_mul(quat, self.indexing.body_iquats)
    return self._get_pose_components(pos_w, quat_w)

  @property
  def body_com_vel_w(self) -> torch.Tensor:
    """Body center-of-mass velocity in simulation world frame. Shape (num_envs, num_bodies, 6)."""
    pos = self.data.xipos[:, self.indexing.body_ids].clone()
    subtree_com = self.data.subtree_com[:, self.indexing.body_root_ids].clone()
    cvel = self.data.cvel[:, self.indexing.body_ids].clone()

    return self._compute_velocity_from_cvel(pos, subtree_com, cvel)

  # Geom properties

  @property
  def geom_pose_w(self) -> torch.Tensor:
    """Geom pose in simulation world frame. Shape (num_envs, num_geoms, 7)."""
    pos_w = self.data.geom_xpos[:, self.indexing.geom_ids].clone()
    xmat = self.data.geom_xmat[:, self.indexing.geom_ids].clone()
    quat_w = math_utils.quat_from_matrix(xmat)
    return self._get_pose_components(pos_w, quat_w)

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
    return math_utils.quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

  @property
  def heading_w(self) -> torch.Tensor:
    """Robot heading angle in world frame. Shape (num_envs,)."""
    forward_w = math_utils.quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
    return torch.atan2(forward_w[:, 1], forward_w[:, 0])

  @property
  def root_link_lin_vel_b(self) -> torch.Tensor:
    """Root link linear velocity in body frame. Shape (num_envs, 3)."""
    return math_utils.quat_apply_inverse(
      self.root_link_quat_w, self.root_link_lin_vel_w
    )

  @property
  def root_link_ang_vel_b(self) -> torch.Tensor:
    """Root link angular velocity in body frame. Shape (num_envs, 3)."""
    return math_utils.quat_apply_inverse(
      self.root_link_quat_w, self.root_link_ang_vel_w
    )

  @property
  def root_com_lin_vel_b(self) -> torch.Tensor:
    """Root COM linear velocity in body frame. Shape (num_envs, 3)."""
    return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

  @property
  def root_com_ang_vel_b(self) -> torch.Tensor:
    """Root COM angular velocity in body frame. Shape (num_envs, 3)."""
    return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_ang_vel_w)
