import torch
from mjlab.entities.indexing import EntityIndexing
import mujoco_warp as mjwarp
from mjlab.utils import math as math_utils


class RobotData:
  body_names: list[str] = None
  geom_names: list[str] = None
  site_names: list[str] = None
  sensor_names: list[str] = None
  joint_names: list[str] = None

  default_root_state: torch.Tensor = None
  default_joint_pos: torch.Tensor = None
  default_joint_vel: torch.Tensor = None

  default_joint_pos_limits: torch.Tensor = None
  joint_pos_limits: torch.Tensor = None
  soft_joint_pos_limits: torch.Tensor = None

  def __init__(self, indexing: EntityIndexing, data: mjwarp.Data, device: str):
    self.indexing = indexing
    self.data = data
    self.device = device
    self._sim_timestamp = 0.0

    self.GRAVITY_VEC_W = torch.tensor(data=(0.0, 0.0, -1.0), device=device).repeat(
      data.nworld, 1
    )
    self.FORWARD_VEC_B = torch.tensor(data=(1.0, 0.0, 0.0), device=device).repeat(
      data.nworld, 1
    )

  def update(self, dt: float) -> None:
    self._sim_timestamp += dt

  # Root properties.

  @property
  def root_link_pose_w(self) -> torch.Tensor:
    """Root link pose in simulation world frame. Shape (num_envs, 7)."""
    pos_w = self.data.xpos[:, self.indexing.root_body_id].clone()
    quat_w = self.data.xquat[:, self.indexing.root_body_id].clone()
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def root_link_vel_w(self) -> torch.Tensor:
    """Root link velocity in simulation world frame. Shape (num_envs, 6)."""
    pos = self.data.xpos[:, self.indexing.root_body_id].clone()
    subtree_com = self.data.subtree_com[:, self.indexing.root_body_id].clone()
    offset = pos - subtree_com
    vel_c = self.data.cvel[:, self.indexing.root_body_id].clone()
    lin_vel_c = vel_c[:, 3:6]
    ang_vel_c = vel_c[:, 0:3]
    lin_vel_w = lin_vel_c - torch.cross(offset, ang_vel_c, dim=-1)
    ang_vel_w = ang_vel_c
    return torch.cat([lin_vel_w, ang_vel_w], dim=-1)

  @property
  def root_com_pose_w(self) -> torch.Tensor:
    """Root center-of-mass pose in simulation world frame. Shape (num_envs, 7)."""
    pos_w = self.data.xipos[:, self.indexing.root_body_id].clone()
    mat_w = self.data.ximat[:, self.indexing.root_body_id].clone()
    quat_w = math_utils.quat_from_matrix(mat_w)
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def root_com_vel_w(self) -> torch.Tensor:
    """Root center-of-mass velocity in world frame. Shape (num_envs, 6)."""
    pos = self.data.xipos[:, self.indexing.root_body_id].clone()
    subtree_com = self.data.subtree_com[:, self.indexing.root_body_id].clone()
    offset = pos - subtree_com
    vel_c = self.data.cvel[:, self.indexing.root_body_id].clone()
    lin_vel_c = vel_c[:, 3:6]
    ang_vel_c = vel_c[:, 0:3]
    lin_vel_w = lin_vel_c - torch.cross(offset, ang_vel_c, dim=-1)
    ang_vel_w = ang_vel_c
    return torch.cat([lin_vel_w, ang_vel_w], dim=-1)

  # Body properties.

  @property
  def body_link_pose_w(self) -> torch.Tensor:
    """Body link pose in simulation world frame. Shape (num_envs, num_bodies, 7)."""
    pos_w = self.data.xpos[:, self.indexing.body_ids].clone()
    quat_w = self.data.xquat[:, self.indexing.body_ids].clone()
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def body_link_vel_w(self):
    """Body link velocity in simulation world frame. Shape (num_envs, num_bodies, 6)."""
    pos = self.data.xpos[:, self.indexing.body_ids].clone()
    subtree_com = self.data.subtree_com[:, self.indexing.body_ids].clone()
    offset = pos - subtree_com
    vel_c = self.data.cvel[:, self.indexing.body_ids].clone()
    lin_vel_c = vel_c[..., 3:6]
    ang_vel_c = vel_c[..., 0:3]
    lin_vel_w = lin_vel_c - torch.cross(offset, ang_vel_c, dim=-1)
    ang_vel_w = ang_vel_c
    return torch.cat([lin_vel_w, ang_vel_w], dim=-1)

  @property
  def body_com_pose_w(self):
    """Body center-of-mass pose in simulation world frame."""
    pos_w = self.data.xipos[:, self.indexing.body_ids].clone()
    mat_w = self.data.ximat[:, self.indexing.body_ids].clone()
    quat_w = math_utils.quat_from_matrix(mat_w)
    return torch.cat([pos_w, quat_w], dim=-1)

  @property
  def body_com_vel_w(self):
    """Body center-of-mass velocity in simulation world frame."""
    pos = self.data.xipos[:, self.indexing.body_ids].clone()
    subtree_com = self.data.subtree_com[:, self.indexing.body_ids].clone()
    offset = pos - subtree_com
    vel_c = self.data.cvel[:, self.indexing.body_ids].clone()
    lin_vel_c = vel_c[..., 3:6]
    ang_vel_c = vel_c[..., 0:3]
    lin_vel_w = lin_vel_c - torch.cross(offset, ang_vel_c, dim=-1)
    ang_vel_w = ang_vel_c
    return torch.cat([lin_vel_w, ang_vel_w], dim=-1)

  # Joint properties.

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

  # Sliced properties.

  @property
  def root_link_pos_w(self):
    return self.root_link_pose_w[:, 0:3]

  @property
  def root_link_quat_w(self):
    return self.root_link_pose_w[:, 3:7]

  @property
  def root_link_lin_vel_w(self):
    return self.root_link_vel_w[:, 0:3]

  @property
  def root_link_ang_vel_w(self):
    return self.root_link_vel_w[:, 3:6]

  @property
  def body_link_pos_w(self) -> torch.Tensor:
    return self.body_link_pose_w[..., 0:3]

  @property
  def body_link_quat_w(self) -> torch.Tensor:
    return self.body_link_pose_w[..., 3:7]

  @property
  def body_link_lin_vel_w(self) -> torch.Tensor:
    return self.body_link_vel_w[..., 0:3]

  @property
  def body_link_ang_vel_w(self) -> torch.Tensor:
    return self.body_link_vel_w[..., 3:6]

  @property
  def body_com_pos_w(self):
    return self.body_com_pose_w[..., 0:3]

  @property
  def body_com_quat_w(self):
    return self.body_com_pose_w[..., 3:7]

  @property
  def body_com_lin_vel_w(self):
    return self.body_com_vel_w[..., 0:3]

  @property
  def body_com_ang_vel_w(self):
    return self.body_com_vel_w[..., 3:6]

  @property
  def root_com_lin_vel_w(self) -> torch.Tensor:
    return self.root_com_vel_w[:, 0:3]

  @property
  def root_com_ang_vel_w(self) -> torch.Tensor:
    return self.root_com_vel_w[:, 3:6]

  # Derived properties.

  @property
  def projected_gravity_b(self) -> torch.Tensor:
    return math_utils.quat_apply_inverse(self.root_link_quat_w, self.GRAVITY_VEC_W)

  @property
  def heading_w(self) -> torch.Tensor:
    forward_w = math_utils.quat_apply(self.root_link_quat_w, self.FORWARD_VEC_B)
    return torch.atan2(forward_w[:, 1], forward_w[:, 0])

  @property
  def root_link_lin_vel_b(self) -> torch.Tensor:
    return math_utils.quat_apply_inverse(
      self.root_link_quat_w, self.root_link_lin_vel_w
    )

  @property
  def root_link_ang_vel_b(self) -> torch.Tensor:
    return self.root_link_ang_vel_w

  @property
  def root_com_lin_vel_b(self) -> torch.Tensor:
    return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

  @property
  def root_com_ang_vel_b(self) -> torch.Tensor:
    return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_ang_vel_w)
