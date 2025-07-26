import torch
from mjlab.entities.indexing import EntityIndexing
import mujoco_warp as mjwarp
from mjlab.utils import math as math_utils


# @dataclass
# class TimestampedBuffer:
#   data: torch.Tensor | None = None
#   timestamp: float = -1.0


class RobotData:
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

    # # Initialize lazy buffers.
    # # link frame wrt world frame.
    # self._root_link_pose_w = TimestampedBuffer()
    # self._root_link_vel_w = TimestampedBuffer()
    # self._body_link_pose_w = TimestampedBuffer()
    # self._body_link_vel_w = TimestampedBuffer()
    # # com frame wrt world frame
    # self._root_com_pose_w = TimestampedBuffer()
    # self._root_com_vel_w = TimestampedBuffer()
    # self._body_com_pose_w = TimestampedBuffer()
    # self._body_com_vel_w = TimestampedBuffer()
    # # joint state
    # self._joint_pos = TimestampedBuffer()
    # self._joint_vel = TimestampedBuffer()
    # self._joint_acc = TimestampedBuffer()

  def update(self, dt: float):
    self._sim_timestamp += dt

  default_root_state: torch.Tensor = None
  default_joint_pos: torch.Tensor = None
  default_joint_vel: torch.Tensor = None
  body_names: list[str] = None
  joint_names: list[str] = None
  joint_pos_target: torch.Tensor = None
  default_joint_pos_limits: torch.Tensor = None
  joint_pos_limits: torch.Tensor = None
  soft_joint_pos_limits: torch.Tensor = None

  # Root properties.

  @property
  def root_link_pose_w(self) -> torch.Tensor:
    """Root link pose in simulation world frame."""
    # if self._root_link_pose_w.timestamp < self._sim_timestamp:
    #   self._root_link_pose_w.data = self.data.qpos[:, self.indexing.free_joint_q_adr]
    # self._root_link_pose_w.timestamp = self._sim_timestamp
    # return self._root_link_pose_w.data
    return self.data.qpos[:, self.indexing.free_joint_q_adr].clone()

  @property
  def root_link_vel_w(self) -> torch.Tensor:
    """Root link velocity in simulation world frame."""
    # if self._root_link_vel_w.timestamp < self._sim_timestamp:
    #   self._root_link_vel_w.data = self.data.qvel[:, self.indexing.free_joint_v_adr]
    #   self._root_link_vel_w.timestamp = self._sim_timestamp
    # return self._root_link_vel_w.data
    return self.data.qvel[:, self.indexing.free_joint_v_adr].clone()

  @property
  def root_com_pose_w(self) -> torch.Tensor:
    """Root center-of-mass pose in simulation world frame."""
    # if self._root_com_pose_w.timestamp < self._sim_timestamp:
    #   pos = self.data.xipos[:, self.indexing.root_body_id]
    #   mat = self.data.ximat[:, self.indexing.root_body_id]
    #   quat = math_utils.quat_from_matrix(mat)
    #   self._root_com_pose_w.data = torch.cat([pos, quat], dim=-1)
    #   self._root_com_pose_w.timestamp = self._sim_timestamp
    # return self._root_com_pose_w.data
    pos = self.data.xipos[:, self.indexing.root_body_id].clone()
    mat = self.data.ximat[:, self.indexing.root_body_id].clone()
    quat = math_utils.quat_from_matrix(mat)
    return torch.cat([pos, quat], dim=-1)

  @property
  def root_com_vel_w(self) -> torch.Tensor:
    """Root center-of-mass velocity [lin_vel, ang_vel] in world frame."""
    # if self._root_com_vel_w.timestamp < self._sim_timestamp:
    #   cvel = self.data.cvel[:, self.indexing.root_body_id]
    #   self._root_com_vel_w = cvel
    #   self._root_com_vel_w.timestamp = self._sim_timestamp
    # return self._root_com_vel_w.data
    return self.data.cvel[:, self.indexing.root_body_id].clone()

  # Body properties.

  @property
  def body_link_pose_w(self) -> torch.Tensor:
    """Body link pose in simulation world frame."""
    # if self._body_link_pose_w.timestamp < self._sim_timestamp:
    #   body_quat = self.data.xquat[:, self.indexing.body_ids]
    #   body_pos = self.data.xpos[:, self.indexing.body_ids]
    #   self._body_link_pose_w.data = torch.cat([body_pos, body_quat], dim=-1)
    #   self._body_link_pose_w.timestamp = self._sim_timestamp
    # return self._body_link_pose_w.data
    body_quat = self.data.xquat[:, self.indexing.body_ids].clone()
    body_pos = self.data.xpos[:, self.indexing.body_ids].clone()
    return torch.cat([body_pos, body_quat], dim=-1)

  @property
  def body_link_vel_w(self):
    """Body link velocity in simulation world frame."""
    # if self._body_link_vel_w.timestamp < self._sim_timestamp:
    #   cvel = self.data.cvel[:, self.indexing.body_ids]  # (num_envs, nbody, 6)
    #   com_vel_lin = cvel[..., 3:6]  # (num_envs, nbody, 3)
    #   com_vel_rot = cvel[..., 0:3]  # (num_envs, nbody, 3)
    #   body_pos = self.data.xpos[:, self.indexing.body_ids]
    #   offset = body_pos - self.data.subtree_com[:, self.indexing.body_root_ids]
    #   lin_vel = com_vel_lin - torch.linalg.cross(offset, com_vel_rot, dim=-1)
    #   ang_vel = com_vel_rot
    #   self._body_link_vel_w.data = torch.cat([lin_vel, ang_vel], dim=-1)
    #   self._body_link_vel_w.timestamp = self._sim_timestamp
    # return self._body_link_vel_w.data
    cvel = self.data.cvel[:, self.indexing.body_ids].clone()  # (num_envs, nbody, 6)
    com_vel_lin = cvel[..., 3:6]  # (num_envs, nbody, 3)
    com_vel_rot = cvel[..., 0:3]  # (num_envs, nbody, 3)
    body_pos = self.data.xpos[:, self.indexing.body_ids].clone()
    offset = body_pos - self.data.subtree_com[:, self.indexing.body_root_ids].clone()
    lin_vel = com_vel_lin - torch.linalg.cross(offset, com_vel_rot, dim=-1)
    ang_vel = com_vel_rot
    return torch.cat([lin_vel, ang_vel], dim=-1)

  @property
  def body_com_pose_w(self):
    """Body center-of-mass pose in simulation world frame."""
    # if self._body_com_pose_w.timestamp < self._sim_timestamp:
    #   com_pos = self.data.xipos[:, self.indexing.body_ids]
    #   com_mat = self.data.ximat[:, self.indexing.body_ids]
    #   com_quat = math_utils.quat_from_matrix(com_mat)
    #   self._body_com_pose_w.data = torch.cat([com_pos, com_quat], dim=-1)
    #   self._body_com_pose_w.timestamp = self._sim_timestamp
    # return self._body_com_pose_w.data
    com_pos = self.data.xipos[:, self.indexing.body_ids].clone()
    com_mat = self.data.ximat[:, self.indexing.body_ids].clone()
    com_quat = math_utils.quat_from_matrix(com_mat)
    return torch.cat([com_pos, com_quat], dim=-1)

  @property
  def body_com_vel_w(self):
    """Body center-of-mass velocity in simulation world frame."""
    # if self._body_com_vel_w.timestamp < self._sim_timestamp:
    #   cvel = self.data.cvel[:, self.indexing.body_ids]  # (num_envs, nbody, 6)
    #   self._body_com_vel_w.data = cvel
    #   self._body_com_vel_w.timestamp = self._sim_timestamp
    # return self._body_com_vel_w.data
    return self.data.cvel[:, self.indexing.body_ids].clone()

  # Joint properties.

  @property
  def joint_pos(self) -> torch.Tensor:
    """Joint positions."""
    # if self._joint_pos.timestamp < self._sim_timestamp:
    #   self._joint_pos.data = self.data.qpos[:, self.indexing.joint_q_adr]
    #   self._joint_pos.timestamp = self._sim_timestamp
    # return self._joint_pos.data
    return self.data.qpos[:, self.indexing.joint_q_adr].clone()

  @property
  def joint_vel(self) -> torch.Tensor:
    """Joint velocities."""
    # if self._joint_vel.timestamp < self._sim_timestamp:
    #   self._joint_vel.data = self.data.qvel[:, self.indexing.joint_v_adr]
    #   self._joint_vel.timestamp = self._sim_timestamp
    # return self._joint_vel.data
    return self.data.qvel[:, self.indexing.joint_v_adr].clone()

  @property
  def joint_acc(self) -> torch.Tensor:
    # if self._joint_acc.timestamp < self._sim_timestamp:
    #   self._joint_acc.data = self.data.qacc[:, self.indexing.joint_v_adr]
    #   self._joint_acc.timestamp = self._sim_timestamp
    # return self._joint_acc.data
    return self.data.qacc[:, self.indexing.joint_v_adr].clone()

  # Sliced properties.

  @property
  def root_link_pos_w(self):
    return self.root_link_pose_w[:, :3]

  @property
  def root_link_quat_w(self):
    return self.root_link_pose_w[:, 3:7]

  @property
  def root_link_lin_vel_w(self):
    return self.root_link_vel_w[:, :3]

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
    return self.root_com_vel_w[:, :3]

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
    """Root center-of-mass linear velocity in base frame."""
    return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

  @property
  def root_com_ang_vel_b(self) -> torch.Tensor:
    # return math_utils.quat_apply_inverse(self.root_link_quat_w, self.root_com_ang_vel_w)
    return self.root_com_ang_vel_w
