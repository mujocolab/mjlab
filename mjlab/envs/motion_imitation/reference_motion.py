"""Reference motion data."""

import jax
import jax.numpy as jp
import numpy as np
from etils import epath
from flax import struct
from mujoco.mjx._src import math as mjx_math


@struct.dataclass
class ReferenceMotion:
  pos: jp.ndarray
  """Body positions in the world frame."""
  rot: jp.ndarray
  """Body orientations in the world frame."""
  qpos: jp.ndarray
  """Joint positions."""
  qvel: jp.ndarray
  """Joint velocities."""
  linvel: jp.ndarray
  """Body linear velocities in the world frame."""
  angvel: jp.ndarray
  """Body angular velocities in the world frame."""
  com: jp.ndarray
  """Center of mass positions in the world frame."""

  def get(self, step_idx) -> "ReferenceMotion":
    return jax.tree.map(lambda x: x[step_idx], self)

  def __len__(self):
    return self.pos.shape[0]

  @property
  def root_pos(self) -> jax.Array:
    """Root position in the world frame."""
    return self.qpos[..., 0:3]  # (b, 3)

  @property
  def root_quat(self) -> jax.Array:
    """Root orientation expressed as a quaternion."""
    return self.qpos[..., 3:7]  # (b, 4)

  @property
  def root_ori(self) -> jax.Array:
    """Root orientation expressed as a rotation matrix."""
    return mjx_math.quat_to_mat(self.root_quat)

  @property
  def joint_pos(self) -> jax.Array:
    """Joint angles."""
    return self.qpos[..., 7:]  # (b, 29)

  @property
  def joint_vel(self) -> jax.Array:
    """Joint velocities."""
    return self.qvel[..., 6:]  # (b, 29)

  @property
  def root_vel(self) -> jax.Array:
    """Root linear velocity."""
    return self.qvel[..., 0:3]  # (b, 3)

  @property
  def root_angvel(self) -> jax.Array:
    """Root angular velocity."""
    return self.qvel[..., 3:6]  # (b, 3)

  def to_egocentric_frame(self, vec: jax.Array) -> jax.Array:
    return jax.vmap(lambda v: self.root_ori.T @ (v - self.root_pos))(vec)

  @classmethod
  def from_npz(cls, path: epath.Path) -> "ReferenceMotion":
    with np.load(path) as f:
      pos, rot, qpos, qvel, linvel, angvel, com = (
        f["xpos"],
        f["xquat"],
        f["qpos"],
        f["qvel"],
        f["linvel"],
        f["angvel"],
        f["com"],
      )
    return cls(
      pos=jp.array(pos),
      rot=jp.array(rot),
      linvel=jp.array(linvel),
      angvel=jp.array(angvel),
      qpos=jp.array(qpos),
      qvel=jp.array(qvel),
      com=jp.array(com),
    )
