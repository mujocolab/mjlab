import os
import numpy as np
import random
import torch
import warp as wp


def seed_rng(seed: int, torch_deterministic: bool = False) -> None:
  np.random.seed(seed)
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)

  wp.rand_init(seed)

  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  if torch_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
  else:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def sample_uniform(
  lower: torch.Tensor | float,
  upper: torch.Tensor | float,
  size: int | tuple[int, ...],
  device: str,
) -> torch.Tensor:
  """Sample uniformly within a range.

  Args:
      lower: Lower bound of uniform range.
      upper: Upper bound of uniform range.
      size: The shape of the tensor.
      device: Device to create tensor on.

  Returns:
      Sampled tensor. Shape is based on :attr:`size`.
  """
  # convert to tuple
  if isinstance(size, int):
    size = (size,)
  # return tensor
  return torch.rand(*size, device=device) * (upper - lower) + lower


@torch.jit.script
def quat_from_euler_xyz(
  roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
  """Convert rotations given as Euler angles in radians to Quaternions.

  Note:
      The euler angles are assumed in XYZ convention.

  Args:
      roll: Rotation around x-axis (in radians). Shape is (N,).
      pitch: Rotation around y-axis (in radians). Shape is (N,).
      yaw: Rotation around z-axis (in radians). Shape is (N,).

  Returns:
      The quaternion in (w, x, y, z). Shape is (N, 4).
  """
  cy = torch.cos(yaw * 0.5)
  sy = torch.sin(yaw * 0.5)
  cr = torch.cos(roll * 0.5)
  sr = torch.sin(roll * 0.5)
  cp = torch.cos(pitch * 0.5)
  sp = torch.sin(pitch * 0.5)
  # compute quaternion
  qw = cy * cr * cp + sy * sr * sp
  qx = cy * sr * cp - sy * cr * sp
  qy = cy * cr * sp + sy * sr * cp
  qz = sy * cr * cp - cy * sr * sp

  return torch.stack([qw, qx, qy, qz], dim=-1)


@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Multiply two quaternions together.

  Args:
      q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
      q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

  Returns:
      The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

  Raises:
      ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
  """
  # check input is correct
  if q1.shape != q2.shape:
    msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
    raise ValueError(msg)
  # reshape to (N, 4) for multiplication
  shape = q1.shape
  q1 = q1.reshape(-1, 4)
  q2 = q2.reshape(-1, 4)
  # extract components from quaternions
  w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
  w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
  # perform multiplication
  ww = (z1 + x1) * (x2 + y2)
  yy = (w1 - y1) * (w2 + z2)
  zz = (w1 + y1) * (w2 - z2)
  xx = ww + yy + zz
  qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
  w = qq - ww + (z1 - y1) * (y2 - z2)
  x = qq - xx + (x1 + w1) * (x2 + w2)
  y = qq - yy + (w1 - x1) * (y2 + z2)
  z = qq - zz + (z1 + y1) * (w2 - x2)

  return torch.stack([w, x, y, z], dim=-1).view(shape)
