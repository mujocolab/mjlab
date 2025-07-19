import torch


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


@torch.jit.script
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  # store shape
  shape = vec.shape
  # reshape to (N, 3) for multiplication
  quat = quat.reshape(-1, 4)
  vec = vec.reshape(-1, 3)
  # extract components from quaternions
  xyz = quat[:, 1:]
  t = xyz.cross(vec, dim=-1) * 2
  return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
  """Returns torch.sqrt(torch.max(0, x)) but with a zero sub-gradient where x is 0.

  Reference:
      https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L91-L99
  """
  ret = torch.zeros_like(x)
  positive_mask = x > 0
  ret[positive_mask] = torch.sqrt(x[positive_mask])
  return ret


@torch.jit.script
def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
  """Convert rotations given as rotation matrices to quaternions.

  Args:
      matrix: The rotation matrices. Shape is (..., 3, 3).

  Returns:
      The quaternion in (w, x, y, z). Shape is (..., 4).

  Reference:
      https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L102-L161
  """
  if matrix.size(-1) != 3 or matrix.size(-2) != 3:
    raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

  batch_dim = matrix.shape[:-2]
  m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
    matrix.reshape(batch_dim + (9,)), dim=-1
  )

  q_abs = _sqrt_positive_part(
    torch.stack(
      [
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
      ],
      dim=-1,
    )
  )

  # we produce the desired quaternion multiplied by each of r, i, j, k
  quat_by_rijk = torch.stack(
    [
      # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
      torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
      # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
      torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
      # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
      torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
      # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
      torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ],
    dim=-2,
  )

  # We floor here at 0.1 but the exact level is not important; if q_abs is small,
  # the candidate won't be picked.
  flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
  quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

  # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
  # forall i; we pick the best-conditioned one (with the largest denominator)
  return quat_candidates[
    torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
  ].reshape(batch_dim + (4,))


@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  """Apply a quaternion rotation to a vector.

  Args:
      quat: The quaternion in (w, x, y, z). Shape is (..., 4).
      vec: The vector in (x, y, z). Shape is (..., 3).

  Returns:
      The rotated vector in (x, y, z). Shape is (..., 3).
  """
  # store shape
  shape = vec.shape
  # reshape to (N, 3) for multiplication
  quat = quat.reshape(-1, 4)
  vec = vec.reshape(-1, 3)
  # extract components from quaternions
  xyz = quat[:, 1:]
  t = xyz.cross(vec, dim=-1) * 2
  return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def combine_frame_transforms(
  t01: torch.Tensor,
  q01: torch.Tensor,
  t12: torch.Tensor | None = None,
  q12: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  r"""Combine transformations between two reference frames into a stationary frame.

  It performs the following transformation operation: :math:`T_{02} = T_{01} \times T_{12}`,
  where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

  Args:
      t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
      q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
      t12: Position of frame 2 w.r.t. frame 1. Shape is (N, 3).
          Defaults to None, in which case the position is assumed to be zero.
      q12: Quaternion orientation of frame 2 w.r.t. frame 1 in (w, x, y, z). Shape is (N, 4).
          Defaults to None, in which case the orientation is assumed to be identity.

  Returns:
      A tuple containing the position and orientation of frame 2 w.r.t. frame 0.
      Shape of the tensors are (N, 3) and (N, 4) respectively.
  """
  # compute orientation
  if q12 is not None:
    q02 = quat_mul(q01, q12)
  else:
    q02 = q01
  # compute translation
  if t12 is not None:
    t02 = t01 + quat_apply(q01, t12)
  else:
    t02 = t01

  return t02, q02


@torch.jit.script
def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
  r"""Wraps input angles (in radians) to the range :math:`[-\pi, \pi]`.

  This function wraps angles in radians to the range :math:`[-\pi, \pi]`, such that
  :math:`\pi` maps to :math:`\pi`, and :math:`-\pi` maps to :math:`-\pi`. In general,
  odd positive multiples of :math:`\pi` are mapped to :math:`\pi`, and odd negative
  multiples of :math:`\pi` are mapped to :math:`-\pi`.

  The function behaves similar to MATLAB's `wrapToPi <https://www.mathworks.com/help/map/ref/wraptopi.html>`_
  function.

  Args:
      angles: Input angles of any shape.

  Returns:
      Angles in the range :math:`[-\pi, \pi]`.
  """
  # wrap to [0, 2*pi)
  wrapped_angle = (angles + torch.pi) % (2 * torch.pi)
  # map to [-pi, pi]
  # we check for zero in wrapped angle to make it go to pi when input angle is odd multiple of pi
  return torch.where(
    (wrapped_angle == 0) & (angles > 0), torch.pi, wrapped_angle - torch.pi
  )


@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))