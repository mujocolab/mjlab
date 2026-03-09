"""Domain randomization functions for body fields."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_from_matrix,
)

from ._core import (
  _DEFAULT_ASSET_CFG,
  Ranges,
  _get_entity_indices,
  _randomize_model_field,
  _randomize_quat_field,
  _sample_angle,
)
from ._types import Distribution, Operation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_CARDANO_TWO_PI_3 = 2.0943951023931953  # 2π/3

# Pseudo-inertia helpers.


def _cholesky_4x4(A: torch.Tensor) -> torch.Tensor:
  """Analytical Cholesky for batched 4x4 SPD matrices.

  Avoids ``torch.linalg.cholesky`` (and the cuSOLVER library it loads), which allocates
  several GB of persistent GPU memory on first use.

  Args:
    A: ``(*batch, 4, 4)`` symmetric positive-definite matrix.

  Returns:
    L: ``(*batch, 4, 4)`` lower-triangular Cholesky factor.
  """
  L = torch.zeros_like(A)
  L[..., 0, 0] = torch.sqrt(A[..., 0, 0])
  L[..., 1, 0] = A[..., 1, 0] / L[..., 0, 0]
  L[..., 2, 0] = A[..., 2, 0] / L[..., 0, 0]
  L[..., 3, 0] = A[..., 3, 0] / L[..., 0, 0]
  L[..., 1, 1] = torch.sqrt(A[..., 1, 1] - L[..., 1, 0] ** 2)
  L[..., 2, 1] = (A[..., 2, 1] - L[..., 2, 0] * L[..., 1, 0]) / L[..., 1, 1]
  L[..., 3, 1] = (A[..., 3, 1] - L[..., 3, 0] * L[..., 1, 0]) / L[..., 1, 1]
  L[..., 2, 2] = torch.sqrt(A[..., 2, 2] - L[..., 2, 0] ** 2 - L[..., 2, 1] ** 2)
  L[..., 3, 2] = (
    A[..., 3, 2] - L[..., 3, 0] * L[..., 2, 0] - L[..., 3, 1] * L[..., 2, 1]
  ) / L[..., 2, 2]
  L[..., 3, 3] = torch.sqrt(
    A[..., 3, 3] - L[..., 3, 0] ** 2 - L[..., 3, 1] ** 2 - L[..., 3, 2] ** 2
  )
  return L


def _eigh_3x3(
  A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Analytical eigendecomposition for batched 3x3 symmetric matrices.

  Computes eigenvalues via Cardano's formula for the characteristic polynomial,
  then extracts eigenvectors from projector products. Avoids Python-level loops,
  so the number of CUDA kernel launches is fixed regardless of desired precision.

  Args:
    A: ``(*batch, 3, 3)`` symmetric matrix.

  Returns:
    eigenvalues: ``(*batch, 3)`` in ascending order.
    V: ``(*batch, 3, 3)`` orthogonal eigenvectors (columns), ``det(V) = +1``.
  """
  # Upper triangle.
  a00 = A[..., 0, 0]
  a01 = A[..., 0, 1]
  a02 = A[..., 0, 2]
  a11 = A[..., 1, 1]
  a12 = A[..., 1, 2]
  a22 = A[..., 2, 2]

  # Eigenvalues via Cardano's formula.
  # Shift: q = tr(A)/3, B = A - qI (traceless).
  q = (a00 + a11 + a22) / 3
  b00 = a00 - q
  b11 = a11 - q
  b22 = a22 - q

  # p^2 = ||B||_F^2 / 6.
  p_sq = (
    b00 * b00 + b11 * b11 + b22 * b22 + 2 * (a01 * a01 + a02 * a02 + a12 * a12)
  ) / 6
  p = torch.sqrt(torch.clamp(p_sq, min=1e-30))

  # r = det(B/p) / 2.  |r| <= 1 for real symmetric matrices.
  inv_p = 1 / p
  c00 = b00 * inv_p
  c01 = a01 * inv_p
  c02 = a02 * inv_p
  c11 = b11 * inv_p
  c12 = a12 * inv_p
  c22 = b22 * inv_p
  r = (
    c00 * (c11 * c22 - c12 * c12)
    - c01 * (c01 * c22 - c12 * c02)
    + c02 * (c01 * c12 - c11 * c02)
  ) / 2

  phi = torch.acos(torch.clamp(r, -1.0, 1.0)) / 3
  two_p = 2 * p

  # Eigenvalues in ascending order (guaranteed by cosine ordering for 0 <= phi <= pi/3).
  eig0 = q + two_p * torch.cos(phi + _CARDANO_TWO_PI_3)  # smallest
  eig1 = q + two_p * torch.cos(phi - _CARDANO_TWO_PI_3)  # middle
  eig2 = q + two_p * torch.cos(phi)  # largest
  eigenvalues = torch.stack([eig0, eig1, eig2], dim=-1)

  # Eigenvectors via projector products.
  # For eigenvalue lambda_i, prod_{j != i} (A - lambda_j I) has columns proportional to
  # v_i.
  I3 = torch.eye(3, device=A.device, dtype=A.dtype)
  M0 = A - eig0[..., None, None] * I3
  M1 = A - eig1[..., None, None] * I3
  M2 = A - eig2[..., None, None] * I3

  P2 = torch.matmul(M0, M1)  # columns proportional to v2
  P0 = torch.matmul(M1, M2)  # columns proportional to v0

  # Select best (largest-norm) column from each projector.
  def _best_col(P: torch.Tensor) -> torch.Tensor:
    col_sq = (P * P).sum(dim=-2)  # (*batch, 3)
    idx = col_sq.argmax(dim=-1)  # (*batch,)
    idx = idx[..., None, None].expand_as(P[..., :1])  # (*batch, 3, 1)
    return P.gather(-1, idx)[..., 0]  # (*batch, 3)

  v2 = _best_col(P2)
  v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-30)

  v0 = _best_col(P0)
  # Gram-Schmidt against v2 to enforce orthogonality.
  v0 = v0 - (v0 * v2).sum(dim=-1, keepdim=True) * v2
  v0_norm = torch.norm(v0, dim=-1, keepdim=True)

  # Fallback for double degeneracy (lambda_0 ~ lambda_1): pick any unit vector
  # perpendicular to v2.
  abs_v2 = torch.abs(v2)
  min_dim = abs_v2.argmin(dim=-1, keepdim=True)  # (*batch, 1)
  fb = torch.zeros_like(v2).scatter_(-1, min_dim, 1.0)
  fb = fb - (fb * v2).sum(dim=-1, keepdim=True) * v2
  fb = fb / (torch.norm(fb, dim=-1, keepdim=True) + 1e-30)

  need_fb = v0_norm[..., 0] < 1e-10
  v0 = torch.where(
    need_fb[..., None],
    fb,
    v0 / (v0_norm + 1e-30),
  )

  # v1 = v2 x v0 guarantees a right-handed frame (det V = +1).
  v1 = torch.linalg.cross(v2, v0)
  v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-30)

  V = torch.stack([v0, v1, v2], dim=-1)

  # Triple degeneracy (p ~ 0 => A ~ qI): fall back to identity.
  degenerate = p_sq < 1e-20
  V = torch.where(degenerate[..., None, None], I3.expand_as(V), V)

  return eigenvalues, V


def _reconstruct_pseudo_inertia_J(
  mass: torch.Tensor,
  ipos: torch.Tensor,
  inertia: torch.Tensor,
  iquat: torch.Tensor,
) -> torch.Tensor:
  """Build the 4x4 pseudo-inertia matrix J from MuJoCo body fields.

  1. Rotate principal moments into body frame via ``body_iquat``.
  2. Apply the parallel-axis theorem to shift inertia from COM to body origin.

  Args:
    mass: ``(*batch,)``.
    ipos: COM in body frame, ``(*batch, 3)``.
    inertia: Principal moments, ``(*batch, 3)``.
    iquat: Principal-to-body quaternion (wxyz), ``(*batch, 4)``.

  Returns:
    J: ``(*batch, 4, 4)``.
  """
  I3 = torch.eye(3, device=mass.device, dtype=mass.dtype)

  # Rotate principal moments into body frame (body_iquat maps principal->body).
  R = matrix_from_quat(iquat)  # (*batch, 3, 3)
  I_com = R @ torch.diag_embed(inertia) @ R.mT  # (*batch, 3, 3)

  # Parallel-axis theorem: shift inertia from COM to body origin.
  c = ipos
  c_sq = (c * c).sum(dim=-1)  # (*batch,)
  c_outer = c.unsqueeze(-1) * c.unsqueeze(-2)  # (*batch, 3, 3)
  m = mass.unsqueeze(-1).unsqueeze(-1)  # (*batch, 1, 1)
  I_origin = I_com + m * (
    c_sq.unsqueeze(-1).unsqueeze(-1) * I3 - c_outer
  )  # (*batch, 3, 3)

  # sigma = 0.5 * Tr(I_origin) * I3 - I_origin
  trace = I_origin.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (*batch,)
  sigma = 0.5 * trace.unsqueeze(-1).unsqueeze(-1) * I3 - I_origin

  h = mass.unsqueeze(-1) * ipos  # first mass moment

  batch_shape = mass.shape
  J = torch.zeros(*batch_shape, 4, 4, device=mass.device, dtype=mass.dtype)
  J[..., :3, :3] = sigma
  J[..., :3, 3] = h
  J[..., 3, :3] = h
  J[..., 3, 3] = mass
  return J


def _decompose_pseudo_inertia_J(
  J: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Decompose pseudo-inertia matrix to MuJoCo body fields (exact).

  Extracts ``body_mass``, ``body_ipos``, ``body_inertia`` (principal moments), and
  ``body_iquat`` (principal-frame rotation) by diagonalizing the full inertia tensor
  via eigendecomposition. This is exact for any perturbation magnitude, including large
  shear.

  Args:
    J: 4x4 pseudo-inertia matrix, shape ``(*batch, 4, 4)``.

  Returns:
    Tuple of (mass, ipos, inertia, iquat) with shapes ``(*batch,)``,
    ``(*batch, 3)``, ``(*batch, 3)``, ``(*batch, 4)``.
  """
  mass = J[..., 3, 3]
  h = J[..., :3, 3]
  ipos = h / mass.unsqueeze(-1)

  sigma = J[..., :3, :3]
  trace_sigma = sigma.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (*batch,)
  I3 = torch.eye(3, device=J.device, dtype=J.dtype)
  # Invert sigma = 0.5*Tr(I)*I3 - I  =>  I_origin = Tr(sigma)*I3 - sigma
  I_origin = trace_sigma.unsqueeze(-1).unsqueeze(-1) * I3 - sigma  # (*batch, 3, 3)

  # Inverse parallel-axis: shift inertia from body origin back to COM.
  c = ipos
  c_sq = (c * c).sum(dim=-1)  # (*batch,)
  c_outer = c.unsqueeze(-1) * c.unsqueeze(-2)  # (*batch, 3, 3)
  m = mass.unsqueeze(-1).unsqueeze(-1)  # (*batch, 1, 1)
  I_com = I_origin - m * (
    c_sq.unsqueeze(-1).unsqueeze(-1) * I3 - c_outer
  )  # (*batch, 3, 3)

  # Columns of V are principal axes in body frame; eigenvalues are principal moments.
  # _eigh_3x3 guarantees det(V) = +1 (right-handed frame via cross product).
  principal_moments, V = _eigh_3x3(I_com)

  # MuJoCo body_iquat is principal->body, i.e. it represents R = V.
  iquat = quat_from_matrix(V)  # (*batch, 4), wxyz

  return mass, ipos, principal_moments, iquat


def _build_perturbation_U(
  alpha: torch.Tensor,
  d1: torch.Tensor,
  d2: torch.Tensor,
  d3: torch.Tensor,
  s12: torch.Tensor,
  s13: torch.Tensor,
  s23: torch.Tensor,
  t1: torch.Tensor,
  t2: torch.Tensor,
  t3: torch.Tensor,
) -> torch.Tensor:
  """Build the upper-triangular perturbation matrix U from 10 parameters.

  .. code-block::

      U = e^alpha * [[e^d1, s12, s13, t1],
                  [0,   e^d2, s23, t2],
                  [0,   0,   e^d3, t3],
                  [0,   0,   0,    1 ]]

  All arguments have shape ``(*batch,)``.

  Returns:
    U: shape ``(*batch, 4, 4)``.
  """
  scale = torch.exp(alpha)  # (*batch,)
  batch_shape = alpha.shape
  U = torch.zeros(*batch_shape, 4, 4, device=alpha.device, dtype=alpha.dtype)
  U[..., 0, 0] = scale * torch.exp(d1)
  U[..., 0, 1] = scale * s12
  U[..., 0, 2] = scale * s13
  U[..., 0, 3] = scale * t1
  U[..., 1, 1] = scale * torch.exp(d2)
  U[..., 1, 2] = scale * s23
  U[..., 1, 3] = scale * t2
  U[..., 2, 2] = scale * torch.exp(d3)
  U[..., 2, 3] = scale * t3
  U[..., 3, 3] = scale  # e^alpha × 1
  return U


# Per-field functions.


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def body_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "scale",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize body mass. Triggers ``set_const`` recomputation.

  .. warning::

    This function only changes ``body_mass`` and leaves ``body_inertia``
    unchanged. For a uniform density change (the typical DR use case),
    inertia should scale proportionally with mass. Use
    :func:`pseudo_inertia` with ``alpha_range`` instead, which scales both
    correctly. ``body_mass`` alone is only appropriate when modelling a
    point mass added at the COM (which contributes zero inertia).
  """
  warnings.warn(
    "dr.body_mass only randomizes mass and leaves the inertia tensor "
    "unchanged. For a physically consistent density change, use "
    "dr.pseudo_inertia(alpha_range=...) instead, which scales both mass "
    "and inertia together. dr.body_mass is only appropriate when modelling "
    "a point mass added at the COM.",
    UserWarning,
    stacklevel=2,
  )
  _randomize_model_field(
    env,
    env_ids,
    "body_mass",
    entity_type="body",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
  )


@requires_model_fields("body_ipos", recompute=RecomputeLevel.set_const)
def body_com_offset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "add",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize body COM offset (body_ipos). Triggers ``set_const``."""
  _randomize_model_field(
    env,
    env_ids,
    "body_ipos",
    entity_type="body",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2],
  )


# Raw alias.
body_ipos = body_com_offset


@requires_model_fields("body_pos", recompute=RecomputeLevel.set_const_0)
def body_pos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "add",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize body position. Triggers ``set_const_0``."""
  _randomize_model_field(
    env,
    env_ids,
    "body_pos",
    entity_type="body",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2],
  )


@requires_model_fields("body_quat", recompute=RecomputeLevel.set_const_0)
def body_quat(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  roll_range: tuple[float, float] = (0.0, 0.0),
  pitch_range: tuple[float, float] = (0.0, 0.0),
  yaw_range: tuple[float, float] = (0.0, 0.0),
  distribution: Distribution | str = "uniform",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize body orientation by composing an RPY perturbation.

  Ranges are in radians. The sampled perturbation is composed with the default
  quaternion (not the current one), so repeated calls do not accumulate. The result is
  always a valid unit quaternion. Triggers ``set_const_0`` recomputation.
  """
  _randomize_quat_field(
    env,
    env_ids,
    "body_quat",
    entity_type="body",
    roll_range=roll_range,
    pitch_range=pitch_range,
    yaw_range=yaw_range,
    distribution=distribution,
    asset_cfg=asset_cfg,
  )


@requires_model_fields(
  "body_mass",
  "body_ipos",
  "body_inertia",
  "body_iquat",
  recompute=RecomputeLevel.set_const,
)
def pseudo_inertia(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  alpha_range: tuple[float, float] = (0.0, 0.0),
  d_range: tuple[float, float] | None = None,
  d1_range: tuple[float, float] = (0.0, 0.0),
  d2_range: tuple[float, float] = (0.0, 0.0),
  d3_range: tuple[float, float] = (0.0, 0.0),
  s12_range: tuple[float, float] = (0.0, 0.0),
  s13_range: tuple[float, float] = (0.0, 0.0),
  s23_range: tuple[float, float] = (0.0, 0.0),
  t_range: tuple[float, float] | None = None,
  t1_range: tuple[float, float] = (0.0, 0.0),
  t2_range: tuple[float, float] = (0.0, 0.0),
  t3_range: tuple[float, float] = (0.0, 0.0),
  distribution: Distribution | str = "uniform",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  r"""Physics-consistent inertial randomization via the pseudo-inertia matrix.

  Jointly randomizes ``body_mass``, ``body_ipos``, ``body_inertia``, and ``body_iquat``
  while guaranteeing exact physical consistency for any perturbation magnitude.
  Triggers ``set_const`` recomputation.

  The parameterization follows `Rucker & Wensing, 2022
  <https://par.nsf.gov/servlets/purl/10347458>`_: the pseudo-inertia matrix
  :math:`J \succ 0` is factored via
  Cholesky as :math:`J = LL^\top`, then perturbed by an upper-triangular matrix
  U: :math:`J' = (UL)(UL)^\top`. The result is diagonalized via eigendecomposition to
  extract principal moments (``body_inertia``) and principal frame rotation
  (``body_iquat``), so it is exact for any perturbation magnitude.

  The 10 parameters and their physical effects:

  - ``alpha``: global mass-density scale — mass and inertia scale by
    :math:`e^{2\alpha}`, COM unchanged.
  - ``d1, d2, d3``: axis-aligned stretch/compress. Use ``d_range`` as a convenience to
    set all three to the same range.
  - ``s12, s13, s23``: shear in the xy, xz, and yz planes.
  - ``t1, t2, t3``: COM shift along x, y, z axes (in body frame). Use ``t_range`` as a
    convenience to set all three to the same range.

  Args:
    env: The RL environment.
    env_ids: Environment indices to randomize. If ``None``, all envs.
    alpha_range: Range for global mass-density log-scale.
    d_range: Convenience shorthand — sets ``d1_range=d2_range=d3_range``.
    d1_range: Stretch/compress along the x axis.
    d2_range: Stretch/compress along the y axis.
    d3_range: Stretch/compress along the z axis.
    s12_range: Shear in the xy plane.
    s13_range: Shear in the xz plane.
    s23_range: Shear in the yz plane.
    t_range: Convenience shorthand — sets ``t1_range=t2_range=t3_range``.
    t1_range: COM shift along the x axis (body frame).
    t2_range: COM shift along the y axis (body frame).
    t3_range: COM shift along the z axis (body frame).
    distribution: Sampling distribution for all parameters.
    asset_cfg: Asset and body selection.
  """
  if d_range is not None:
    d1_range = d2_range = d3_range = d_range
  if t_range is not None:
    t1_range = t2_range = t3_range = t_range

  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  entity_indices = _get_entity_indices(asset.indexing, asset_cfg, "body", False)
  n_envs = len(env_ids)
  n_bodies = len(entity_indices)
  shape = (n_envs, n_bodies)

  def_mass = env.sim.get_default_field("body_mass")[entity_indices]
  def_ipos = env.sim.get_default_field("body_ipos")[entity_indices]
  def_inertia = env.sim.get_default_field("body_inertia")[entity_indices]
  def_iquat = env.sim.get_default_field("body_iquat")[entity_indices]

  # Reconstruct J_default for each body: (n_bodies, 4, 4).
  J_default = _reconstruct_pseudo_inertia_J(def_mass, def_ipos, def_inertia, def_iquat)

  # Cholesky factor L: (n_bodies, 4, 4), lower triangular.
  L = _cholesky_4x4(J_default)

  # Sample perturbation parameters, each (n_envs, n_bodies).
  def sample(r: tuple[float, float]) -> torch.Tensor:
    return _sample_angle(distribution, r, shape, env.device)

  alpha = sample(alpha_range)
  d1 = sample(d1_range)
  d2 = sample(d2_range)
  d3 = sample(d3_range)
  s12 = sample(s12_range)
  s13 = sample(s13_range)
  s23 = sample(s23_range)
  t1 = sample(t1_range)
  t2 = sample(t2_range)
  t3 = sample(t3_range)

  # Build U: (n_envs, n_bodies, 4, 4), upper triangular.
  U = _build_perturbation_U(alpha, d1, d2, d3, s12, s13, s23, t1, t2, t3)

  # L_new = U @ L, broadcast over envs.
  L_exp = L.unsqueeze(0).expand(n_envs, n_bodies, 4, 4)
  L_new = torch.matmul(U, L_exp)  # (n_envs, n_bodies, 4, 4)

  # New pseudo-inertia: J' = L_new @ L_newᵀ.
  J_new = torch.matmul(L_new, L_new.mT)  # (n_envs, n_bodies, 4, 4)

  # Decompose back to MuJoCo fields via eigendecomposition (exact).
  mass_new, ipos_new, inertia_new, iquat_new = _decompose_pseudo_inertia_J(J_new)

  env_grid, entity_grid = torch.meshgrid(env_ids, entity_indices, indexing="ij")
  env.sim.model.body_mass[env_grid, entity_grid] = mass_new
  env.sim.model.body_ipos[env_grid, entity_grid] = ipos_new
  env.sim.model.body_inertia[env_grid, entity_grid] = inertia_new
  env.sim.model.body_iquat[env_grid, entity_grid] = iquat_new
