"""Terrain normal estimation from raycast hit points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def fit_terrain_normal(
  points: torch.Tensor,
  valid_mask: torch.Tensor,
) -> torch.Tensor:
  """Fit a plane normal from 3D points via covariance eigendecomposition.

  Args:
    points: [B, N, 3] world-frame positions.
    valid_mask: [B, N] boolean (True = valid).

  Returns:
    [B, 3] unit normal oriented upward. Falls back to [0, 0, 1]
    when fewer than 3 valid points.
  """
  B = points.shape[0]
  device = points.device

  count = valid_mask.sum(dim=1)
  enough = count >= 3

  mask_f = valid_mask.float().unsqueeze(-1)
  masked_points = points * mask_f
  count_clamped = count.clamp(min=1).float().unsqueeze(-1)
  centroid = masked_points.sum(dim=1) / count_clamped
  centered = (points - centroid.unsqueeze(1)) * mask_f

  cov = torch.einsum("bni,bnj->bij", centered, centered)
  eigenvalues, eigenvectors = torch.linalg.eigh(cov)
  normal = eigenvectors[:, :, 0]  # Smallest eigenvalue = plane normal.
  normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-8)

  # Orient upward.
  normal = torch.where((normal[:, 2] < 0).unsqueeze(-1), -normal, normal)

  # Fall back when the fit is degenerate (collinear or near-duplicate
  # points). A valid plane has one small eigenvalue and two materially
  # larger ones; line-like or point-like clouds do not.
  eps = torch.finfo(eigenvalues.dtype).eps
  plane_like = (eigenvalues[:, 0] / eigenvalues[:, 1].clamp(min=eps)) < 0.1
  has_spread = eigenvalues[:, 1] > eigenvalues[:, 2].clamp(min=eps) * 1e-6
  reliable = enough & plane_like & has_spread

  up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, 3)
  return torch.where(reliable.unsqueeze(-1), normal, up)


# Cached subsample indices to avoid per-step allocation.
_subsample_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _subsample_indices(
  total: int, max_points: int, device: torch.device
) -> torch.Tensor:
  key = (total, max_points, device)
  if key not in _subsample_cache:
    _subsample_cache[key] = torch.linspace(
      0, total - 1, max_points, device=device
    ).long()
  return _subsample_cache[key]


def terrain_normal_from_sensors(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  max_points: int = 32,
) -> torch.Tensor:
  """Estimate terrain normal from one or more raycast sensors.

  Gathers hit positions, subsamples to *max_points* per sensor, and fits a plane
  normal via :func:`fit_terrain_normal`.

  Returns:
    [B, 3] unit terrain normal in world frame.
  """
  all_points: list[torch.Tensor] = []
  all_valid: list[torch.Tensor] = []

  for name in sensor_names:
    sensor = env.scene[name]
    if not isinstance(sensor, RayCastSensor):
      raise TypeError(
        f"Sensor '{name}' is {type(sensor).__name__}, expected RayCastSensor."
      )

    hit_pos = sensor.data.hit_pos_w
    valid = sensor.data.distances >= 0

    N = hit_pos.shape[1]
    if N > max_points:
      idx = _subsample_indices(N, max_points, hit_pos.device)
      hit_pos = hit_pos[:, idx]
      valid = valid[:, idx]

    all_points.append(hit_pos)
    all_valid.append(valid)

  points = torch.cat(all_points, dim=1)
  valid_mask = torch.cat(all_valid, dim=1)
  return fit_terrain_normal(points, valid_mask)
