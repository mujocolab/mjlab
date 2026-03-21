"""Terrain height sensor for per-frame vertical clearance."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.sensor.raycast_sensor import RayCastData, RayCastSensor, RayCastSensorCfg


@dataclass
class TerrainHeightData(RayCastData):
  """Raycast data extended with per-frame terrain clearance.

  Inherits all fields from :class:`RayCastData` (distances, hit positions, normals,
  frame poses) and adds :attr:`heights`.
  """

  heights: torch.Tensor
  """Vertical clearance per frame. Shape is ``[B, F]`` when a reduction is
  applied, or ``[B, F, N]`` with ``reduction="none"``."""


@dataclass
class TerrainHeightSensorCfg(RayCastSensorCfg):
  """RayCastSensor that reports per-frame vertical clearance above terrain.

  Inherits all raycasting configuration from :class:`RayCastSensorCfg`. The sensor
  computes ``frame_z - hit_z`` for each ray and reduces across rays per frame using
  the chosen :attr:`reduction`.
  """

  reduction: str = "min"
  """How to aggregate rays within each frame: ``"min"``, ``"max"``, ``"mean"``,
  or ``"none"`` (no reduction, returns ``[B, F, N]``).
  Defaults to ``"min"`` (closest terrain point)."""

  def build(self) -> TerrainHeightSensor:
    return TerrainHeightSensor(self)


class TerrainHeightSensor(RayCastSensor):
  """Per-frame vertical clearance above terrain.

  Inherits all raycasting from :class:`RayCastSensor`. Access terrain heights via
  ``sensor.data.heights`` (shape ``[B, F]``).
  """

  cfg: TerrainHeightSensorCfg

  @property
  def data(self) -> TerrainHeightData:
    """Raycast data with per-frame terrain clearance heights."""
    return super().data  # type: ignore[return-value]

  def _compute_data(self) -> TerrainHeightData:
    raw = super()._compute_data()
    F, N = self.num_frames, self.num_rays_per_frame
    B = raw.distances.shape[0]

    frame_z = raw.frame_pos_w[:, :, 2]  # [B, F]
    hit_z = raw.hit_pos_w[:, :, 2].view(B, F, N)  # [B, F, N]
    heights = frame_z.unsqueeze(-1) - hit_z  # [B, F, N]

    miss = raw.distances.view(B, F, N) < 0
    # When all rays for a frame miss there are two cases:
    # 1. Frame is below or at the terrain surface (rays start below and
    #    point down, never hitting anything). True clearance is ~0.
    # 2. Frame is genuinely above max_distance. True clearance >=
    #    max_distance.
    # We distinguish them using frame_z clamped to [0, max_distance].
    # For partial misses (some rays hit, some don't), max_distance is
    # the right fallback since the frame is above terrain.
    all_miss = miss.all(dim=-1, keepdim=True).expand_as(miss)  # [B, F, N]
    fallback = frame_z.unsqueeze(-1).clamp(0, self.cfg.max_distance)
    fallback = fallback.expand_as(heights)  # [B, F, N]
    miss_value = torch.where(all_miss, fallback, self.cfg.max_distance)
    heights = torch.where(miss, miss_value, heights)

    reduction = self.cfg.reduction
    if reduction == "min":
      reduced = heights.min(dim=-1).values
    elif reduction == "max":
      reduced = heights.max(dim=-1).values
    elif reduction == "mean":
      reduced = heights.mean(dim=-1)
    elif reduction == "none":
      reduced = heights
    else:
      raise ValueError(f"Unknown reduction: {reduction!r}")

    return TerrainHeightData(**vars(raw), heights=reduced)
