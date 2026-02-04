"""Tests for flat patch sampling on heightfield terrains."""

import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.terrains.heightfield_terrains import HfRandomUniformTerrainCfg
from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg
from mjlab.terrains.utils import find_flat_patches_from_heightfield


@pytest.fixture
def rng() -> np.random.Generator:
  return np.random.default_rng(42)


def test_patches_avoid_step_boundary(rng: np.random.Generator):
  """Patches must not land near a sharp height discontinuity."""
  heights = np.zeros((80, 80), dtype=np.float64)
  # Step along column axis → appears on MuJoCo x-axis.
  heights[:, 40:] = 1.0
  patch_radius = 0.3
  cfg = FlatPatchSamplingCfg(
    num_patches=50, patch_radius=patch_radius, max_height_diff=0.05
  )
  patches = find_flat_patches_from_heightfield(
    heights=heights, horizontal_scale=0.1, z_offset=0.0, cfg=cfg, rng=rng
  )
  assert patches.shape == (50, 3)
  for i in range(len(patches)):
    x, z = patches[i, 0], patches[i, 2]
    # Step is at x=4.0; no patch center should be within patch_radius of it.
    assert x < 3.7 or x >= 4.3, f"Patch {i} at x={x:.2f} too close to step"
    # z must be consistent with the side the patch landed on.
    if x < 3.7:
      assert abs(z) < 0.1, f"Low-x patch should have z~0, got {z}"
    else:
      assert abs(z - 1.0) < 0.1, f"High-x patch should have z~1, got {z}"


def test_range_constraints(rng: np.random.Generator):
  """x/y range filters restrict where patches can land."""
  heights = np.zeros((80, 80), dtype=np.float64)
  cfg = FlatPatchSamplingCfg(
    num_patches=10,
    patch_radius=0.3,
    max_height_diff=0.1,
    x_range=(2.0, 6.0),
    y_range=(2.0, 6.0),
  )
  patches = find_flat_patches_from_heightfield(
    heights=heights, horizontal_scale=0.1, z_offset=0.0, cfg=cfg, rng=rng
  )
  assert np.all(patches[:, 0] >= 2.0) and np.all(patches[:, 0] <= 6.0)
  assert np.all(patches[:, 1] >= 2.0) and np.all(patches[:, 1] <= 6.0)


def test_fallback_when_no_valid_patches(rng: np.random.Generator):
  """When nothing is flat, all patches fall back to the terrain center."""
  # Linear ramp along rows — every footprint spans a large height range.
  heights = np.arange(80).reshape(80, 1).repeat(80, axis=1).astype(np.float64)
  cfg = FlatPatchSamplingCfg(num_patches=5, patch_radius=0.3, max_height_diff=0.001)
  patches = find_flat_patches_from_heightfield(
    heights=heights, horizontal_scale=0.1, z_offset=0.0, cfg=cfg, rng=rng
  )
  center_x = 80 * 0.1 / 2.0
  center_y = 80 * 0.1 / 2.0
  np.testing.assert_allclose(patches[:, 0], center_x)
  np.testing.assert_allclose(patches[:, 1], center_y)


def test_patches_respect_edge_margin(rng: np.random.Generator):
  """No patch center should be within patch_radius of the heightfield edge."""
  heights = np.zeros((80, 80), dtype=np.float64)
  patch_radius = 0.5
  h_scale = 0.1
  cfg = FlatPatchSamplingCfg(
    num_patches=200, patch_radius=patch_radius, max_height_diff=0.1
  )
  patches = find_flat_patches_from_heightfield(
    heights=heights, horizontal_scale=h_scale, z_offset=0.0, cfg=cfg, rng=rng
  )
  terrain_x = 80 * h_scale
  terrain_y = 80 * h_scale
  margin = patch_radius
  # Every patch center must be at least patch_radius from each edge.
  assert np.all(patches[:, 0] >= margin - h_scale)
  assert np.all(patches[:, 0] <= terrain_x - margin + h_scale)
  assert np.all(patches[:, 1] >= margin - h_scale)
  assert np.all(patches[:, 1] <= terrain_y - margin + h_scale)


def test_grid_resolution(rng: np.random.Generator):
  """Finer grid_resolution still produces correct flat patches on a step."""
  heights = np.zeros((80, 80), dtype=np.float64)
  heights[:, 40:] = 1.0
  cfg = FlatPatchSamplingCfg(
    num_patches=30,
    patch_radius=0.3,
    max_height_diff=0.05,
    grid_resolution=0.025,
  )
  patches = find_flat_patches_from_heightfield(
    heights=heights, horizontal_scale=0.1, z_offset=0.0, cfg=cfg, rng=rng
  )
  assert patches.shape == (30, 3)
  for i in range(len(patches)):
    x, z = patches[i, 0], patches[i, 2]
    assert x < 3.7 or x >= 4.3, f"Patch {i} at x={x:.2f} too close to step"
    if x < 3.7:
      assert abs(z) < 0.1
    else:
      assert abs(z - 1.0) < 0.1


def test_terrain_importer_end_to_end():
  """Flat patches flow through generator → importer with correct shape and bounds."""
  device = get_test_device()
  terrain_size = (4.0, 4.0)
  num_patches = 5
  patch_cfg = FlatPatchSamplingCfg(
    num_patches=num_patches, patch_radius=0.3, max_height_diff=0.1
  )
  gen_cfg = TerrainGeneratorCfg(
    seed=123,
    size=terrain_size,
    num_rows=2,
    num_cols=2,
    sub_terrains={
      "rough": HfRandomUniformTerrainCfg(
        proportion=1.0,
        noise_range=(0.02, 0.08),
        noise_step=0.01,
        flat_patch_sampling={"spawn": patch_cfg},
      ),
    },
  )
  importer_cfg = TerrainImporterCfg(
    terrain_type="generator", terrain_generator=gen_cfg, num_envs=4
  )
  importer = TerrainImporter(importer_cfg, device=device)

  assert "spawn" in importer.flat_patches
  patches = importer.flat_patches["spawn"]
  assert patches.shape == (2, 2, num_patches, 3)
  assert patches.dtype == torch.float

  # Every patch should be within the world-space extent of the full terrain grid.
  # Grid is centered at origin → spans [-total/2, +total/2].
  half_x = gen_cfg.num_rows * terrain_size[0] / 2
  half_y = gen_cfg.num_cols * terrain_size[1] / 2
  pts = patches.cpu().numpy().reshape(-1, 3)
  assert np.all(pts[:, 0] >= -half_x) and np.all(pts[:, 0] <= half_x)
  assert np.all(pts[:, 1] >= -half_y) and np.all(pts[:, 1] <= half_y)
