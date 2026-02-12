"""Tests for bugs found in PR #609.

Bug 1 (safety checks): BoxPyramidStairsTerrainCfg and
  BoxInvertedPyramidStairsTerrainCfg can produce zero or negative geom
  sizes when border_width is large.

Bug 2 (border_width validation): HfPerlinNoiseTerrainCfg validates
  against `resolution` instead of `horizontal_scale`, inconsistent
  with every other heightfield terrain class.
"""

import mujoco
import numpy as np
import pytest

from mjlab.terrains.heightfield_terrains import HfPerlinNoiseTerrainCfg
from mjlab.terrains.primitive_terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)


@pytest.fixture
def rng() -> np.random.Generator:
  return np.random.default_rng(42)


def test_pyramid_stairs_large_border_no_crash(
  rng: np.random.Generator,
):
  """Large border_width should not crash with negative geom sizes."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")

  cfg = BoxPyramidStairsTerrainCfg(
    proportion=1.0,
    size=(8.0, 8.0),
    step_height_range=(0.0, 0.2),
    step_width=0.3,
    platform_width=3.0,
    border_width=3.5,
  )

  output = cfg.function(difficulty=1.0, spec=spec, rng=rng)
  spec.compile()
  assert len(output.geometries) > 0


def test_inverted_pyramid_stairs_large_border_no_crash(
  rng: np.random.Generator,
):
  """Same test for inverted variant."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")

  cfg = BoxInvertedPyramidStairsTerrainCfg(
    proportion=1.0,
    size=(8.0, 8.0),
    step_height_range=(0.0, 0.2),
    step_width=0.3,
    platform_width=3.0,
    border_width=3.5,
  )

  output = cfg.function(difficulty=1.0, spec=spec, rng=rng)
  spec.compile()
  assert len(output.geometries) > 0


def test_pyramid_stairs_platform_matches_config(
  rng: np.random.Generator,
):
  """The center platform should be at least platform_width wide."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")

  cfg = BoxPyramidStairsTerrainCfg(
    proportion=1.0,
    size=(8.0, 8.0),
    step_height_range=(0.0, 0.2),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  output = cfg.function(difficulty=0.5, spec=spec, rng=rng)

  platform_geom = output.geometries[-1].geom
  assert platform_geom is not None
  platform_full_width = 2 * platform_geom.size[0]
  assert platform_full_width >= cfg.platform_width


def test_perlin_border_width_accepts_value_gte_horizontal_scale(
  rng: np.random.Generator,
):
  """Perlin terrain should accept border_width >= horizontal_scale."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")

  cfg = HfPerlinNoiseTerrainCfg(
    proportion=1.0,
    size=(10.0, 10.0),
    height_range=(0.1, 0.5),
    horizontal_scale=0.1,
    resolution=0.5,
    border_width=0.2,
  )
  output = cfg.function(difficulty=0.5, spec=spec, rng=rng)
  assert len(output.geometries) > 0
