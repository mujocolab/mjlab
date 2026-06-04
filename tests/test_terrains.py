"""Tests for terrain generation."""

import mujoco
import numpy as np
import pytest

from mjlab.terrains.config import ALL_TERRAIN_PRESETS
from mjlab.terrains.primitive_terrains import (
  _MIN_BORDER_HEIGHT,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxSteppingStonesTerrainCfg,
)

_CFG = BoxSteppingStonesTerrainCfg(
  proportion=1.0,
  size=(8.0, 8.0),
  stone_size_range=(0.2, 0.6),
  stone_distance_range=(0.05, 0.25),
  stone_height=0.2,
  stone_height_variation=0.05,
  stone_size_variation=0.05,
  displacement_range=0.1,
  floor_depth=2.0,
  platform_width=1.5,
  border_width=0.25,
)


def _generate_stones(
  cfg: BoxSteppingStonesTerrainCfg,
  difficulty: float,
  rng: np.random.Generator,
) -> list[tuple[float, float, float, float]]:
  """Generate terrain and return stone (cx, cy, half_x, half_y) tuples."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=difficulty, spec=spec, rng=rng)

  center = cfg.size[0] / 2
  stones = []
  for geom_info in output.geometries:
    geom = geom_info.geom
    if geom is None:
      continue
    pos, size = geom.pos, geom.size
    # Skip platform, floor, and border geoms. The platform is the geom centered
    # exactly at the patch center (its size is grid-snapped, not the configured
    # width, so it is identified by position alone).
    is_platform = np.isclose(pos[0], center) and np.isclose(pos[1], center)
    is_full_span = np.isclose(size[0], cfg.size[0] / 2) or np.isclose(
      size[1], cfg.size[1] / 2
    )
    if is_platform or is_full_span:
      continue
    stones.append((pos[0], pos[1], size[0], size[1]))
  return stones


def test_no_stone_centers_inside_platform():
  """No stone center should fall inside the platform."""
  center = _CFG.size[0] / 2
  p_half = _CFG.platform_width / 2
  p_min, p_max = center - p_half, center + p_half

  for difficulty in [0.0, 0.5, 1.0]:
    stones = _generate_stones(_CFG, difficulty, np.random.default_rng(42))
    for cx, cy, _, _ in stones:
      assert not (p_min <= cx <= p_max and p_min <= cy <= p_max), (
        f"Stone at ({cx:.3f}, {cy:.3f}) inside platform at difficulty={difficulty}"
      )


def test_stone_size_decreases_with_difficulty():
  """Average stone size should be smaller at higher difficulty."""
  sizes = {}
  for difficulty in [0.0, 1.0]:
    stones = _generate_stones(_CFG, difficulty, np.random.default_rng(42))
    sizes[difficulty] = np.mean([hx + hy for _, _, hx, hy in stones])

  assert sizes[0.0] > sizes[1.0]


@pytest.mark.parametrize(
  "cfg_cls", [BoxPyramidStairsTerrainCfg, BoxInvertedPyramidStairsTerrainCfg]
)
def test_pyramid_stairs_border_present_at_zero_difficulty(cfg_cls):
  """At difficulty 0 the step height collapses to 0, but the flat border frame
  must still be generated as solid, non-degenerate geometry (regression for the
  empty-boundary bug, issue #1033)."""
  cfg = cfg_cls(
    size=(8.0, 8.0),
    step_height_range=(0.0, 0.2),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=0.0, spec=spec, rng=np.random.default_rng(0))

  # The border frame sits below z=0 (top flush at ground level); inner step
  # boxes are centered at z=0. Identify the frame by its downward offset.
  border_geoms = [
    g.geom for g in output.geometries if g.geom is not None and g.geom.pos[2] < -1e-4
  ]
  assert len(border_geoms) == 4, "Expected four border frame boxes."
  for geom in border_geoms:
    # Each frame box must be solid, not a degenerate zero-height geom, and its
    # top must be flush with the ground plane at z=0.
    assert geom.size[2] >= _MIN_BORDER_HEIGHT / 2 - 1e-9
    assert np.isclose(geom.pos[2] + geom.size[2], 0.0, atol=1e-6)


@pytest.mark.parametrize("preset_name", sorted(ALL_TERRAIN_PRESETS))
@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_preset_compiles_across_difficulty(preset_name, difficulty):
  """Every terrain preset must generate compilable MuJoCo geometry across the
  full difficulty range. Difficulty 0 is exercised explicitly because curriculum
  row 0 lands there deterministically, which previously produced degenerate
  geometry (zero-height hfields, NaN colors, missing borders)."""
  cfg = ALL_TERRAIN_PRESETS[preset_name](size=(8.0, 8.0))
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  cfg.function(difficulty=difficulty, spec=spec, rng=np.random.default_rng(0))
  # Compiling validates geom/hfield sizes and rgba values (catches NaNs and
  # non-positive sizes that MuJoCo rejects).
  spec.compile()
