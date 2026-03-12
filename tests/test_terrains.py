"""Tests for terrain generation."""

import mujoco
import numpy as np

from mjlab.terrains.primitive_terrains import BoxSteppingStonesTerrainCfg

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
    # Skip platform, floor, and border geoms.
    is_platform = (
      np.isclose(pos[0], center)
      and np.isclose(pos[1], center)
      and np.isclose(size[0], cfg.platform_width / 2, atol=1e-4)
    )
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
