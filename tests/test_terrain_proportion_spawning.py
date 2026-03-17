"""Tests for proportion-based robot spawning on terrain columns."""

import torch

from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg


def _make_terrain_entity(num_envs: int) -> TerrainEntity:
  """Create a TerrainEntity with plane terrain for testing."""
  cfg = TerrainEntityCfg(terrain_type="plane", num_envs=num_envs, env_spacing=2.0)
  return TerrainEntity(cfg, device="cpu")


def _make_origins(num_rows: int, num_cols: int) -> torch.Tensor:
  """Create mock terrain origins tensor [num_rows, num_cols, 3]."""
  origins = torch.zeros(num_rows, num_cols, 3)
  for r in range(num_rows):
    for c in range(num_cols):
      origins[r, c, 0] = float(r)
      origins[r, c, 1] = float(c)
  return origins


def _make_generator_cfg(proportions: list[float]) -> TerrainGeneratorCfg:
  """Create a TerrainGeneratorCfg with given proportions."""
  sub_terrains = {}
  for i, p in enumerate(proportions):
    sub_terrains[f"terrain_{i}"] = BoxFlatTerrainCfg(
      proportion=p,
      size=(4.0, 4.0),
    )
  return TerrainGeneratorCfg(
    size=(4.0, 4.0),
    sub_terrains=sub_terrains,
  )


class TestEvenDistributionFallback:
  """Without terrain_generator_cfg, robots should be distributed evenly."""

  def test_even_distribution_without_generator_cfg(self) -> None:
    num_envs = 30
    num_cols = 3
    entity = _make_terrain_entity(num_envs)
    origins = _make_origins(5, num_cols)

    entity._compute_env_origins_curriculum(num_envs, origins, None)

    # Each column should get num_envs / num_cols = 10 robots.
    for col in range(num_cols):
      count = (entity.terrain_types == col).sum().item()
      assert count == 10, f"Column {col} got {count} envs, expected 10"


class TestProportionDistribution:
  """With terrain_generator_cfg, robots should be distributed by proportion."""

  def test_basic_proportions(self) -> None:
    num_envs = 100
    proportions = [0.5, 0.3, 0.2]
    gen_cfg = _make_generator_cfg(proportions)
    entity = _make_terrain_entity(num_envs)
    origins = _make_origins(5, len(proportions))

    entity._compute_env_origins_curriculum(num_envs, origins, gen_cfg)

    counts = [
      (entity.terrain_types == col).sum().item() for col in range(len(proportions))
    ]
    assert counts[0] == 50
    assert counts[1] == 30
    assert counts[2] == 20

  def test_counts_sum_to_num_envs(self) -> None:
    """Total terrain_types length must always equal num_envs."""
    for num_envs in [32, 64, 100, 512, 1024, 4096]:
      proportions = [0.6, 0.25, 0.1, 0.05]
      gen_cfg = _make_generator_cfg(proportions)
      entity = _make_terrain_entity(num_envs)
      origins = _make_origins(5, len(proportions))

      entity._compute_env_origins_curriculum(num_envs, origins, gen_cfg)

      assert len(entity.terrain_types) == num_envs, (
        f"num_envs={num_envs}: terrain_types has {len(entity.terrain_types)} elements"
      )

  def test_minimum_one_per_column(self) -> None:
    """Every column should get at least 1 robot, even with tiny proportions."""
    num_envs = 100
    proportions = [0.9, 0.05, 0.05]
    gen_cfg = _make_generator_cfg(proportions)
    entity = _make_terrain_entity(num_envs)
    origins = _make_origins(5, len(proportions))

    entity._compute_env_origins_curriculum(num_envs, origins, gen_cfg)

    for col in range(len(proportions)):
      count = (entity.terrain_types == col).sum().item()
      assert count >= 1, f"Column {col} got 0 envs despite minimum-1 guarantee"

  def test_equal_proportions_gives_equal_counts(self) -> None:
    """Equal proportions should give (approximately) equal counts."""
    num_envs = 30
    proportions = [0.25, 0.25, 0.25, 0.25]
    gen_cfg = _make_generator_cfg(proportions)
    entity = _make_terrain_entity(num_envs)
    origins = _make_origins(5, len(proportions))

    entity._compute_env_origins_curriculum(num_envs, origins, gen_cfg)

    counts = [
      (entity.terrain_types == col).sum().item() for col in range(len(proportions))
    ]
    # 30 / 4 = 7.5 → expects 8, 8, 7, 7 (or similar rounding).
    for c in counts:
      assert 7 <= c <= 8, f"Expected 7 or 8 envs per column, got {c}"

  def test_env_origins_shape(self) -> None:
    """env_origins must be [num_envs, 3]."""
    num_envs = 20
    proportions = [0.5, 0.3, 0.2]
    gen_cfg = _make_generator_cfg(proportions)
    entity = _make_terrain_entity(num_envs)
    origins = _make_origins(5, len(proportions))

    result = entity._compute_env_origins_curriculum(num_envs, origins, gen_cfg)

    assert result.shape == (num_envs, 3)

  def test_mismatched_cols_falls_back_to_even(self) -> None:
    """When sub_terrains count != num_cols, should fall back to even."""
    num_envs = 30
    num_cols = 3
    # Generator config has 4 sub-terrains but origins only have 3 columns.
    gen_cfg = _make_generator_cfg([0.4, 0.3, 0.2, 0.1])
    entity = _make_terrain_entity(num_envs)
    origins = _make_origins(5, num_cols)

    entity._compute_env_origins_curriculum(num_envs, origins, gen_cfg)

    # Should fall back to even distribution: 10 per column.
    for col in range(num_cols):
      count = (entity.terrain_types == col).sum().item()
      assert count == 10, f"Column {col} got {count} envs, expected 10 (fallback)"
