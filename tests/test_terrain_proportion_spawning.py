"""Tests for proportion-based robot spawning on terrain columns."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg


def _make_terrain_entity(num_envs: int) -> TerrainEntity:
  cfg = TerrainEntityCfg(terrain_type="plane", num_envs=num_envs, env_spacing=2.0)
  return TerrainEntity(cfg, device="cpu")


def _make_origins(num_rows: int, num_cols: int) -> torch.Tensor:
  origins = torch.zeros(num_rows, num_cols, 3)
  for r in range(num_rows):
    for c in range(num_cols):
      origins[r, c, 0] = float(r)
      origins[r, c, 1] = float(c)
  return origins


def _normalized(proportions: list[float]) -> np.ndarray:
  p = np.array(proportions)
  return p / p.sum()


def _get_counts(entity: TerrainEntity, num_cols: int) -> list[int]:
  return [int((entity.terrain_types == col).sum().item()) for col in range(num_cols)]


# Fallback (no proportions).


def test_even_distribution_without_proportions() -> None:
  num_envs = 30
  num_cols = 3
  entity = _make_terrain_entity(num_envs)
  origins = _make_origins(5, num_cols)

  entity._compute_env_origins_curriculum(num_envs, origins, None)

  assert _get_counts(entity, num_cols) == [10, 10, 10]


# Proportional distribution.


def test_basic_proportions() -> None:
  num_envs = 100
  proportions = _normalized([0.5, 0.3, 0.2])
  entity = _make_terrain_entity(num_envs)
  origins = _make_origins(5, len(proportions))

  entity._compute_env_origins_curriculum(num_envs, origins, proportions)

  assert _get_counts(entity, len(proportions)) == [50, 30, 20]


@pytest.mark.parametrize("num_envs", [3, 7, 32, 64, 100, 512, 1024, 4096])
def test_counts_sum_to_num_envs(num_envs: int) -> None:
  """Total terrain_types length must always equal num_envs."""
  proportions = _normalized([0.6, 0.25, 0.1, 0.05])
  entity = _make_terrain_entity(num_envs)
  origins = _make_origins(5, len(proportions))

  entity._compute_env_origins_curriculum(num_envs, origins, proportions)

  assert len(entity.terrain_types) == num_envs


@pytest.mark.parametrize(
  "num_envs, proportions",
  [
    (3, [0.98, 0.01, 0.01]),
    (5, [0.9, 0.05, 0.03, 0.01, 0.01]),
    (100, [0.9, 0.05, 0.05]),
    (10, [0.97, 0.01, 0.01, 0.01]),
  ],
)
def test_every_column_gets_at_least_one(
  num_envs: int, proportions: list[float]
) -> None:
  """When num_envs >= num_cols, every column gets at least 1 robot."""
  p = _normalized(proportions)
  entity = _make_terrain_entity(num_envs)
  origins = _make_origins(5, len(p))

  entity._compute_env_origins_curriculum(num_envs, origins, p)

  counts = _get_counts(entity, len(p))
  for col, count in enumerate(counts):
    assert count >= 1, f"Column {col} got 0 envs (counts={counts})"


def test_fewer_envs_than_cols() -> None:
  """When num_envs < num_cols, should still work without errors."""
  num_envs = 2
  proportions = _normalized([0.5, 0.3, 0.1, 0.1])
  entity = _make_terrain_entity(num_envs)
  origins = _make_origins(5, len(proportions))

  result = entity._compute_env_origins_curriculum(num_envs, origins, proportions)

  assert len(entity.terrain_types) == num_envs
  assert result.shape == (num_envs, 3)


def test_equal_proportions_gives_equal_counts() -> None:
  num_envs = 30
  proportions = _normalized([1.0, 1.0, 1.0, 1.0])
  entity = _make_terrain_entity(num_envs)
  origins = _make_origins(5, len(proportions))

  entity._compute_env_origins_curriculum(num_envs, origins, proportions)

  counts = _get_counts(entity, len(proportions))
  for c in counts:
    assert 7 <= c <= 8, f"Expected 7 or 8 envs per column, got {c}"


def test_env_origins_shape() -> None:
  num_envs = 20
  proportions = _normalized([0.5, 0.3, 0.2])
  entity = _make_terrain_entity(num_envs)
  origins = _make_origins(5, len(proportions))

  result = entity._compute_env_origins_curriculum(num_envs, origins, proportions)

  assert result.shape == (num_envs, 3)


def test_env_origins_match_terrain_origins() -> None:
  """Each env_origin must correspond to a valid (level, type) in origins."""
  num_envs = 50
  proportions = _normalized([0.6, 0.3, 0.1])
  entity = _make_terrain_entity(num_envs)
  origins = _make_origins(5, len(proportions))

  result = entity._compute_env_origins_curriculum(num_envs, origins, proportions)

  for i in range(num_envs):
    level = int(entity.terrain_levels[i].item())
    ttype = int(entity.terrain_types[i].item())
    expected = origins[level, ttype]
    assert torch.allclose(result[i], expected), (
      f"env {i}: origin {result[i].tolist()} != "
      f"origins[{level},{ttype}]={expected.tolist()}"
    )


def test_mismatched_cols_falls_back_to_even() -> None:
  """When proportions length != num_cols, should fall back to even."""
  num_envs = 30
  num_cols = 3
  # 4 proportions but only 3 columns in origins.
  proportions = _normalized([0.4, 0.3, 0.2, 0.1])
  entity = _make_terrain_entity(num_envs)
  origins = _make_origins(5, num_cols)

  entity._compute_env_origins_curriculum(num_envs, origins, proportions)

  assert _get_counts(entity, num_cols) == [10, 10, 10]


def test_proportions_are_respected_approximately() -> None:
  """With large num_envs, actual distribution should track proportions."""
  num_envs = 10000
  raw = [0.5, 0.3, 0.15, 0.05]
  proportions = _normalized(raw)
  entity = _make_terrain_entity(num_envs)
  origins = _make_origins(5, len(proportions))

  entity._compute_env_origins_curriculum(num_envs, origins, proportions)

  counts = _get_counts(entity, len(proportions))
  for col, (count, prop) in enumerate(zip(counts, proportions, strict=True)):
    expected = prop * num_envs
    assert abs(count - expected) < 10, (
      f"Column {col}: got {count}, expected ~{expected:.0f}"
    )


def test_generator_does_not_mutate_cfg_num_cols() -> None:
  """TerrainGenerator must not mutate the caller's cfg.num_cols."""
  cfg = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    curriculum=True,
    num_cols=1,
    sub_terrains={
      "a": BoxFlatTerrainCfg(proportion=0.5, size=(4.0, 4.0)),
      "b": BoxFlatTerrainCfg(proportion=0.3, size=(4.0, 4.0)),
      "c": BoxFlatTerrainCfg(proportion=0.2, size=(4.0, 4.0)),
    },
  )
  TerrainGenerator(cfg)
  assert cfg.num_cols == 1, f"cfg.num_cols was mutated to {cfg.num_cols}"
