"""Tests for terrain configuration presets."""

import mujoco
import numpy as np
import pytest

import mjlab.terrains as terrain_gen
from mjlab.terrains.config import (
  ALL_TERRAIN_PRESETS,
  ALL_TERRAINS_CFG,
  ROUGH_TERRAINS_CFG,
  STAIRS_TERRAINS_CFG,
  pyramid_stairs,
  terrain_preset,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg


def test_all_presets_return_sub_terrain_cfg():
  for name, fn in ALL_TERRAIN_PRESETS.items():
    cfg = fn(proportion=1.0)
    assert isinstance(cfg, SubTerrainCfg), (
      f"Preset {name!r} returned {type(cfg)}, expected SubTerrainCfg"
    )


def test_preset_overrides():
  cfg = pyramid_stairs(proportion=0.5, step_width=0.5)
  assert cfg.proportion == 0.5
  assert cfg.step_width == 0.5
  # Default should still apply for unoverridden fields.
  assert cfg.platform_width == 3.0


def test_rough_terrains_cfg_structure():
  assert ROUGH_TERRAINS_CFG.size == (8.0, 8.0)
  assert ROUGH_TERRAINS_CFG.num_rows == 10
  assert ROUGH_TERRAINS_CFG.num_cols == 20
  assert len(ROUGH_TERRAINS_CFG.sub_terrains) == 7
  total = sum(c.proportion for c in ROUGH_TERRAINS_CFG.sub_terrains.values())
  assert abs(total - 1.0) < 1e-6


def test_stairs_terrains_cfg_structure():
  assert STAIRS_TERRAINS_CFG.curriculum is True
  assert len(STAIRS_TERRAINS_CFG.sub_terrains) == 4
  assert "flat" in STAIRS_TERRAINS_CFG.sub_terrains
  assert "easy_stairs" in STAIRS_TERRAINS_CFG.sub_terrains


def test_all_terrains_cfg_matches_presets():
  assert set(ALL_TERRAINS_CFG.sub_terrains.keys()) == set(ALL_TERRAIN_PRESETS.keys())
  assert ALL_TERRAINS_CFG.num_cols == len(ALL_TERRAIN_PRESETS)


def test_terrain_preset_decorator():
  """Custom preset is registered in ALL_TERRAIN_PRESETS."""

  @terrain_preset
  def _test_custom(**overrides):
    return terrain_gen.BoxFlatTerrainCfg(**overrides)

  assert "_test_custom" in ALL_TERRAIN_PRESETS
  cfg = ALL_TERRAIN_PRESETS["_test_custom"](proportion=0.5)
  assert isinstance(cfg, SubTerrainCfg)
  assert cfg.proportion == 0.5

  # Clean up.
  del ALL_TERRAIN_PRESETS["_test_custom"]


@pytest.mark.slow
def test_all_presets_generate_terrain():
  """Each preset can generate terrain without error."""
  for _name, fn in ALL_TERRAIN_PRESETS.items():
    cfg = fn(proportion=1.0, size=(4.0, 4.0))
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    rng = np.random.default_rng(42)
    cfg.function(difficulty=0.5, spec=spec, rng=rng)
