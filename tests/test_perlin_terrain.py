"""Tests for Perlin noise terrain generation."""

import mujoco
import numpy as np
import pytest

from mjlab.terrains.heightfield_terrains import HfPerlinNoiseTerrainCfg


@pytest.fixture
def rng() -> np.random.Generator:
  return np.random.default_rng(42)


def test_perlin_terrain_generation(rng: np.random.Generator):
  """Verify that Perlin noise terrain generates a valid TerrainOutput."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")

  cfg = HfPerlinNoiseTerrainCfg(
    proportion=1.0,
    size=(10.0, 10.0),
    height_range=(0.1, 0.5),
    octaves=4,
    persistence=0.5,
    lacunarity=2.0,
    scale=10.0,
    horizontal_scale=0.1,
    resolution=0.1,
  )

  output = cfg.function(difficulty=0.5, spec=spec, rng=rng)

  # Basic checks for TerrainOutput.
  assert output.origin.shape == (3,)
  assert len(output.geometries) == 1
  assert output.geometries[0].geom is not None
  assert output.geometries[0].hfield is not None

  # Check hfield size.
  # size = [x/2, y/2, max_height, base_thickness]
  hfield = output.geometries[0].hfield
  assert hfield.size[0] == 5.0
  assert hfield.size[1] == 5.0
  assert hfield.size[2] > 0  # max height should be positive

  # Check hfield data.
  data = np.asarray(hfield.userdata)
  assert len(data) == 100 * 100  # (10.0/0.1) * (10.0/0.1)
  assert np.all(data >= 0.0)
  assert np.all(data <= 1.0)
  assert np.max(data) > 0.0  # it should not be flat 0

  # Check if height variation is present.
  assert np.min(data) < np.max(data)


def test_perlin_terrain_resolution(rng: np.random.Generator):
  """Verify that higher resolution increases pixel count."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")

  cfg = HfPerlinNoiseTerrainCfg(
    proportion=1.0,
    size=(10.0, 10.0),
    height_range=(0.1, 0.5),
    horizontal_scale=0.1,
    resolution=0.05,  # 200 pixels
  )

  output = cfg.function(difficulty=0.5, spec=spec, rng=rng)
  hfield = output.geometries[0].hfield
  assert hfield is not None
  # (10.0 / 0.05) = 200
  assert hfield.nrow == 200
  assert hfield.ncol == 200
  assert len(hfield.userdata) == 200 * 200


def test_perlin_terrain_with_border(rng: np.random.Generator):
  """Verify Perlin noise terrain with a border."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")

  cfg = HfPerlinNoiseTerrainCfg(
    proportion=1.0,
    size=(10.0, 10.0),
    height_range=(0.1, 0.5),
    border_width=2.0,
    horizontal_scale=0.1,
    resolution=0.1,
  )

  output = cfg.function(difficulty=0.5, spec=spec, rng=rng)
  hfield = output.geometries[0].hfield
  assert hfield is not None
  data = np.asarray(hfield.userdata).reshape(100, 100)

  # Border pixels: 2.0 / 0.1 = 20
  border_pixels = 20
  top_border = data[:border_pixels, :]
  bottom_border = data[-border_pixels:, :]
  left_border = data[:, :border_pixels]
  right_border = data[:, -border_pixels:]

  assert np.all(top_border == top_border[0, 0])
  assert np.all(bottom_border == bottom_border[0, 0])
  assert np.all(left_border == left_border[0, 0])
  assert np.all(right_border == right_border[0, 0])
