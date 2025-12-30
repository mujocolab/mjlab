"""Tests for TerrainEntity."""

import pytest
import torch

from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  TerrainEntity,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)
from mjlab.utils import spec_config as spec_cfg


@pytest.fixture
def device():
  return "cpu"


def test_terrain_entity_plane_creation():
  """Test TerrainEntity can be created with plane terrain."""
  cfg = TerrainEntityCfg(
    terrain_type="plane",
    num_envs=4,
    env_spacing=2.0,
  )
  terrain = cfg.build()

  assert isinstance(terrain, TerrainEntity)
  assert terrain.spec is not None


def test_terrain_entity_generator_creation():
  """Test TerrainEntity can be created with generator terrain."""
  cfg = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      size=(4.0, 4.0),
      num_rows=2,
      num_cols=2,
      sub_terrains={
        "stairs": BoxPyramidStairsTerrainCfg(
          proportion=1.0,
          step_height_range=(0.0, 0.1),
          step_width=0.3,
        )
      },
    ),
    num_envs=4,
  )
  terrain = cfg.build()

  assert isinstance(terrain, TerrainEntity)
  assert terrain.terrain_generator is not None
  assert terrain.spec is not None


def test_terrain_entity_with_editors():
  """Test TerrainEntity supports entity editors."""
  custom_texture = spec_cfg.TextureCfg(
    name="test_texture",
    type="2d",
    builtin="gradient",
    rgb1=(0.5, 0.5, 0.5),
    rgb2=(0.2, 0.2, 0.2),
    width=128,
    height=128,
  )

  cfg = TerrainEntityCfg(
    terrain_type="plane",
    num_envs=2,
    textures=(custom_texture,),
  )
  terrain = cfg.build()

  # Verify the texture was added to the spec
  texture_names = [t.name for t in terrain.spec.textures]
  assert "test_texture" in texture_names


def test_terrain_entity_in_scene(device):
  """Test TerrainEntity works correctly in a Scene."""
  scene_cfg = SceneCfg(
    num_envs=4,
    entities={"terrain": TerrainEntityCfg(terrain_type="plane")},
  )
  scene = Scene(scene_cfg, device)

  assert scene.terrain is not None
  assert isinstance(scene.terrain, TerrainEntity)

  # Initialize the scene
  mj_model = scene.compile()
  sim = Simulation(num_envs=4, cfg=SimulationCfg(), model=mj_model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  # Check env_origins are accessible
  env_origins = scene.env_origins
  assert env_origins.shape == (4, 3)


def test_terrain_entity_env_origins(device):
  """Test terrain entity provides correct env_origins."""
  scene_cfg = SceneCfg(
    num_envs=4,
    env_spacing=3.0,
    entities={"terrain": TerrainEntityCfg(terrain_type="plane")},
  )
  scene = Scene(scene_cfg, device)
  mj_model = scene.compile()
  sim = Simulation(num_envs=4, cfg=SimulationCfg(), model=mj_model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  # env_origins should be computed based on env_spacing
  env_origins = scene.env_origins
  assert env_origins.shape == (4, 3)
  # Origins should be on a grid
  assert torch.abs(env_origins[:, 2]).max() < 1e-6  # All z values should be 0


def test_terrain_entity_curriculum(device):
  """Test terrain entity with curriculum terrain."""
  cfg = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      size=(8.0, 8.0),
      num_rows=3,
      num_cols=2,
      sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=1.0),
      },
      curriculum=True,
    ),
    num_envs=4,
    max_init_terrain_level=1,
  )

  # Build terrain entity directly without scene compilation
  # This tests curriculum setup without relying on MuJoCo compilation
  terrain = cfg.build()

  # Check curriculum properties are set up correctly from the generator
  assert terrain.terrain_generator is not None
  # Note: curriculum is a property of the config, not the generator object
  assert cfg.terrain_generator is not None
  assert cfg.terrain_generator.curriculum is True

  # Check terrain origins are computed (numpy before initialize)
  assert terrain._terrain_origins_np is not None
  assert terrain._terrain_origins_np.shape == (3, 2, 3)  # num_rows x num_cols x 3

  # Check terrain levels/types are initialized (numpy before initialize)
  assert terrain._terrain_levels_np is not None
  assert terrain._terrain_levels_np.shape == (4,)  # num_envs
  assert terrain._terrain_types_np is not None
  assert terrain._terrain_types_np.shape == (4,)  # num_envs

  # All initial levels should be <= max_init_terrain_level
  assert (terrain._terrain_levels_np <= 1).all()


def test_terrain_entity_max_init_level():
  """Test terrain entity respects max_init_terrain_level setting."""
  cfg = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      size=(8.0, 8.0),
      num_rows=4,  # 4 difficulty levels (rows)
      num_cols=2,
      sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=1.0),
      },
      curriculum=True,
    ),
    num_envs=8,
    max_init_terrain_level=2,  # Limit initial level to 0, 1, or 2
  )

  terrain = cfg.build()

  # All initial levels should be <= max_init_terrain_level
  assert terrain._terrain_levels_np is not None
  assert (terrain._terrain_levels_np <= 2).all()


def test_terrain_entity_has_curriculum_state():
  """Test terrain entity with generator has curriculum state arrays."""
  cfg = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      size=(8.0, 8.0),
      num_rows=3,
      num_cols=2,
      sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=1.0),
      },
    ),
    num_envs=4,
  )

  terrain = cfg.build()

  # Generator terrain should have curriculum state
  assert terrain._terrain_origins_np is not None
  assert terrain._terrain_origins_np.shape == (3, 2, 3)
  assert terrain._terrain_levels_np is not None
  assert terrain._terrain_levels_np.shape == (4,)
  assert terrain._terrain_types_np is not None
  assert terrain._terrain_types_np.shape == (4,)


def test_scene_terrain_getitem_access(device):
  """Test terrain can be accessed via scene['terrain']."""
  scene_cfg = SceneCfg(
    num_envs=2,
    entities={"terrain": TerrainEntityCfg(terrain_type="plane")},
  )
  scene = Scene(scene_cfg, device)

  terrain = scene["terrain"]
  assert isinstance(terrain, TerrainEntity)
  assert terrain is scene.terrain
