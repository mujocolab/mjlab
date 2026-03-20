"""Tests for MDP termination functions."""

from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.envs.mdp.terminations import nan_detection
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.velocity.mdp.terminations import (
  out_of_terrain_bounds,
  terrain_edge_reached,
)


@pytest.fixture
def simple_model():
  xml = """
  <mujoco>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>
  </mujoco>
  """
  return mujoco.MjModel.from_xml_string(xml)


@pytest.fixture
def mock_env_with_sim(simple_model):
  env = Mock()
  env.num_envs = 4
  env.device = get_test_device()
  env.episode_length_buf = torch.zeros(4, dtype=torch.long, device=env.device)
  env.max_episode_length = 1000
  env.sim = Simulation(
    num_envs=env.num_envs, cfg=SimulationCfg(), model=simple_model, device=env.device
  )
  return env


def test_nan_detection_function(mock_env_with_sim):
  """Test that nan_detection correctly identifies NaN/Inf environments."""
  env = mock_env_with_sim

  # No NaNs initially.
  result = nan_detection(env)
  assert result.shape == (4,)
  assert not result.any()

  # Inject NaN in qpos for env 1.
  env.sim.data.qpos[1, 0] = float("nan")
  result = nan_detection(env)
  assert result[1] and not result[0] and not result[2] and not result[3]

  # Inject Inf in qacc_warmstart for env 3.
  env.sim.data.qacc_warmstart[3, 0] = float("-inf")
  result = nan_detection(env)
  assert result[1] and result[3] and not result[0] and not result[2]


def test_nan_detection_with_termination_manager(mock_env_with_sim):
  """Test that nan_detection is properly logged by termination manager."""
  env = mock_env_with_sim

  cfg = {
    "nan_term": TerminationTermCfg(func=nan_detection, params={}, time_out=False),
  }

  manager = TerminationManager(cfg, env)

  # No terminations initially.
  result = manager.compute()
  assert not result.any()
  assert not manager.terminated.any()
  assert not manager.time_outs.any()

  # Inject NaN in env 1.
  env.sim.data.qpos[1, 0] = float("nan")

  # Should detect termination in env 1.
  result = manager.compute()
  assert result[1] and not result[0] and not result[2] and not result[3]
  assert manager.terminated[1]
  assert not manager.time_outs[1]

  # Reset should log the termination.
  reset_info = manager.reset(torch.tensor([1], device=env.device))
  assert "Episode_Termination/nan_term" in reset_info
  assert reset_info["Episode_Termination/nan_term"] == 1

  # Inject Inf in multiple envs.
  env.sim.data.qvel[0, 0] = float("inf")
  env.sim.data.qacc[2, 0] = float("-inf")

  result = manager.compute()
  assert result[0] and result[2]

  # Reset should log multiple terminations.
  reset_info = manager.reset(torch.tensor([0, 2], device=env.device))
  assert reset_info["Episode_Termination/nan_term"] == 2


@pytest.fixture
def mock_terrain_env():
  device = get_test_device()
  num_envs = 4

  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.episode_length_buf = torch.full((num_envs,), 10, dtype=torch.long, device=device)

  # Terrain with 8x8m sub-terrains.
  terrain = Mock()
  terrain.cfg.terrain_type = "generator"
  terrain.cfg.terrain_generator.size = (8.0, 8.0)

  env.scene.terrain = terrain
  env.scene.env_origins = torch.zeros(num_envs, 3, device=device)

  # Robot at spawn origin.
  asset = Mock()
  asset.data.root_link_pos_w = torch.zeros(num_envs, 3, device=device)
  env.scene.__getitem__ = Mock(return_value=asset)

  return env, asset


def test_terrain_edge_reached_within_bounds(mock_terrain_env):
  env, asset = mock_terrain_env
  result = terrain_edge_reached(env, threshold_fraction=0.95)
  assert not result.any()


def test_terrain_edge_reached_at_edge(mock_terrain_env):
  env, asset = mock_terrain_env
  # Move env 1 past 95% of half-size (4.0 * 0.95 = 3.8m).
  asset.data.root_link_pos_w[1, 0] = 3.9
  result = terrain_edge_reached(env, threshold_fraction=0.95)
  assert result[1] and not result[0] and not result[2] and not result[3]


def test_terrain_edge_reached_skips_early_steps(mock_terrain_env):
  env, asset = mock_terrain_env
  asset.data.root_link_pos_w[0, 0] = 5.0
  env.episode_length_buf[:] = 1  # Too early.
  result = terrain_edge_reached(env, threshold_fraction=0.95)
  assert not result.any()


def test_terrain_edge_reached_no_terrain(mock_terrain_env):
  env, asset = mock_terrain_env
  env.scene.terrain = None
  asset.data.root_link_pos_w[0, 0] = 100.0
  result = terrain_edge_reached(env, threshold_fraction=0.95)
  assert not result.any()


@pytest.fixture
def mock_grid_terrain_env():
  device = get_test_device()
  num_envs = 4

  env = Mock()
  env.num_envs = num_envs
  env.device = device

  # 10x10 grid of 8x8m sub-terrains → half_x = 40m, limit = 39.7m.
  terrain = Mock()
  terrain.cfg.terrain_type = "generator"
  terrain.cfg.terrain_generator.size = (8.0, 8.0)
  terrain.cfg.terrain_generator.num_rows = 10
  terrain.cfg.terrain_generator.num_cols = 10

  env.scene.terrain = terrain

  asset = Mock()
  asset.data.root_link_pos_w = torch.zeros(num_envs, 3, device=device)
  env.scene.__getitem__ = Mock(return_value=asset)

  return env, asset


def test_out_of_terrain_bounds_within(mock_grid_terrain_env):
  env, asset = mock_grid_terrain_env
  result = out_of_terrain_bounds(env)
  assert not result.any()


def test_out_of_terrain_bounds_outside(mock_grid_terrain_env):
  env, asset = mock_grid_terrain_env
  # Past the grid edge (limit_x = 40 - 0.3 = 39.7m).
  asset.data.root_link_pos_w[2, 0] = 40.0
  result = out_of_terrain_bounds(env)
  assert result[2] and not result[0] and not result[1] and not result[3]
