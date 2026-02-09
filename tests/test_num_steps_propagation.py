import tempfile
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.scripts.train import TrainConfig
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel
from mjlab.terrains import TerrainImporterCfg


@pytest.fixture
def task_id():
  return "Mjlab-Velocity-Rough-Unitree-G1"


def test_cli_argument_parsing(task_id):
  """Test that --agent.num-steps-per-env is parsed correctly by tyro."""
  # Simulate arguments
  overridden_steps = 48
  args_list = [
    f"--agent.num_steps_per_env={overridden_steps}",
    "--agent.max_iterations=1",
  ]

  # Parse
  parsed_config = tyro.cli(
    TrainConfig, args=args_list, default=TrainConfig.from_task(task_id)
  )

  assert parsed_config.agent.num_steps_per_env == overridden_steps


def test_runner_propagates_num_steps(task_id):
  """Test that MjlabOnPolicyRunner propagates num_steps_per_env to the environment config."""
  from conftest import get_test_device

  overridden_steps = 48
  device = get_test_device()

  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),
      num_envs=1,
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "dummy": ObservationTermCfg(
            func=lambda env: torch.zeros(env.num_envs, 1, device=env.device)
          )
        }
      ),
      "critic": ObservationGroupCfg(
        terms={
          "dummy": ObservationTermCfg(
            func=lambda env: torch.zeros(env.num_envs, 1, device=env.device)
          )
        }
      ),
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01)),
    decimation=1,
    episode_length_s=1.0,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapper = RslRlVecEnvWrapper(env)

  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=overridden_steps,
    max_iterations=1,
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    MjlabOnPolicyRunner(wrapper, asdict(agent_cfg), log_dir=tmpdir, device=device)
    assert env.cfg.num_steps_per_env == overridden_steps


@pytest.fixture
def mock_env_with_command():
  """Create a mock environment with command manager for curriculum tests."""
  mock_env = MagicMock(spec=ManagerBasedRlEnv)
  mock_env.command_manager = MagicMock()
  mock_env.cfg = MagicMock(spec=ManagerBasedRlEnvCfg)
  mock_env.cfg.num_steps_per_env = 48
  mock_env.common_step_counter = 0

  mock_command_term = MagicMock(spec=CommandTerm)
  mock_cfg = UniformVelocityCommandCfg(
    resampling_time_range=(1.0, 2.0),
    entity_name="dummy",
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
    ),
  )
  mock_command_term.cfg = mock_cfg
  mock_env.command_manager.get_term.return_value = mock_command_term

  return mock_env, mock_cfg


def test_curriculum_with_step_type(mock_env_with_command):
  """Test curriculum using 'step' type (used directly as threshold)."""
  mock_env, mock_cfg = mock_env_with_command

  # Stages using "step" - threshold is used directly (not scaled)
  stages = [
    {"step": 0, "lin_vel_x": (-1.0, 1.0)},
    {"step": 5000, "lin_vel_x": (-2.0, 2.0)},
  ]

  # Before threshold (4000 < 5000)
  mock_env.common_step_counter = 4000
  commands_vel(mock_env, None, "cmd", stages)
  assert mock_cfg.ranges.lin_vel_x == (-1.0, 1.0)

  # After threshold (6000 > 5000)
  mock_env.common_step_counter = 6000
  commands_vel(mock_env, None, "cmd", stages)
  assert mock_cfg.ranges.lin_vel_x == (-2.0, 2.0)

  # Verify step is independent of num_steps_per_env
  mock_cfg.ranges.lin_vel_x = (-1.0, 1.0)  # Reset
  mock_env.cfg.num_steps_per_env = 10  # Should not affect step-based stages
  mock_env.common_step_counter = 4000  # Still < 5000
  commands_vel(mock_env, None, "cmd", stages)
  assert mock_cfg.ranges.lin_vel_x == (-1.0, 1.0)  # Should NOT trigger stage 2


def test_curriculum_with_iteration_type(mock_env_with_command):
  """Test curriculum using 'iteration' type (multiplied by num_steps_per_env)."""
  mock_env, mock_cfg = mock_env_with_command
  mock_env.cfg.num_steps_per_env = 48

  # Stages using "iteration" - threshold = iteration * num_steps_per_env
  stages = [
    {"iteration": 0, "lin_vel_x": (-1.0, 1.0)},
    {"iteration": 100, "lin_vel_x": (-2.0, 2.0)},  # Threshold = 100 * 48 = 4800
  ]

  # Before threshold (4000 < 4800)
  mock_env.common_step_counter = 4000
  commands_vel(mock_env, None, "cmd", stages)
  assert mock_cfg.ranges.lin_vel_x == (-1.0, 1.0)

  # After threshold (5000 > 4800)
  mock_env.common_step_counter = 5000
  commands_vel(mock_env, None, "cmd", stages)
  assert mock_cfg.ranges.lin_vel_x == (-2.0, 2.0)

  # Verify threshold scales with num_steps_per_env
  mock_cfg.ranges.lin_vel_x = (-1.0, 1.0)  # Reset
  mock_env.cfg.num_steps_per_env = 10  # New threshold = 100 * 10 = 1000
  mock_env.common_step_counter = 1500  # > 1000
  commands_vel(mock_env, None, "cmd", stages)
  assert mock_cfg.ranges.lin_vel_x == (-2.0, 2.0)


def test_curriculum_rejects_mixed_step_and_iteration():
  """Test that curriculum raises error when both step and iteration are specified."""
  mock_env = MagicMock(spec=ManagerBasedRlEnv)
  mock_env.command_manager = MagicMock()
  mock_env.cfg = MagicMock(spec=ManagerBasedRlEnvCfg)
  mock_env.cfg.num_steps_per_env = 48
  mock_env.common_step_counter = 100

  mock_command_term = MagicMock(spec=CommandTerm)
  mock_cfg = UniformVelocityCommandCfg(
    resampling_time_range=(1.0, 2.0),
    entity_name="dummy",
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
    ),
  )
  mock_command_term.cfg = mock_cfg
  mock_env.command_manager.get_term.return_value = mock_command_term

  # Invalid: both step and iteration specified
  invalid_stages = [
    {"step": 0, "iteration": 0, "lin_vel_x": (-1.0, 1.0)},
  ]

  with pytest.raises(ValueError, match="step or iteration"):
    commands_vel(mock_env, None, "cmd", invalid_stages)


if __name__ == "__main__":
  # Allow running this file directly for debugging
  pytest.main([__file__])
