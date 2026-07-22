"""Tests for velocity-football task registration."""

from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity_football.rl import VelocityOnPolicyRunner

_TASK_ID = "Mjlab-Velocity-Football-Flat-Unitree-G1"
_PRETRAIN_TASK_ID = "Mjlab-Velocity-Football-Pretrain-Flat-Unitree-G1"


def test_football_task_is_registered_without_replacing_native_velocity_task() -> None:
  task_ids = list_tasks()

  assert _TASK_ID in task_ids
  assert _PRETRAIN_TASK_ID in task_ids
  assert "Mjlab-Velocity-Flat-Unitree-G1" in task_ids


def test_registered_football_task_loads_training_and_play_configs() -> None:
  training_cfg = load_env_cfg(_TASK_ID)
  play_cfg = load_env_cfg(_TASK_ID, play=True)

  assert set(training_cfg.scene.entities) == {"robot", "ball"}
  assert training_cfg.scene.terrain is not None
  assert training_cfg.scene.terrain.terrain_type == "plane"
  assert training_cfg.scene.terrain.terrain_generator is None
  assert training_cfg.observations["actor"].enable_corruption

  assert set(play_cfg.scene.entities) == {"robot", "ball"}
  assert not play_cfg.observations["actor"].enable_corruption
  assert "push_robot" not in play_cfg.events
  assert play_cfg.episode_length_s >= 1e9


def test_registered_football_task_uses_dedicated_training_outputs() -> None:
  rl_cfg = load_rl_cfg(_TASK_ID)

  assert rl_cfg.experiment_name == "g1_velocity_football"
  assert load_runner_cls(_TASK_ID) is VelocityOnPolicyRunner


def test_registered_pretrain_task_has_no_football_dependencies() -> None:
  training_cfg = load_env_cfg(_PRETRAIN_TASK_ID)
  play_cfg = load_env_cfg(_PRETRAIN_TASK_ID, play=True)

  assert set(training_cfg.scene.entities) == {"robot"}
  assert "reset_base" in training_cfg.events
  assert "reset_football" not in training_cfg.events
  assert "ball_friction" not in training_cfg.events
  assert "track_ball_lin_vel_xy_exp" not in training_cfg.rewards
  assert "ball_front_control" not in training_cfg.rewards
  assert "ball_out_of_control" not in training_cfg.terminations
  assert "push_robot" not in play_cfg.events

  command = training_cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.resampling_time_range == (10.0, 12.0)
  assert command.ranges.lin_vel_x == (-0.5, 1.0)
  assert command.ranges.lin_vel_y == (-0.5, 0.5)

  rl_cfg = load_rl_cfg(_PRETRAIN_TASK_ID)
  assert rl_cfg.experiment_name == "g1_velocity_football_pretrain"
  assert load_runner_cls(_PRETRAIN_TASK_ID) is VelocityOnPolicyRunner
