"""Tests for velocity-football task registration."""

from typing import Any, cast

from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommandCfg as BaseUniformVelocityCommandCfg,
)
from mjlab.tasks.velocity_football.mdp import (
  UniformVelocityCommandCfg as FootballUniformVelocityCommandCfg,
)
from mjlab.tasks.velocity_football.rl import VelocityOnPolicyRunner

_TASK_ID = "Mjlab-Velocity-Football-Flat-Unitree-G1"
_PRETRAIN_TASK_ID = "Mjlab-Velocity-Football-Pretrain-Flat-Unitree-G1"
_CURRENT_PRETRAIN_TASK_ID = "Mjlab-Velocity-Football-Current-Pretrain-Flat-Unitree-G1"
_TEMPORAL_PRETRAIN_TASK_ID = "Mjlab-Velocity-Football-Temporal-Pretrain-Flat-Unitree-G1"
_TEMPORAL_TASK_ID = "Mjlab-Velocity-Football-Temporal-Flat-Unitree-G1"
_TEMPORAL_STOP_REWARD_TASK_ID = (
  "Mjlab-Velocity-Football-Temporal-StopReward-Flat-Unitree-G1"
)
_VISUAL_MASK_TASK_ID = "Mjlab-Velocity-Football-VisualMask-Flat-Unitree-G1"
_ABLATION_TASK_IDS = {
  "Mjlab-Velocity-Football-R0-IsaacLab-Robust-Flat-Unitree-G1",
  "Mjlab-Velocity-Football-R1-E1-Robust-Flat-Unitree-G1",
  "Mjlab-Velocity-Football-R2-No-Relative-Velocity-Flat-Unitree-G1",
  "Mjlab-Velocity-Football-R3-No-Relative-Position-Flat-Unitree-G1",
}
_FACTORIAL_TASK_IDS = {
  f"Mjlab-Velocity-Football-A{actor}R{reward}-Flat-Unitree-G1"
  for actor in (0, 1)
  for reward in (0, 1)
}


def test_football_task_is_registered_without_replacing_native_velocity_task() -> None:
  task_ids = list_tasks()

  assert _TASK_ID in task_ids
  assert _PRETRAIN_TASK_ID in task_ids
  assert _CURRENT_PRETRAIN_TASK_ID in task_ids
  assert _TEMPORAL_PRETRAIN_TASK_ID in task_ids
  assert _TEMPORAL_TASK_ID in task_ids
  assert _TEMPORAL_STOP_REWARD_TASK_ID in task_ids
  assert _VISUAL_MASK_TASK_ID in task_ids
  assert _ABLATION_TASK_IDS <= set(task_ids)
  assert _FACTORIAL_TASK_IDS <= set(task_ids)
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


def test_temporal_task_has_history_mask_and_no_command_generator() -> None:
  training_cfg = load_env_cfg(_TEMPORAL_TASK_ID)
  play_cfg = load_env_cfg(_TEMPORAL_TASK_ID, play=True)
  rl_cfg = cast(Any, load_rl_cfg(_TEMPORAL_TASK_ID))

  assert training_cfg.observations["actor"].history_length is None
  assert training_cfg.observations["actor_history"].history_length == 10
  assert not training_cfg.observations["actor_history"].flatten_history_dim
  assert training_cfg.observations["critic_history"].history_length == 10
  assert "ball_visible_mask" in training_cfg.observations["actor"].terms
  assert (
    training_cfg.observations["actor"]
    .terms["ball_visible_mask"]
    .params["dropout_probability"]
    == 0.0
  )
  assert (
    play_cfg.observations["actor"]
    .terms["ball_visible_mask"]
    .params["dropout_probability"]
    == 0.0
  )
  command = training_cfg.commands["twist"]
  assert isinstance(command, FootballUniformVelocityCommandCfg)
  assert command.zero_command_ramp_time_range is None
  assert command.ball_relative_velocity_reference is None
  assert command.stop_skill_velocity_reference is None
  assert rl_cfg.obs_groups["actor"] == ("actor", "actor_history")
  assert rl_cfg.actor.class_name.endswith(":TemporalCNNModel")


def test_visual_mask_ablation_uses_current_frame_only() -> None:
  env_cfg = load_env_cfg(_VISUAL_MASK_TASK_ID)
  rl_cfg = cast(Any, load_rl_cfg(_VISUAL_MASK_TASK_ID))

  assert set(env_cfg.observations) == {"actor", "critic", "critic_history"}
  assert "ball_visible_mask" in env_cfg.observations["actor"].terms
  assert rl_cfg.obs_groups == {
    "actor": ("actor",),
    "critic": ("critic", "critic_history"),
  }
  assert rl_cfg.actor.class_name == "MLPModel"
  assert rl_cfg.critic.class_name.endswith(":TemporalCNNModel")


def test_temporal_stop_reward_task_only_adds_low_speed_ball_reward() -> None:
  cfg = load_env_cfg(_TEMPORAL_STOP_REWARD_TASK_ID)
  assert "stop_ball_lin_vel_xy_exp" in cfg.rewards
  term = cfg.rewards["stop_ball_lin_vel_xy_exp"]
  assert term.weight == 0.5
  assert term.params["command_threshold"] == 0.10


def test_temporal_history_length_ablation_tasks() -> None:
  for history_length in (5, 10, 20):
    task_id = (
      "Mjlab-Velocity-Football-Temporal-Flat-Unitree-G1"
      if history_length == 10
      else f"Mjlab-Velocity-Football-Temporal-History{history_length}-Flat-Unitree-G1"
    )
    cfg = load_env_cfg(task_id)
    assert cfg.observations["actor_history"].history_length == history_length
    assert cfg.observations["critic_history"].history_length == history_length


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
  assert isinstance(command, BaseUniformVelocityCommandCfg)
  assert command.resampling_time_range == (10.0, 12.0)
  assert command.ranges.lin_vel_x == (-0.5, 1.0)
  assert command.ranges.lin_vel_y == (-0.5, 0.5)

  rl_cfg = load_rl_cfg(_PRETRAIN_TASK_ID)
  assert rl_cfg.experiment_name == "g1_velocity_football_pretrain"
  assert load_runner_cls(_PRETRAIN_TASK_ID) is VelocityOnPolicyRunner


def test_temporal_pretrain_task_matches_temporal_football_history() -> None:
  env_cfg = load_env_cfg(_TEMPORAL_PRETRAIN_TASK_ID)
  rl_cfg = cast(Any, load_rl_cfg(_TEMPORAL_PRETRAIN_TASK_ID))

  assert set(env_cfg.scene.entities) == {"robot"}
  assert env_cfg.observations["actor"].history_length is None
  assert env_cfg.observations["actor_history"].history_length == 10
  assert not env_cfg.observations["actor_history"].flatten_history_dim
  assert rl_cfg.obs_groups["actor"] == ("actor", "actor_history")
  assert rl_cfg.actor.class_name.endswith(":TemporalCNNModel")
  assert load_runner_cls(_TEMPORAL_PRETRAIN_TASK_ID) is VelocityOnPolicyRunner


def test_factorial_tasks_freeze_b1_and_reward_contracts() -> None:
  for actor_variant in (0, 1):
    for reward_variant in (0, 1):
      task_id = (
        f"Mjlab-Velocity-Football-A{actor_variant}R{reward_variant}-Flat-Unitree-G1"
      )
      env_cfg = load_env_cfg(task_id)
      rl_cfg = cast(Any, load_rl_cfg(task_id))
      actor = env_cfg.observations["actor"]
      assert actor.history_length == 5
      assert env_cfg.observations["critic_history"].history_length == 10

      if actor_variant == 0:
        assert tuple(actor.terms)[-2:] == (
          "ball_pos_b",
          "ball_to_feet_vectors_b",
        )
        assert actor.terms["ball_pos_b"].params["dropout_probability"] == 0.0
        assert actor.terms["ball_pos_b"].params["x_range"] == (0.05, 1.00)
        assert actor.terms["ball_pos_b"].params["y_range"] == (-0.70, 0.70)
        assert "ball_visible_mask" not in actor.terms
        assert "actor_history" not in env_cfg.observations
        assert rl_cfg.actor.class_name == "MLPModel"
        assert rl_cfg.obs_groups["actor"] == ("actor",)
      else:
        assert "ball_pos_b" not in actor.terms
        assert "ball_to_feet_vectors_b" not in actor.terms
        assert "ball_visible_mask" not in actor.terms
        actor_history = env_cfg.observations["actor_history"]
        assert tuple(actor_history.terms) == (
          "ball_pos_b",
          "ball_to_feet_vectors_b",
          "ball_visible_mask",
        )
        assert actor_history.history_length == 10
        assert not actor_history.flatten_history_dim
        assert actor_history.terms["ball_visible_mask"].params[
          "dropout_probability"
        ] == 0.0
        assert actor_history.terms["ball_visible_mask"].params["x_range"] == (
          0.05,
          1.00,
        )
        assert actor_history.terms["ball_visible_mask"].params["y_range"] == (
          -0.70,
          0.70,
        )
        assert rl_cfg.obs_groups["actor"] == ("actor", "actor_history")
        assert rl_cfg.actor.cnn_cfg == {
          "output_channels": (64, 64, 64),
          "kernel_size": 3,
          "activation": "elu",
          "dilations": (1, 2, 4),
          "causal": True,
          "output_mode": "last",
        }

      ball_reward = env_cfg.rewards["track_ball_lin_vel_xy_exp"]
      if reward_variant == 0:
        assert ball_reward.weight == 1.0
        assert ball_reward.params["std"] == 0.5
        assert not ball_reward.params["gate_by_position"]
        assert env_cfg.rewards["track_linear_velocity"].weight == 1.0
        assert env_cfg.rewards["track_linear_velocity"].params["std"] == 0.5
        assert env_cfg.rewards["track_angular_velocity"].weight == 2.0
        assert env_cfg.rewards["track_angular_velocity"].params["std"] == 0.5
        assert env_cfg.rewards["track_ball_relative_vel_xy_exp"].weight == 0.0
        assert env_cfg.rewards["track_ball_relative_pos_xy_exp"].weight == 0.0
        assert env_cfg.rewards["ball_outside_control_zone"].weight == 0.0
        assert env_cfg.rewards["ball_front_control"].weight == 0.5
      else:
        assert ball_reward.weight == 2.0
        assert ball_reward.params["std"] == 0.8
        assert "ball_front_control" not in env_cfg.rewards
