"""Tests for velocity-football task registration."""

from typing import Any, cast

import pytest

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
from mjlab.utils.noise import UniformNoiseCfg

_TASK_ID = "Mjlab-Velocity-Football-Flat-Unitree-G1"
_PRETRAIN_TASK_ID = "Mjlab-Velocity-Football-Pretrain-Flat-Unitree-G1"
_CURRENT_PRETRAIN_TASK_ID = "Mjlab-Velocity-Football-Current-Pretrain-Flat-Unitree-G1"
_TEMPORAL_PRETRAIN_TASK_ID = "Mjlab-Velocity-Football-Temporal-Pretrain-Flat-Unitree-G1"
_TEMPORAL_TASK_ID = "Mjlab-Velocity-Football-Temporal-Flat-Unitree-G1"
_TEMPORAL_STOP_REWARD_TASK_ID = (
  "Mjlab-Velocity-Football-Temporal-StopReward-Flat-Unitree-G1"
)
_VISUAL_MASK_TASK_ID = "Mjlab-Velocity-Football-VisualMask-Flat-Unitree-G1"
_ISAACLAB_ALIGNED_TASK_ID = "Mjlab-Velocity-Football-IsaacLabAligned-Flat-Unitree-G1"
_ISAACLAB_HISTORY5_LONG_DROPOUT10_TASK_ID = (
  "Mjlab-Velocity-Football-IsaacLabAligned-History5-LongDropout10-Flat-Unitree-G1"
)
_EPISODE_DROPOUT_TASK_ID = "Mjlab-Velocity-Football-A1R0-Dropout5-Flat-Unitree-G1"
_DROPOUT5_ENVELOPE30_TASK_ID = (
  "Mjlab-Velocity-Football-A1R0-Dropout5-Envelope30-Flat-Unitree-G1"
)
_TRANSITION_DROPOUT25_TASK_ID = (
  "Mjlab-Velocity-Football-A1R0-TransitionDropout25-Envelope30-Flat-Unitree-G1"
)
_TRANSITION_DROPOUT25_LEGACY_CURRICULUM_TASK_ID = (
  "Mjlab-Velocity-Football-A1R0-TransitionDropout25-Envelope30-"
  "LegacyCurriculum-Flat-Unitree-G1"
)
_LONG_DROPOUT10_LEGACY_CURRICULUM_TASK_ID = (
  "Mjlab-Velocity-Football-A1R0-LongDropout10-Envelope30-"
  "LegacyCurriculum-Flat-Unitree-G1"
)
_VISIBLE_ONLY_LEGACY_CURRICULUM_TASK_ID = (
  "Mjlab-Velocity-Football-A1R0-VisibleOnly-Envelope30-LegacyCurriculum-Flat-Unitree-G1"
)
_LONG_DROPOUT10_ROUGH10MM_TASK_ID = (
  "Mjlab-Velocity-Football-A1R0-LongDropout10-Envelope30-"
  "LegacyCurriculum-Rough10mm-Unitree-G1"
)
_VISIBILITY_BLEND_TASK_ID = (
  "Mjlab-Velocity-Football-A1R0-VisibilityBlend-Flat-Unitree-G1"
)
_VISIBILITY_BLEND_V2_TASK_ID = (
  "Mjlab-Velocity-Football-A1R0-VisibilityBlend-V2-Flat-Unitree-G1"
)
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
  assert _ISAACLAB_ALIGNED_TASK_ID in task_ids
  assert _ISAACLAB_HISTORY5_LONG_DROPOUT10_TASK_ID in task_ids
  assert _VISIBILITY_BLEND_V2_TASK_ID in task_ids
  assert _DROPOUT5_ENVELOPE30_TASK_ID in task_ids
  assert _TRANSITION_DROPOUT25_TASK_ID in task_ids
  assert _TRANSITION_DROPOUT25_LEGACY_CURRICULUM_TASK_ID in task_ids
  assert _LONG_DROPOUT10_ROUGH10MM_TASK_ID in task_ids
  assert _VISIBLE_ONLY_LEGACY_CURRICULUM_TASK_ID in task_ids
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


def test_dropout5_task_disables_ball_observation_for_whole_episodes() -> None:
  training_cfg = load_env_cfg(_EPISODE_DROPOUT_TASK_ID)
  play_cfg = load_env_cfg(_EPISODE_DROPOUT_TASK_ID, play=True)

  for term in training_cfg.observations["actor_history"].terms.values():
    assert term.params["dropout_probability"] == 0.0
    assert term.params["episode_dropout_probability"] == 0.05
  assert training_cfg.terminations["ball_out_of_control"].params[
    "ignore_episode_hidden"
  ]
  assert (
    "ignore_when_ball_unseen"
    not in training_cfg.terminations["ball_out_of_control"].params
  )

  for term in play_cfg.observations["actor_history"].terms.values():
    assert term.params["dropout_probability"] == 0.0
    assert term.params["episode_dropout_probability"] == 0.0
  assert not play_cfg.terminations["ball_out_of_control"].params[
    "ignore_episode_hidden"
  ]


def test_dropout5_envelope30_only_adds_bounded_velocity_penalty() -> None:
  baseline = load_env_cfg(_EPISODE_DROPOUT_TASK_ID)
  cfg = load_env_cfg(_DROPOUT5_ENVELOPE30_TASK_ID)

  assert set(cfg.rewards) == set(baseline.rewards) | {"command_velocity_envelope"}
  envelope = cfg.rewards["command_velocity_envelope"]
  assert envelope.func.__name__ == "command_velocity_envelope_l2"
  assert envelope.weight == pytest.approx(-1.0)
  assert envelope.params == {
    "command_name": "twist",
    "min_tolerance_x": 0.10,
    "min_tolerance_y": 0.10,
    "min_tolerance_yaw": 0.15,
    "relative_tolerance": 0.30,
  }
  assert cfg.observations == baseline.observations
  assert cfg.terminations == baseline.terminations


def test_transition_dropout25_replaces_whole_episode_blindness() -> None:
  cfg = load_env_cfg(_TRANSITION_DROPOUT25_TASK_ID)
  play_cfg = load_env_cfg(_TRANSITION_DROPOUT25_TASK_ID, play=True)
  baseline_cfg = load_env_cfg(_DROPOUT5_ENVELOPE30_TASK_ID)

  for term in cfg.observations["actor_history"].terms.values():
    assert term.params["episode_dropout_probability"] == 0.0
    assert term.params["transition_dropout_probability"] == pytest.approx(0.25)
    assert term.params["transition_dropout_start_range_s"] == (2.0, 6.0)
    assert term.params["transition_dropout_duration_range_s"] == (0.2, 0.8)
    assert term.params["transition_dropout_until_end_probability"] == pytest.approx(
      0.40
    )
    assert term.params["sensor_reward_fade_out_s"] == pytest.approx(0.5)
    assert term.params["sensor_reward_fade_in_s"] == pytest.approx(0.5)
  assert cfg.commands == baseline_cfg.commands
  termination = cfg.terminations["ball_out_of_control"].params
  assert not termination["ignore_episode_hidden"]
  assert termination["ignore_when_sensor_hidden"]
  assert cfg.rewards["track_ball_lin_vel_xy_exp"].params["gate_by_sensor_health"]
  assert cfg.rewards["ball_front_control"].params["gate_by_sensor_health"]
  assert cfg.rewards["action_acc_l2"].weight == pytest.approx(-0.1)
  assert "ball_control_success" in cfg.metrics
  curriculum = cfg.curriculum["lin_vel_cmd_levels"]
  assert curriculum.func.__name__ == "normal_control_lin_vel_cmd_levels"
  assert curriculum.params["min_normal_episodes"] == 256
  assert curriculum.params["validation_interval_steps"] == 12_000
  assert curriculum.params["consecutive_successes"] == 3

  for term in play_cfg.observations["actor_history"].terms.values():
    assert term.params["transition_dropout_probability"] == 0.0
  assert not play_cfg.terminations["ball_out_of_control"].params[
    "ignore_when_sensor_hidden"
  ]


def test_transition_dropout25_legacy_task_only_restores_old_curriculum() -> None:
  new_cfg = load_env_cfg(_TRANSITION_DROPOUT25_TASK_ID)
  legacy_cfg = load_env_cfg(_TRANSITION_DROPOUT25_LEGACY_CURRICULUM_TASK_ID)

  curriculum = legacy_cfg.curriculum["lin_vel_cmd_levels"]
  assert curriculum.func.__name__ == "lin_vel_cmd_levels"
  assert curriculum.params == {
    "command_name": "twist",
    "reward_term_name": "track_linear_velocity",
    "max_lin_vel_x": (-0.5, 2.0),
    "max_lin_vel_y": (-0.5, 0.5),
    "success_threshold": 0.7,
    "range_step": 0.1,
  }
  assert "ball_control_success" not in legacy_cfg.metrics
  assert legacy_cfg.observations == new_cfg.observations
  assert legacy_cfg.rewards == new_cfg.rewards
  assert legacy_cfg.terminations == new_cfg.terminations


def test_visible_only_teacher_removes_only_sensor_loss_behavior() -> None:
  visible_cfg = load_env_cfg(_VISIBLE_ONLY_LEGACY_CURRICULUM_TASK_ID)
  dropout_cfg = load_env_cfg(_LONG_DROPOUT10_LEGACY_CURRICULUM_TASK_ID)

  assert visible_cfg.commands == dropout_cfg.commands
  assert visible_cfg.actions == dropout_cfg.actions
  assert visible_cfg.events == dropout_cfg.events
  assert visible_cfg.curriculum == dropout_cfg.curriculum
  assert (
    visible_cfg.observations["actor"].terms == dropout_cfg.observations["actor"].terms
  )

  for term in visible_cfg.observations["actor_history"].terms.values():
    assert term.params["dropout_probability"] == 0.0
    assert term.params["episode_dropout_probability"] == 0.0
    assert term.params["transition_dropout_probability"] == 0.0
    assert term.params["transition_dropout_until_end_probability"] == 0.0

  termination = visible_cfg.terminations["ball_out_of_control"].params
  assert not termination["ignore_episode_hidden"]
  assert not termination["ignore_when_sensor_hidden"]
  assert not visible_cfg.rewards["track_ball_lin_vel_xy_exp"].params[
    "gate_by_sensor_health"
  ]
  assert not visible_cfg.rewards["ball_front_control"].params["gate_by_sensor_health"]
  assert visible_cfg.rewards["command_velocity_envelope"].weight == pytest.approx(-1.0)
  assert visible_cfg.rewards["action_acc_l2"].weight == pytest.approx(-0.1)


def test_long_dropout10_removes_short_visual_loss() -> None:
  cfg = load_env_cfg(_LONG_DROPOUT10_LEGACY_CURRICULUM_TASK_ID)
  play_cfg = load_env_cfg(_LONG_DROPOUT10_LEGACY_CURRICULUM_TASK_ID, play=True)

  for term in cfg.observations["actor_history"].terms.values():
    assert term.params["episode_dropout_probability"] == 0.0
    assert term.params["transition_dropout_probability"] == pytest.approx(0.10 / 0.95)
    assert term.params["transition_dropout_until_end_probability"] == pytest.approx(1.0)
    assert term.params["transition_excluded_standing_command_name"] == "twist"
    assert term.params["sensor_reward_fade_out_s"] == pytest.approx(0.5)
    assert term.params["sensor_reward_fade_in_s"] == pytest.approx(0.5)
  for term in play_cfg.observations["actor_history"].terms.values():
    assert term.params["transition_dropout_probability"] == 0.0
    assert term.params["transition_dropout_until_end_probability"] == 0.0
  assert cfg.curriculum["lin_vel_cmd_levels"].func.__name__ == "lin_vel_cmd_levels"
  assert cfg.commands["twist"].rel_standing_envs == pytest.approx(0.05)
  assert cfg.commands["twist"].standing_mode_per_episode


def test_long_dropout10_matches_isaaclab_actor_observation_randomization() -> None:
  cfg = load_env_cfg(_LONG_DROPOUT10_LEGACY_CURRICULUM_TASK_ID)

  assert "encoder_bias" not in cfg.events
  assert not cfg.observations["actor"].terms["joint_pos"].params["biased"]
  history = cfg.observations["actor_history"]
  for term in history.terms.values():
    assert term.params["bias_range"] == 0.0
    assert term.params["frame_noise_range"] == 0.0

  ball_pos = history.terms["ball_pos_b"]
  assert isinstance(ball_pos.noise, UniformNoiseCfg)
  assert (ball_pos.noise.n_min, ball_pos.noise.n_max) == (-0.05, 0.05)
  assert (ball_pos.delay_min_lag, ball_pos.delay_max_lag) == (0, 2)

  ball_to_feet = history.terms["ball_to_feet_vectors_b"]
  assert isinstance(ball_to_feet.noise, UniformNoiseCfg)
  assert (ball_to_feet.noise.n_min, ball_to_feet.noise.n_max) == (-0.10, 0.10)
  assert (ball_to_feet.delay_min_lag, ball_to_feet.delay_max_lag) == (0, 2)


def test_isaaclab_aligned_task_uses_five_masked_full_frames_and_mlp_actor() -> None:
  cfg = load_env_cfg(_ISAACLAB_ALIGNED_TASK_ID)
  runner_cfg = load_rl_cfg(_ISAACLAB_ALIGNED_TASK_ID)

  actor = cfg.observations["actor"]
  assert actor.history_length == 5
  assert actor.flatten_history_dim
  assert tuple(actor.terms) == (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "phase",
    "joint_pos",
    "joint_vel",
    "actions",
    "ball_pos_b",
    "ball_to_feet_vectors_b",
    "ball_visible_mask",
  )
  assert "actor_history" not in cfg.observations
  assert "encoder_bias" not in cfg.events
  assert runner_cfg.obs_groups["actor"] == ("actor",)
  assert runner_cfg.actor.class_name == "MLPModel"

  ball_pos = actor.terms["ball_pos_b"]
  assert isinstance(ball_pos.noise, UniformNoiseCfg)
  assert (ball_pos.noise.n_min, ball_pos.noise.n_max) == (-0.05, 0.05)
  assert (ball_pos.delay_min_lag, ball_pos.delay_max_lag) == (0, 2)
  ball_to_feet = actor.terms["ball_to_feet_vectors_b"]
  assert isinstance(ball_to_feet.noise, UniformNoiseCfg)
  assert (ball_to_feet.noise.n_min, ball_to_feet.noise.n_max) == (-0.10, 0.10)
  assert (ball_to_feet.delay_min_lag, ball_to_feet.delay_max_lag) == (0, 2)
  mask = actor.terms["ball_visible_mask"]
  assert mask.params["bias_range"] == 0.0
  assert mask.params["frame_noise_range"] == 0.0


def test_isaaclab_history5_long_dropout10_only_adds_sensor_loss_package() -> None:
  cfg = load_env_cfg(_ISAACLAB_HISTORY5_LONG_DROPOUT10_TASK_ID)
  play_cfg = load_env_cfg(_ISAACLAB_HISTORY5_LONG_DROPOUT10_TASK_ID, play=True)
  runner_cfg = cast(Any, load_rl_cfg(_ISAACLAB_HISTORY5_LONG_DROPOUT10_TASK_ID))
  command_cfg = cast(Any, cfg.commands["twist"])

  actor = cfg.observations["actor"]
  assert actor.history_length == 5
  assert actor.flatten_history_dim
  assert "actor_history" not in cfg.observations
  assert runner_cfg.obs_groups["actor"] == ("actor",)
  assert runner_cfg.actor.class_name == "MLPModel"
  assert command_cfg.standing_mode_per_episode

  for term_name in ("ball_pos_b", "ball_to_feet_vectors_b", "ball_visible_mask"):
    term = actor.terms[term_name]
    assert term.params["episode_dropout_probability"] == 0.0
    assert term.params["transition_dropout_probability"] == pytest.approx(0.10 / 0.95)
    assert term.params["transition_dropout_start_range_s"] == (2.0, 6.0)
    assert term.params["transition_dropout_duration_range_s"] == (0.2, 0.8)
    assert term.params["transition_dropout_until_end_probability"] == 1.0
    assert term.params["transition_excluded_standing_command_name"] == "twist"
    assert term.params["sensor_reward_fade_out_s"] == 0.5
    assert term.params["sensor_reward_fade_in_s"] == 0.5

    play_term = play_cfg.observations["actor"].terms[term_name]
    assert play_term.params["transition_dropout_probability"] == 0.0
    assert play_term.params["transition_dropout_until_end_probability"] == 0.0

  assert cfg.rewards["track_ball_lin_vel_xy_exp"].params["gate_by_sensor_health"]
  assert cfg.rewards["ball_front_control"].params["gate_by_sensor_health"]
  assert cfg.terminations["ball_out_of_control"].params["ignore_when_sensor_hidden"]
  assert not play_cfg.terminations["ball_out_of_control"].params[
    "ignore_when_sensor_hidden"
  ]
  assert "command_velocity_envelope" not in cfg.rewards
  assert "action_acc_l2" not in cfg.rewards


def test_long_dropout10_rough10mm_uses_fixed_five_stage_terrain_schedule() -> None:
  cfg = load_env_cfg(_LONG_DROPOUT10_ROUGH10MM_TASK_ID)

  terrain = cfg.scene.terrain
  assert terrain is not None
  assert terrain.terrain_type == "generator"
  assert terrain.max_init_terrain_level == 0
  assert cfg.sim.njmax == 512
  assert cfg.sim.nconmax == 128
  assert cfg.sim.contact_sensor_maxmatch == 128
  assert cfg.sim.mujoco.ccd_iterations == 50
  generator = terrain.terrain_generator
  assert generator is not None
  assert generator.curriculum
  assert generator.num_rows == 6
  assert tuple(generator.sub_terrains) == ("random_rough",)
  rough = generator.sub_terrains["random_rough"]
  assert rough.noise_range == pytest.approx((0.0, 0.02))
  assert rough.noise_step == pytest.approx(0.002)
  assert rough.vertical_scale == pytest.approx(0.001)
  assert rough.platform_width == pytest.approx(1.5)
  assert rough.scale_with_difficulty
  curriculum = cfg.curriculum["terrain_levels"]
  assert curriculum.func.__name__ == "scheduled_rough_terrain_levels"
  assert curriculum.params == {
    "steps_per_level": 24_000,
    "max_level": 5,
    "start_step": 69_998 * 24,
  }


def test_long_dropout10_rough10mm_play_uses_maximum_roughness() -> None:
  cfg = load_env_cfg(_LONG_DROPOUT10_ROUGH10MM_TASK_ID, play=True)

  terrain = cfg.scene.terrain
  assert terrain is not None
  generator = terrain.terrain_generator
  assert generator is not None
  assert generator.difficulty_range == pytest.approx((1.0, 1.0))
  assert generator.num_rows == 1
  assert terrain.max_init_terrain_level == 0
  assert "terrain_levels" not in cfg.curriculum


def test_visibility_blend_task_is_independent_and_uses_bounded_recovery() -> None:
  training_cfg = load_env_cfg(_VISIBILITY_BLEND_TASK_ID)
  play_cfg = load_env_cfg(_VISIBILITY_BLEND_TASK_ID, play=True)

  actor_history = training_cfg.observations["actor_history"]
  assert actor_history.history_length == 10
  assert actor_history.terms["ball_visible_mask"].params[
    "episode_dropout_probability"
  ] == pytest.approx(0.05)
  assert training_cfg.terminations["ball_out_of_control"].params[
    "ignore_episode_hidden"
  ]
  assert training_cfg.terminations["ball_out_of_control"].params[
    "ignore_when_ball_unseen"
  ]

  linear = training_cfg.rewards["track_linear_velocity"]
  angular = training_cfg.rewards["track_angular_velocity"]
  envelope = training_cfg.rewards["command_velocity_envelope"]
  assert linear.func.__name__ == "track_visibility_blended_linear_velocity"
  assert linear.weight == 2.0
  assert linear.params["relative_tolerance"] == pytest.approx(0.20)
  assert angular.func.__name__ == "track_visibility_blended_angular_velocity"
  assert envelope.weight == -4.0
  assert envelope.params["min_tolerance_x"] == pytest.approx(0.10)
  assert training_cfg.rewards["track_ball_lin_vel_xy_exp"].params["gate_by_visibility"]
  assert training_cfg.rewards["ball_front_control"].params["gate_by_visibility"]

  reset = training_cfg.events["reset_football"].params
  assert reset["ball_velocity_range"] == (-0.4, 0.4)
  assert reset["stationary_ball_probability"] == pytest.approx(0.80)
  assert training_cfg.events["kick_football"].interval_range_s == (5.0, 8.0)
  assert training_cfg.events["kick_football"].params["probability"] == pytest.approx(
    0.10
  )

  assert "kick_football" not in play_cfg.events
  assert not play_cfg.terminations["ball_out_of_control"].params[
    "ignore_when_ball_unseen"
  ]
  assert play_cfg.events["reset_football"].params["ball_velocity_range"] == (
    0.0,
    0.0,
  )


def test_visibility_blend_v2_has_explicit_modes_and_dual_metric_curriculum() -> None:
  cfg = load_env_cfg(_VISIBILITY_BLEND_V2_TASK_ID)
  play_cfg = load_env_cfg(_VISIBILITY_BLEND_V2_TASK_ID, play=True)

  assert "episode_ball_observation_hidden" in cfg.observations["critic"].terms
  assert "episode_ball_observation_hidden" not in cfg.observations["actor"].terms
  assert cfg.observations["critic"].terms["episode_ball_observation_hidden"].params[
    "episode_dropout_probability"
  ] == pytest.approx(0.20)

  assert cfg.rewards["track_linear_velocity"].func.__name__ == ("track_linear_velocity")
  assert cfg.rewards["visible_recovery_linear_velocity"].func.__name__ == (
    "track_visible_recovery_linear_velocity"
  )
  assert cfg.rewards["hidden_command_linear_velocity"].func.__name__ == (
    "track_hidden_linear_velocity"
  )
  assert cfg.rewards["visible_recovery_angular_velocity"].func.__name__ == (
    "track_visible_recovery_angular_velocity"
  )
  assert cfg.rewards["hidden_command_angular_velocity"].func.__name__ == (
    "track_hidden_angular_velocity"
  )

  assert set(cfg.metrics) >= {
    "user_command_error_xy",
    "user_command_error_yaw",
    "command_envelope_violation",
    "ball_control_success",
  }
  curriculum = cfg.curriculum["visibility_blend_task_levels"]
  stages = curriculum.params["stages"]
  assert stages[0]["episode_dropout_probability"] == pytest.approx(0.20)
  assert stages[-1]["episode_dropout_probability"] == pytest.approx(0.05)
  assert stages[-1]["lin_vel_x"] == (-0.5, 1.6)
  assert stages[0]["visible_ball_control_min"] == pytest.approx(0.10)
  assert stages[0]["envelope_compliance_min"] == pytest.approx(0.35)
  assert stages[-1]["visible_ball_control_min"] == pytest.approx(0.30)
  assert stages[-1]["envelope_compliance_min"] == pytest.approx(0.55)
  assert cfg.metrics["command_envelope_violation"].params[
    "smoothing_alpha"
  ] == pytest.approx(0.10)
  assert play_cfg.commands["twist"].ranges.lin_vel_x == (-0.5, 1.6)

  visual = cfg.observations["actor_history"].terms["ball_visible_mask"]
  assert visual.params["visibility_rise_alpha"] == pytest.approx(0.20)
  assert visual.params["visibility_fall_alpha"] == pytest.approx(0.05)


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
        assert (
          actor_history.terms["ball_visible_mask"].params["dropout_probability"] == 0.0
        )
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


def test_a1r0_history30_has_matched_history_and_full_receptive_field() -> None:
  task_id = "Mjlab-Velocity-Football-A1R0-History30-Flat-Unitree-G1"
  env_cfg = load_env_cfg(task_id)
  rl_cfg = cast(Any, load_rl_cfg(task_id))

  assert env_cfg.observations["actor_history"].history_length == 30
  assert env_cfg.observations["critic_history"].history_length == 30
  assert tuple(env_cfg.observations["actor_history"].terms) == (
    "ball_pos_b",
    "ball_to_feet_vectors_b",
    "ball_visible_mask",
  )
  assert tuple(env_cfg.observations["critic_history"].terms) == (
    "ball_pos_b",
    "ball_to_feet_vectors_b",
    "ball_visible_mask",
  )
  for group_name in ("actor_history", "critic_history"):
    for term in env_cfg.observations[group_name].terms.values():
      assert term.params["bias_range"] == 0.10
      assert term.params["frame_noise_range"] == 0.20
      assert term.params["dropout_probability"] == 0.0
  assert rl_cfg.actor.cnn_cfg["output_channels"] == (64, 64, 64, 64)
  assert rl_cfg.actor.cnn_cfg["dilations"] == (1, 2, 4, 8)
  assert rl_cfg.actor.cnn_cfg["causal"] is True
  assert rl_cfg.actor.cnn_cfg["output_mode"] == "last"
  assert rl_cfg.actor.cnn_cfg["activate_last"] is False
