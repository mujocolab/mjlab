"""Tests for the velocity-football scene configuration."""

import mujoco

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity_football.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_reward_ablation_flat_env_cfg,
)
from mjlab.tasks.velocity_football.config.g1.pose import (
  get_isaaclab_default_keyframe,
)
from mjlab.tasks.velocity_football.config.g1.velocity_env_cfgs import (
  unitree_g1_velocity_pretrain_flat_env_cfg,
)
from mjlab.tasks.velocity_football.football import (
  FOOTBALL_CONDIM,
  FOOTBALL_FRICTION,
  FOOTBALL_INITIAL_POS,
  FOOTBALL_MASS,
  FOOTBALL_RADIUS,
  FOOTBALL_RGBA,
  get_football_spec,
)
from mjlab.tasks.velocity_football.mdp.observations import (
  perceived_ball_pos_b,
  perceived_ball_to_feet_vectors_b,
)
from mjlab.tasks.velocity_football.mdp.velocity_command import (
  StopSkillVelocityReferenceCfg,
  UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity_football.velocity_football_env_cfg import (
  make_velocity_env_cfg,
)
from mjlab.utils.noise import UniformNoiseCfg


def test_football_scene_uses_infinite_plane_by_default() -> None:
  cfg = make_velocity_env_cfg()

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.scene.terrain.terrain_generator is None


def test_g1_football_scene_contains_robot() -> None:
  cfg = unitree_g1_flat_env_cfg()

  assert set(cfg.scene.entities) == {"robot", "ball"}
  assert cfg.scene.entities["robot"].articulation is not None
  assert cfg.viewer.entity_name == "robot"
  assert cfg.viewer.body_name == "torso_link"

  sensor_names = {sensor.name for sensor in cfg.scene.sensors}
  assert "feet_ground_contact" in sensor_names
  assert "self_collision" in sensor_names


def test_reward_ablation_variants_share_robust_perception_and_direct_commands() -> None:
  for variant in (
    "r0_isaaclab",
    "r1_e1",
    "r2_no_relative_velocity",
    "r3_no_relative_position",
  ):
    cfg = unitree_g1_reward_ablation_flat_env_cfg(variant)
    command = cfg.commands["twist"]
    assert isinstance(command, UniformVelocityCommandCfg)
    assert command.zero_command_ramp_time_range is None
    assert command.ball_relative_velocity_reference is None
    assert command.stop_skill_velocity_reference is None

    actor = cfg.observations["actor"]
    ball_position = actor.terms["ball_pos_b"]
    ball_to_feet = actor.terms["ball_to_feet_vectors_b"]
    assert ball_position.func is perceived_ball_pos_b
    assert ball_to_feet.func is perceived_ball_to_feet_vectors_b
    assert ball_position.params == {
      "bias_range": 0.10,
      "frame_noise_range": 0.06,
    }
    assert ball_to_feet.params["bias_range"] == 0.10
    assert ball_to_feet.params["frame_noise_range"] == 0.06
    assert ball_position.noise is None
    assert ball_to_feet.noise is None


def test_reward_ablation_variants_change_only_the_intended_task_rewards() -> None:
  r0 = unitree_g1_reward_ablation_flat_env_cfg("r0_isaaclab")
  assert r0.rewards["track_ball_lin_vel_xy_exp"].weight == 1.0
  assert r0.rewards["track_ball_lin_vel_xy_exp"].params["std"] == 0.5
  assert not r0.rewards["track_ball_lin_vel_xy_exp"].params["gate_by_position"]
  assert r0.rewards["track_angular_velocity"].weight == 2.0
  assert r0.rewards["track_ball_relative_vel_xy_exp"].weight == 0.0
  assert r0.rewards["track_ball_relative_pos_xy_exp"].weight == 0.0
  assert r0.rewards["ball_outside_control_zone"].weight == 0.0
  assert r0.rewards["ball_front_control"].weight == 0.5

  r1 = unitree_g1_reward_ablation_flat_env_cfg("r1_e1")
  assert r1.rewards["track_ball_relative_vel_xy_exp"].weight == 0.25
  assert r1.rewards["track_ball_relative_pos_xy_exp"].weight == 0.5
  assert r1.rewards["ball_outside_control_zone"].weight == -0.5

  r2 = unitree_g1_reward_ablation_flat_env_cfg("r2_no_relative_velocity")
  assert r2.rewards["track_ball_relative_vel_xy_exp"].weight == 0.0
  assert r2.rewards["track_ball_relative_pos_xy_exp"].weight == 0.5

  r3 = unitree_g1_reward_ablation_flat_env_cfg("r3_no_relative_position")
  assert r3.rewards["track_ball_relative_vel_xy_exp"].weight == 0.25
  assert r3.rewards["track_ball_relative_pos_xy_exp"].weight == 0.0
  assert r3.rewards["ball_outside_control_zone"].weight == 0.0


def test_pretrain_and_football_use_isaaclab_default_pose() -> None:
  pretrain_robot = unitree_g1_velocity_pretrain_flat_env_cfg().scene.entities["robot"]
  football_robot = unitree_g1_flat_env_cfg().scene.entities["robot"]
  expected_pose = get_isaaclab_default_keyframe()

  assert pretrain_robot.init_state == expected_pose
  assert football_robot.init_state == expected_pose
  assert pretrain_robot.init_state is not football_robot.init_state
  assert expected_pose.pos == (0.0, 0.0, 0.78)
  assert expected_pose.joint_pos == {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
    ".*_elbow_joint": 0.6,
  }


def test_football_entity_uses_confirmed_physical_properties() -> None:
  cfg = make_velocity_env_cfg()
  ball_cfg = cfg.scene.entities["ball"]
  spec = get_football_spec()

  assert ball_cfg.init_state.pos == FOOTBALL_INITIAL_POS

  ball_joint = next(joint for joint in spec.joints if joint.name == "ball_freejoint")
  assert ball_joint.type == mujoco.mjtJoint.mjJNT_FREE

  ball_geom = next(geom for geom in spec.geoms if geom.name == "ball_collision")
  assert ball_geom.type == mujoco.mjtGeom.mjGEOM_SPHERE
  assert ball_geom.size[0] == FOOTBALL_RADIUS
  assert ball_geom.mass == FOOTBALL_MASS
  assert tuple(ball_geom.rgba) == FOOTBALL_RGBA
  assert ball_geom.condim == FOOTBALL_CONDIM
  assert tuple(ball_geom.friction) == FOOTBALL_FRICTION
  assert ball_geom.group == 3

  model = spec.compile()
  ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_collision")
  assert model.geom_condim[ball_geom_id] == FOOTBALL_CONDIM
  assert tuple(model.geom_friction[ball_geom_id]) == FOOTBALL_FRICTION


def test_football_velocity_command_matches_reference() -> None:
  cfg = make_velocity_env_cfg()
  command = cfg.commands["twist"]

  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.resampling_time_range == (5.0, 6.0)
  assert command.rel_standing_envs == 0.05
  assert command.zero_command_ramp_time_range == (0.3, 0.5)
  assert command.ball_relative_velocity_reference is None
  assert command.stop_skill_velocity_reference is None
  assert command.rel_heading_envs == 1.0
  assert command.rel_forward_envs == 0.0
  assert command.heading_command
  assert command.heading_control_stiffness == 0.5
  assert command.ranges.lin_vel_x == (-0.25, 1.0)
  assert command.ranges.lin_vel_y == (-0.25, 0.25)
  assert command.ranges.ang_vel_z == (-1.0, 1.0)
  assert command.ranges.heading == (-mujoco.mjPI, mujoco.mjPI)


def test_g1_football_enables_stop_skill_velocity_reference() -> None:
  cfg = unitree_g1_flat_env_cfg()
  command = cfg.commands["twist"]

  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.rel_standing_envs == 0.05
  assert command.zero_command_ramp_time_range is None
  assert command.ball_relative_velocity_reference is None
  reference = command.stop_skill_velocity_reference
  assert isinstance(reference, StopSkillVelocityReferenceCfg)
  assert reference.rise_amplitude == 0.2
  assert reference.rise_duration == 0.3
  assert reference.fall_duration == 0.3
  assert reference.trigger_window == 5
  ball_velocity = cfg.rewards["track_ball_lin_vel_xy_exp"]
  assert ball_velocity.params["use_user_command"] is False
  assert ball_velocity.params["use_ball_command"] is True


def test_g1_play_keeps_reference_command_ranges() -> None:
  training_command = unitree_g1_flat_env_cfg().commands["twist"]
  play_command = unitree_g1_flat_env_cfg(play=True).commands["twist"]

  assert isinstance(training_command, UniformVelocityCommandCfg)
  assert isinstance(play_command, UniformVelocityCommandCfg)
  assert play_command.ranges == training_command.ranges


def test_actor_observations_match_football_reference() -> None:
  cfg = make_velocity_env_cfg()
  actor = cfg.observations["actor"]

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
  )
  assert actor.history_length == 5
  assert actor.flatten_history_dim
  assert actor.terms["phase"].params == {
    "period": 0.6,
    "command_name": "twist",
  }

  for term_name in ("actions", "ball_pos_b", "ball_to_feet_vectors_b"):
    term = actor.terms[term_name]
    assert term.delay_min_lag == 0
    assert term.delay_max_lag == 2

  ball_pos_noise = actor.terms["ball_pos_b"].noise
  feet_noise = actor.terms["ball_to_feet_vectors_b"].noise
  assert isinstance(ball_pos_noise, UniformNoiseCfg)
  assert isinstance(feet_noise, UniformNoiseCfg)
  assert (ball_pos_noise.n_min, ball_pos_noise.n_max) == (-0.06, 0.06)
  assert (feet_noise.n_min, feet_noise.n_max) == (-0.1, 0.1)

  feet_cfg = actor.terms["ball_to_feet_vectors_b"].params["asset_cfg"]
  assert isinstance(feet_cfg, SceneEntityCfg)
  assert feet_cfg.body_names == (r".*_ankle_roll_link",)


def test_critic_observations_match_football_reference() -> None:
  cfg = make_velocity_env_cfg()
  critic = cfg.observations["critic"]

  assert tuple(critic.terms) == (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "command",
    "phase",
    "joint_pos",
    "joint_vel",
    "actions",
    "ball_vel_b",
    "ball_pos_b",
    "ball_to_feet_vectors_b",
  )
  assert not critic.enable_corruption
  assert critic.history_length == 5
  assert critic.flatten_history_dim
  assert critic.terms["phase"].params == {
    "period": 0.6,
    "command_name": "twist",
  }
  assert critic.terms["actions"].clip == (-10.0, 10.0)

  for term in critic.terms.values():
    assert term.noise is None
    assert term.delay_min_lag == 0
    assert term.delay_max_lag == 0

  feet_cfg = critic.terms["ball_to_feet_vectors_b"].params["asset_cfg"]
  assert isinstance(feet_cfg, SceneEntityCfg)
  assert feet_cfg.body_names == (r".*_ankle_roll_link",)


def test_robot_regularization_adds_non_duplicate_reference_terms() -> None:
  cfg = make_velocity_env_cfg()

  terminated = cfg.rewards["is_terminated"]
  torque = cfg.rewards["joint_torques_l2"]
  acceleration = cfg.rewards["joint_acc_l2"]

  assert terminated.weight == -200.0
  assert torque.weight == -1e-5
  assert acceleration.weight == -1e-7

  torque_asset = torque.params["asset_cfg"]
  acceleration_asset = acceleration.params["asset_cfg"]
  assert isinstance(torque_asset, SceneEntityCfg)
  assert isinstance(acceleration_asset, SceneEntityCfg)
  assert torque_asset.actuator_names == r".*"
  assert acceleration_asset.joint_names == (r".*",)

  # Existing MJLab G1 regularizers remain active with their original weights.
  assert cfg.rewards["upright"].weight == 1.0
  assert cfg.rewards["pose"].weight == 1.0
  assert cfg.rewards["dof_pos_limits"].weight == -1.0
  assert cfg.rewards["action_rate_l2"].weight == -0.2
  assert "stop_ball_lin_vel_xy_exp" not in cfg.rewards


def test_football_core_rewards_are_connected_to_environment() -> None:
  cfg = make_velocity_env_cfg()

  relative_velocity = cfg.rewards["track_ball_relative_vel_xy_exp"]
  relative_position = cfg.rewards["track_ball_relative_pos_xy_exp"]

  assert relative_velocity.weight == 0.25
  assert relative_velocity.params == {"std": 0.5}
  assert relative_position.weight == 0.5
  assert relative_position.params == {
    "command_name": "twist",
    "anchor_x": 0.19,
    "anchor_x_speed_gain": 0.0,
    "anchor_x_range": (0.19, 0.19),
    "std_x": 0.5,
    "std_y": 0.5,
  }
  ball_velocity = cfg.rewards["track_ball_lin_vel_xy_exp"]
  assert ball_velocity.weight == 2.0
  assert ball_velocity.params["use_user_command"] is True
  assert "use_ball_command" not in ball_velocity.params
  assert cfg.rewards["ball_outside_control_zone"].weight == -0.5

  assert cfg.rewards["track_linear_velocity"].weight == 1.0
  assert cfg.rewards["track_angular_velocity"].weight == 1.5


def test_football_command_curriculum_matches_isaac_lab_reference() -> None:
  cfg = make_velocity_env_cfg()

  assert "command_vel" not in cfg.curriculum
  curriculum = cfg.curriculum["lin_vel_cmd_levels"]
  assert curriculum.params == {
    "command_name": "twist",
    "reward_term_name": "track_linear_velocity",
    "max_lin_vel_x": (-0.5, 2.0),
    "max_lin_vel_y": (-0.5, 0.5),
    "success_threshold": 0.7,
    "range_step": 0.1,
  }


def test_football_terminations_leave_margin_outside_visual_rectangle() -> None:
  cfg = make_velocity_env_cfg()

  assert cfg.terminations["fell_over"].params == {"limit_angle": 0.8}
  assert cfg.terminations["ball_out_of_control"].params == {
    "max_distance": 2.0,
    "min_forward": -0.20,
    "max_forward": 1.80,
    "max_lateral": 0.90,
    "max_height": 0.5,
    "ball_cfg": SceneEntityCfg("ball"),
  }
  assert not cfg.terminations["ball_out_of_control"].time_out


def test_football_reset_events_match_reference_ranges() -> None:
  cfg = make_velocity_env_cfg()

  assert "reset_base" not in cfg.events
  football_reset = cfg.events["reset_football"]
  assert football_reset.mode == "reset"
  assert football_reset.params == {
    "robot_cfg": SceneEntityCfg("robot"),
    "ball_cfg": SceneEntityCfg("ball"),
    "ball_radius": 0.1098,
    "robot_xy_noise_range": (-0.05, 0.05),
    "robot_yaw_range": (-3.14, 3.14),
    "ball_forward_range": (0.1, 0.5),
    "ball_lateral_range": (-0.15, 0.15),
    "ball_velocity_range": (-1.5, 1.5),
  }

  joint_reset = cfg.events["reset_robot_joints"]
  assert joint_reset.mode == "reset"
  assert joint_reset.params["position_range"] == (-0.1, 0.1)
  assert joint_reset.params["velocity_range"] == (0.0, 0.0)


def test_robot_push_event_matches_football_reference() -> None:
  cfg = make_velocity_env_cfg()
  push = cfg.events["push_robot"]

  assert push.mode == "interval"
  assert push.interval_range_s == (5.0, 6.0)
  assert push.params["velocity_range"] == {
    "x": (-0.5, 0.5),
    "y": (-0.3, 0.3),
    "z": (-0.2, 0.2),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
  }


def test_ball_sliding_friction_randomization_matches_reference() -> None:
  cfg = make_velocity_env_cfg()
  friction = cfg.events["ball_friction"]

  assert friction.mode == "startup"
  assert friction.params == {
    "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_collision",)),
    "operation": "abs",
    "ranges": (0.05, 0.15),
    "axes": [0],
    "shared_random": True,
  }
