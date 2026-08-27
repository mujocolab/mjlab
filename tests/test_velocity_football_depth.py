"""Configuration and observation tests for direct-depth football control."""

from types import SimpleNamespace
from typing import cast

import mujoco
import pytest
import torch

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity_football.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity_football_depth import (
  TASK_ID,
  TEMPORAL_CALIBRATED_FROZEN_MLP_DISTILLATION_TASK_ID,
  TEMPORAL_DEPLOYMENT_ROBUST_V2_DISTILLATION_TASK_ID,
  TEMPORAL_LONG_DROPOUT10_CAMERA_DR_DISTILLATION_TASK_ID,
  TEMPORAL_MOUNT_RANGE_FROZEN_MLP_DISTILLATION_TASK_ID,
  TEMPORAL_MOUNT_RANGE_STRONG_FROZEN_MLP_DISTILLATION_TASK_ID,
)
from mjlab.tasks.velocity_football_depth.env_cfg import (
  DEPTH_CAMERA_POSITION_DR_METERS,
  DEPTH_CAMERA_ROTATION_DR_RADIANS,
  DEPTH_HEIGHT,
  DEPTH_SENSOR_NAME,
  DEPTH_WIDTH,
)
from mjlab.tasks.velocity_football_depth.observations import normalized_camera_depth


def test_depth_football_task_registration_and_network_contract() -> None:
  assert TASK_ID in list_tasks()
  cfg = load_env_cfg(TASK_ID)
  rl_cfg = cast(RslRlOnPolicyRunnerCfg, load_rl_cfg(TASK_ID))

  assert tuple(cfg.observations) == ("actor", "critic", "critic_ball", "depth")
  assert rl_cfg.obs_groups == {
    "actor": ("actor", "depth"),
    "critic": ("critic", "depth", "critic_ball"),
  }
  assert rl_cfg.actor.class_name.endswith(":SpatialSoftmaxCNNModel")
  assert rl_cfg.critic.class_name.endswith(":SpatialSoftmaxCNNModel")
  assert not rl_cfg.algorithm.share_cnn_encoders


def test_depth_football_actor_and_critic_are_asymmetric() -> None:
  cfg = load_env_cfg(TASK_ID)
  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]

  assert actor.history_length == 5
  assert tuple(actor.terms) == (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "phase",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  assert "ball_pos_b" not in actor.terms
  assert "ball_pos_b" not in critic.terms
  assert "ball_vel_b" not in critic.terms
  assert "ball_to_feet_vectors_b" not in critic.terms
  critic_ball = cfg.observations["critic_ball"]
  assert critic_ball.history_length is None
  assert tuple(critic_ball.terms) == ("position",)


def test_depth_football_preserves_baseline_task_objectives() -> None:
  cfg = load_env_cfg(TASK_ID)
  baseline = unitree_g1_flat_env_cfg()

  assert cfg.rewards == baseline.rewards
  assert cfg.events == baseline.events
  assert cfg.curriculum == baseline.curriculum
  assert cfg.terminations == baseline.terminations
  assert cfg.commands == baseline.commands
  assert cfg.actions == baseline.actions


def test_depth_camera_matches_deployment_calibration() -> None:
  cfg = load_env_cfg(TASK_ID)
  cameras = [sensor for sensor in cfg.scene.sensors if sensor.name == DEPTH_SENSOR_NAME]
  assert len(cameras) == 1
  camera = cameras[0]
  assert isinstance(camera, CameraSensorCfg)
  assert camera.parent_body == "robot/torso_link"
  assert camera.pos == pytest.approx((0.1135993074, 0.01753, 0.3934754688))
  assert camera.quat == pytest.approx(
    (0.6980107470, 0.1130530720, -0.1130530720, -0.6980107470)
  )
  assert (camera.height, camera.width) == (DEPTH_HEIGHT, DEPTH_WIDTH)
  assert camera.data_types == ("depth",)


def test_normalized_camera_depth_sanitizes_invalid_pixels() -> None:
  raw = torch.tensor([[[[0.1], [0.6], [3.5], [float("nan")]]]])
  camera = SimpleNamespace(data=SimpleNamespace(depth=raw))
  env = SimpleNamespace(scene={DEPTH_SENSOR_NAME: camera})

  actual = normalized_camera_depth(
    env,  # type: ignore[arg-type]
    DEPTH_SENSOR_NAME,
    min_depth=0.2,
    max_depth=3.0,
  )

  assert actual.shape == (1, 1, 1, 4)
  torch.testing.assert_close(actual, torch.tensor([[[[1.0, 0.2, 1.0, 1.0]]]]))


def test_depth_ball_pixels_are_masked_only_in_sensor_hidden_envs() -> None:
  raw = torch.tensor(
    [
      [[[0.6], [0.9], [1.2]]],
      [[[0.6], [0.9], [1.2]]],
    ]
  )
  geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
  segmentation = torch.tensor(
    [
      [[[3, geom_type], [7, geom_type], [4, geom_type]]],
      [[[3, geom_type], [7, geom_type], [4, geom_type]]],
    ]
  )
  camera = SimpleNamespace(
    data=SimpleNamespace(depth=raw, segmentation=segmentation)
  )
  ball = SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.tensor([7])))
  env = SimpleNamespace(scene={DEPTH_SENSOR_NAME: camera, "ball": ball})
  env._football_masked_ball_visual = {
    "episode_hidden": torch.tensor([False, False]),
    "synthetic_hidden": torch.tensor([True, False]),
  }

  actual = normalized_camera_depth(
    env,  # type: ignore[arg-type]
    DEPTH_SENSOR_NAME,
    min_depth=0.2,
    max_depth=3.0,
    mask_ball_when_sensor_hidden=True,
  )

  expected = torch.tensor(
    [
      [[[0.2, 1.0, 0.4]]],
      [[[0.2, 0.3, 0.4]]],
    ]
  )
  torch.testing.assert_close(actual, expected)


def test_hidden_ball_inpainting_uses_background_depth_not_invalid_silhouette() -> None:
  raw = torch.tensor([[[[0.6], [0.7], [0.9], [1.2], [1.5]]]])
  geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
  segmentation = torch.tensor(
    [[[[3, geom_type], [3, geom_type], [7, geom_type], [4, geom_type], [4, geom_type]]]]
  )
  camera = SimpleNamespace(
    data=SimpleNamespace(depth=raw, segmentation=segmentation)
  )
  ball = SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.tensor([7])))
  env = SimpleNamespace(scene={DEPTH_SENSOR_NAME: camera, "ball": ball})
  env._football_masked_ball_visual = {
    "episode_hidden": torch.tensor([False]),
    "synthetic_hidden": torch.tensor([True]),
  }

  actual = normalized_camera_depth(
    env,  # type: ignore[arg-type]
    DEPTH_SENSOR_NAME,
    min_depth=0.2,
    max_depth=3.0,
    inpaint_ball_when_sensor_hidden=True,
  )

  assert actual[0, 0, 0, 2] < 1.0
  assert actual[0, 0, 0, 2] == pytest.approx(0.5)


def test_temporal_long_dropout_camera_dr_configuration() -> None:
  cfg = load_env_cfg(TEMPORAL_LONG_DROPOUT10_CAMERA_DR_DISTILLATION_TASK_ID)

  for term in cfg.observations["actor_history"].terms.values():
    assert term.params["transition_dropout_probability"] == pytest.approx(
      0.10 / 0.95
    )
    assert term.params["transition_dropout_until_end_probability"] == 1.0
  depth_term = cfg.observations["depth"].terms["image"]
  assert depth_term.params["mask_ball_when_sensor_hidden"] is True
  camera = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == DEPTH_SENSOR_NAME
  )
  assert camera.data_types == ("depth", "segmentation")

  position_event = cfg.events["randomize_depth_camera_position"]
  assert position_event.params["ranges"] == {
    0: (-DEPTH_CAMERA_POSITION_DR_METERS, DEPTH_CAMERA_POSITION_DR_METERS),
    1: (-DEPTH_CAMERA_POSITION_DR_METERS, DEPTH_CAMERA_POSITION_DR_METERS),
    2: (-DEPTH_CAMERA_POSITION_DR_METERS, DEPTH_CAMERA_POSITION_DR_METERS),
  }
  orientation_event = cfg.events["randomize_depth_camera_orientation"]
  assert orientation_event.params["roll_range"] == (
    -DEPTH_CAMERA_ROTATION_DR_RADIANS,
    DEPTH_CAMERA_ROTATION_DR_RADIANS,
  )


def test_deployment_robust_v2_visual_and_temporal_randomization() -> None:
  cfg = load_env_cfg(TEMPORAL_DEPLOYMENT_ROBUST_V2_DISTILLATION_TASK_ID)
  depth = cfg.observations["depth"].terms["image"]

  assert depth.params["inpaint_ball_when_sensor_hidden"] is True
  assert depth.params["mask_ball_when_sensor_hidden"] is False
  assert depth.params["depth_scale_range"] == (0.98, 1.02)
  assert depth.params["depth_bias_range"] == (-0.02, 0.02)
  assert depth.params["crop_shift_x_pixels"] == 4
  assert depth.params["crop_shift_y_pixels"] == 4
  assert depth.params["frame_repeat_probability"] == 0.10
  assert (depth.delay_min_lag, depth.delay_max_lag) == (0, 2)
  assert depth.delay_hold_prob == 0.90
  assert "randomize_depth_camera_extrinsics" in cfg.events
  fovy = cfg.events["randomize_depth_camera_fovy"]
  assert fovy.params["ranges"] == (40.5, 44.5)

  agent = load_rl_cfg(TEMPORAL_DEPLOYMENT_ROBUST_V2_DISTILLATION_TASK_ID)
  assert agent.algorithm.rollout_policy == "mixed"
  assert agent.algorithm.student_rollout_warmup_updates == 1_000
  assert agent.algorithm.student_rollout_ramp_updates == 3_000


def test_calibrated_frozen_mlp_visual_randomization_configuration() -> None:
  cfg = load_env_cfg(TEMPORAL_CALIBRATED_FROZEN_MLP_DISTILLATION_TASK_ID)
  depth = cfg.observations["depth"].terms["image"]

  for term in cfg.observations["actor_history"].terms.values():
    assert term.params["transition_dropout_probability"] == 0.0
    assert term.params["transition_dropout_until_end_probability"] == 0.0

  assert depth.params["depth_scale_range"] == (0.98, 1.02)
  assert depth.params["depth_bias_range"] == (-0.02, 0.02)
  assert depth.params["crop_shift_x_pixels"] == 4
  assert depth.params["crop_shift_y_pixels"] == 4
  assert depth.params["dropout_probability"] == 0.05
  assert depth.params["frame_repeat_probability"] == 0.0
  assert (depth.delay_min_lag, depth.delay_max_lag) == (0, 0)
  assert depth.delay_hold_prob == 0.0

  extrinsics = cfg.events["randomize_depth_camera_extrinsics"]
  assert extrinsics.params["first_position"] == extrinsics.params["second_position"]
  assert extrinsics.params["first_quaternion"] == extrinsics.params[
    "second_quaternion"
  ]
  assert extrinsics.params["position_residual_range"] == (-0.005, 0.005)
  assert extrinsics.params["rotation_residual_range"] == pytest.approx(
    (-DEPTH_CAMERA_ROTATION_DR_RADIANS, DEPTH_CAMERA_ROTATION_DR_RADIANS)
  )
  fovy = cfg.events["randomize_depth_camera_fovy"]
  assert fovy.params["ranges"] == (40.5, 44.5)

  agent = load_rl_cfg(TEMPORAL_CALIBRATED_FROZEN_MLP_DISTILLATION_TASK_ID)
  assert agent.student.cnn_cfg["freeze_coordinate_actor"] is True
  assert agent.max_iterations == 10_000


def test_mount_range_frozen_mlp_camera_randomization_configuration() -> None:
  cfg = load_env_cfg(TEMPORAL_MOUNT_RANGE_FROZEN_MLP_DISTILLATION_TASK_ID)
  event = cfg.events["randomize_depth_camera_extrinsics"]

  assert event.func.__name__ == "randomize_camera_between_uncertain_limits"
  assert event.params["alpha_range"] == (0.0, 0.25)
  assert event.params["fixed_lateral_position"] == pytest.approx(0.01753)
  assert event.params["lower_x_residual_range"] == (-0.03, 0.03)
  assert event.params["lower_z_residual_range"] == (-0.01, 0.01)
  assert event.params["lower_pitch_residual_range"] == pytest.approx(
    (-DEPTH_CAMERA_ROTATION_DR_RADIANS, DEPTH_CAMERA_ROTATION_DR_RADIANS)
  )

  depth = cfg.observations["depth"].terms["image"]
  assert depth.params["crop_shift_x_pixels"] == 1
  assert depth.params["crop_shift_y_pixels"] == 1
  assert depth.params["dropout_probability"] == 0.05
  assert depth.params["frame_repeat_probability"] == 0.0
  assert (depth.delay_min_lag, depth.delay_max_lag) == (0, 0)


def test_mount_range_strong_camera_randomization_configuration() -> None:
  cfg = load_env_cfg(TEMPORAL_MOUNT_RANGE_STRONG_FROZEN_MLP_DISTILLATION_TASK_ID)
  event = cfg.events["randomize_depth_camera_extrinsics"]

  assert event.params["alpha_range"] == (0.0, 0.35)
  assert event.params["fixed_lateral_position"] == pytest.approx(0.01753)
  assert event.params["lower_x_residual_range"] == (-0.035, 0.035)
  assert event.params["lower_z_residual_range"] == (-0.015, 0.015)
  assert event.params["lower_pitch_residual_range"] == pytest.approx(
    (-1.5 * DEPTH_CAMERA_ROTATION_DR_RADIANS, 1.5 * DEPTH_CAMERA_ROTATION_DR_RADIANS)
  )


@pytest.mark.parametrize(
  ("min_depth", "max_depth"),
  [(-0.1, 3.0), (0.2, 0.2), (1.0, 0.5)],
)
def test_normalized_camera_depth_rejects_invalid_ranges(
  min_depth: float,
  max_depth: float,
) -> None:
  env = SimpleNamespace(scene={})
  with pytest.raises(ValueError):
    normalized_camera_depth(
      env,  # type: ignore[arg-type]
      DEPTH_SENSOR_NAME,
      min_depth=min_depth,
      max_depth=max_depth,
    )
