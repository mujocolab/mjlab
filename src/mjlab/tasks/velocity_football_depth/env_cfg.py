"""Environment configuration for direct-depth G1 football control."""

from __future__ import annotations

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.velocity_football.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_isaaclab_aligned_flat_env_cfg,
  unitree_g1_long_dropout10_envelope30_legacy_curriculum_flat_env_cfg,
)
from mjlab.tasks.velocity_football.mdp.observations import ball_pos_b

from .events import (
  randomize_camera_between_calibrations,
  randomize_camera_between_uncertain_limits,
)
from .observations import (
  ball_auxiliary_target,
  normalized_camera_depth,
  normalized_camera_depth_frame,
)

DEPTH_SENSOR_NAME = "football_depth_camera"
# torso -> camera pose: this project's custom-remounted D435i position
# (distinct from Unitree's official factory mount), confirmed to match
# params/env.yaml of the trained temporal-distillation checkpoint
# (2026-08-19_10-19-55_DepthStudent_DirectLatent_..._10k). The 20260817
# four-point-fusion refined calibration (recommended, largest sample size)
# converts to (0.139264, 0.035343, 0.393266) / (0.6889379, 0.031495,
# -0.040079, -0.7230258) -- within 2.6cm/1.8cm/0.2mm of this value, much
# closer than the earlier not-deployment-ready fused result.
DEPTH_CAMERA_POS = (0.1135993074, 0.01753, 0.3934754688)
DEPTH_CAMERA_QUAT = (0.6980107470, 0.1130530720, -0.1130530720, -0.6980107470)
DEPTH_CAMERA_REFINED_POS = (0.139264, 0.035343, 0.393266)
DEPTH_CAMERA_REFINED_QUAT = (0.6889379, 0.031495, -0.040079, -0.7230258)
# Adjustable real-camera mount endpoints. The official URDF pose is the upper
# mechanical limit. The lower endpoint is the rough 2026-08-17 multiview
# optical calibration converted both to torso_link coordinates and MuJoCo's
# camera-axis convention. Its lateral coordinate is replaced with the official
# value by the constrained event.
DEPTH_CAMERA_UPPER_POS = (0.0576235, 0.01753, 0.42987)
DEPTH_CAMERA_UPPER_QUAT = (
  0.6592524821,
  0.2557071857,
  -0.2557071857,
  -0.6592524821,
)
DEPTH_CAMERA_LOWER_ESTIMATE_POS = (
  0.139264,
  0.035343,
  0.393266,
)
DEPTH_CAMERA_LOWER_ESTIMATE_QUAT = (
  0.6889379,
  0.031495,
  -0.040079,
  -0.7230258,
)
DEPTH_HEIGHT = 60
DEPTH_WIDTH = 80
DEPTH_MIN_METERS = 0.2
DEPTH_MAX_METERS = 3.0
DEPTH_HISTORY_LENGTH = 5
DEPTH_POLICY_HEIGHT = 30
DEPTH_POLICY_WIDTH = 40
TEMPORAL_TEACHER_DEPTH_HISTORY_LENGTH = 10
# Domain randomization toward real stereo depth cameras (e.g. RealSense
# D435), whose ideal MuJoCo rendering has neither dropped/invalid pixels nor
# range-dependent noise. Training-only; play/eval sees the clean render.
# Sampled from a generator private to the depth observation term (see
# observations._depth_noise_generator), never the env's seeded RNG stream.
# Values (already in meters) are adapted from Project-Instinct's InstinctMJ
# ``DepthSteroNoiseCfg`` defaults.
DEPTH_NEAR_NOISE_STD = 0.02
DEPTH_FAR_NOISE_STD = 0.08
DEPTH_FAR_DISTANCE = 2.0
DEPTH_DROPOUT_PROBABILITY = 0.02
DEPTH_CAMERA_POSITION_DR_METERS = 0.02
DEPTH_CAMERA_ROTATION_DR_RADIANS = math.radians(3.0)


def unitree_g1_depth_asymmetric_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a football task with depth-only ball perception for the Actor.

  Rewards, events, curricula, actions, commands, and terminations are inherited
  unchanged from the existing flat G1 football task. The Actor retains the
  490-dimensional five-frame proprioceptive prefix and receives one depth image.
  The Critic receives the same depth image plus the true yaw-frame ball position.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)

  depth_camera = CameraSensorCfg(
    name=DEPTH_SENSOR_NAME,
    parent_body="robot/torso_link",
    pos=DEPTH_CAMERA_POS,
    quat=DEPTH_CAMERA_QUAT,
    fovy=42.5,
    width=DEPTH_WIDTH,
    height=DEPTH_HEIGHT,
    data_types=("depth",),
    use_textures=False,
    use_shadows=False,
    enabled_geom_groups=(0, 2, 3),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_camera,)

  actor = cfg.observations["actor"]
  actor.terms.pop("ball_pos_b")
  actor.terms.pop("ball_to_feet_vectors_b")

  critic = cfg.observations["critic"]
  critic.terms.pop("ball_vel_b")
  critic.terms.pop("ball_pos_b")
  critic.terms.pop("ball_to_feet_vectors_b")

  cfg.observations["critic_ball"] = ObservationGroupCfg(
    terms={"position": ObservationTermCfg(func=ball_pos_b)},
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.observations["depth"] = ObservationGroupCfg(
    terms={
      "image": ObservationTermCfg(
        func=normalized_camera_depth,
        params={
          "sensor_name": DEPTH_SENSOR_NAME,
          "min_depth": DEPTH_MIN_METERS,
          "max_depth": DEPTH_MAX_METERS,
        },
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg


def unitree_g1_depth_auxiliary_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the V1 temporal-depth task with privileged supervision targets."""
  cfg = unitree_g1_depth_asymmetric_flat_env_cfg(play=play)

  sensors = list(cfg.scene.sensors or ())
  depth_camera = next(sensor for sensor in sensors if sensor.name == DEPTH_SENSOR_NAME)
  assert isinstance(depth_camera, CameraSensorCfg)
  depth_camera.data_types = ("depth", "segmentation")
  cfg.scene.sensors = tuple(sensors)

  cfg.observations["depth"] = ObservationGroupCfg(
    terms={
      "image": ObservationTermCfg(
        func=normalized_camera_depth_frame,
        params={
          "sensor_name": DEPTH_SENSOR_NAME,
          "min_depth": DEPTH_MIN_METERS,
          "max_depth": DEPTH_MAX_METERS,
          "output_size": (DEPTH_POLICY_HEIGHT, DEPTH_POLICY_WIDTH),
        },
      )
    },
    concatenate_terms=True,
    history_length=DEPTH_HISTORY_LENGTH,
    flatten_history_dim=False,
    enable_corruption=False,
  )
  cfg.observations["ball_aux_target"] = ObservationGroupCfg(
    terms={
      "target": ObservationTermCfg(
        func=ball_auxiliary_target,
        params={
          "sensor_name": DEPTH_SENSOR_NAME,
          "max_position": DEPTH_MAX_METERS,
        },
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg


def unitree_g1_depth_teacher_student_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Expose synchronized coordinate-Teacher and depth-Student observations."""
  cfg = unitree_g1_isaaclab_aligned_flat_env_cfg(play=play)

  depth_camera = CameraSensorCfg(
    name=DEPTH_SENSOR_NAME,
    parent_body="robot/torso_link",
    pos=DEPTH_CAMERA_POS,
    quat=DEPTH_CAMERA_QUAT,
    fovy=42.5,
    width=DEPTH_WIDTH,
    height=DEPTH_HEIGHT,
    data_types=("depth",),
    use_textures=False,
    use_shadows=False,
    enabled_geom_groups=(0, 2, 3),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_camera,)

  teacher_actor = cfg.observations.pop("actor")
  cfg.observations["teacher_actor"] = teacher_actor
  student_proprio = deepcopy(teacher_actor)
  for term_name in ("ball_pos_b", "ball_to_feet_vectors_b", "ball_visible_mask"):
    student_proprio.terms.pop(term_name)
  cfg.observations["student_proprio"] = student_proprio
  cfg.observations["depth"] = ObservationGroupCfg(
    terms={
      "image": ObservationTermCfg(
        func=normalized_camera_depth_frame,
        params={
          "sensor_name": DEPTH_SENSOR_NAME,
          "min_depth": DEPTH_MIN_METERS,
          "max_depth": DEPTH_MAX_METERS,
          "output_size": (DEPTH_POLICY_HEIGHT, DEPTH_POLICY_WIDTH),
        },
      )
    },
    concatenate_terms=True,
    history_length=DEPTH_HISTORY_LENGTH,
    flatten_history_dim=False,
    enable_corruption=False,
  )

  # The PPO fine-tuning Critic uses only the current 110-D privileged state.
  # The coordinate Teacher target remains an observation group but is never
  # selected by the Student model during inference.
  cfg.observations.pop("critic_history", None)
  return cfg


def unitree_g1_depth_temporal_teacher_student_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Expose the B1 coordinate Teacher and a coordinate-free depth Student."""
  cfg = unitree_g1_long_dropout10_envelope30_legacy_curriculum_flat_env_cfg(play=play)

  # Artificial coordinate dropout would leave the rendered depth image intact,
  # creating contradictory supervision. Distillation uses natural camera/FOV
  # visibility; the frozen Teacher still receives its direct (B,T,7) coordinate
  # history, including the visibility mask, noise, and configured delay.
  if not play:
    for term in cfg.observations["actor_history"].terms.values():
      term.params["transition_dropout_probability"] = 0.0
      term.params["transition_dropout_until_end_probability"] = 0.0

  depth_camera = CameraSensorCfg(
    name=DEPTH_SENSOR_NAME,
    parent_body="robot/torso_link",
    pos=DEPTH_CAMERA_POS,
    quat=DEPTH_CAMERA_QUAT,
    fovy=42.5,
    width=DEPTH_WIDTH,
    height=DEPTH_HEIGHT,
    data_types=("depth",),
    use_textures=False,
    use_shadows=False,
    enabled_geom_groups=(0, 2, 3),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_camera,)

  # B1's actor is already the 490-D five-frame proprioceptive prefix. The
  # direct football coordinates live only in actor_history and are never read
  # by DepthTemporalLatentStudentModel.get_latent().
  cfg.observations["student_proprio"] = deepcopy(cfg.observations["actor"])
  cfg.observations["depth"] = ObservationGroupCfg(
    terms={
      "image": ObservationTermCfg(
        func=normalized_camera_depth_frame,
        params={
          "sensor_name": DEPTH_SENSOR_NAME,
          "min_depth": DEPTH_MIN_METERS,
          "max_depth": DEPTH_MAX_METERS,
          "output_size": (DEPTH_POLICY_HEIGHT, DEPTH_POLICY_WIDTH),
          "near_noise_std": 0.0 if play else DEPTH_NEAR_NOISE_STD,
          "far_noise_std": 0.0 if play else DEPTH_FAR_NOISE_STD,
          "far_distance": DEPTH_FAR_DISTANCE,
          "dropout_probability": 0.0 if play else DEPTH_DROPOUT_PROBABILITY,
        },
      )
    },
    concatenate_terms=True,
    history_length=TEMPORAL_TEACHER_DEPTH_HISTORY_LENGTH,
    flatten_history_dim=False,
    enable_corruption=False,
  )
  cfg.observations.pop("critic_history", None)
  return cfg


def unitree_g1_depth_temporal_long_dropout10_camera_dr_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Temporal depth distillation with aligned long ball loss and camera DR."""
  cfg = unitree_g1_depth_temporal_teacher_student_flat_env_cfg(play=play)

  if not play:
    # Match the coordinate Teacher's unconditional 10% long sensor-loss share:
    # sample only from the 95% non-standing episodes, start at 2--6 seconds,
    # and keep the ball hidden until the episode ends.
    transition_probability = 0.10 / 0.95
    for term in cfg.observations["actor_history"].terms.values():
      term.params["transition_dropout_probability"] = transition_probability
      term.params["transition_dropout_until_end_probability"] = 1.0
      term.params["transition_excluded_standing_command_name"] = "twist"

    depth_camera = next(
      sensor
      for sensor in cfg.scene.sensors or ()
      if sensor.name == DEPTH_SENSOR_NAME
    )
    assert isinstance(depth_camera, CameraSensorCfg)
    depth_camera.data_types = ("depth", "segmentation")
    depth_term = cfg.observations["depth"].terms["image"]
    depth_term.params["mask_ball_when_sensor_hidden"] = True

    camera_cfg = SceneEntityCfg(
      "robot",
      camera_names=(DEPTH_SENSOR_NAME,),
    )
    position_range = DEPTH_CAMERA_POSITION_DR_METERS
    cfg.events["randomize_depth_camera_position"] = EventTermCfg(
      func=envs_mdp.dr.cam_pos,
      mode="reset",
      params={
        "ranges": {
          0: (-position_range, position_range),
          1: (-position_range, position_range),
          2: (-position_range, position_range),
        },
        "operation": "add",
        "asset_cfg": camera_cfg,
      },
    )
    rotation_range = DEPTH_CAMERA_ROTATION_DR_RADIANS
    cfg.events["randomize_depth_camera_orientation"] = EventTermCfg(
      func=envs_mdp.dr.cam_quat,
      mode="reset",
      params={
        "roll_range": (-rotation_range, rotation_range),
        "pitch_range": (-rotation_range, rotation_range),
        "yaw_range": (-rotation_range, rotation_range),
        "asset_cfg": camera_cfg,
      },
    )

  return cfg


def unitree_g1_depth_temporal_deployment_robust_v2_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Deployment-robust temporal Student with calibrated visual randomization."""
  cfg = unitree_g1_depth_temporal_long_dropout10_camera_dr_flat_env_cfg(play=play)
  if play:
    return cfg

  camera_cfg = SceneEntityCfg("robot", camera_names=(DEPTH_SENSOR_NAME,))
  cfg.events.pop("randomize_depth_camera_position")
  cfg.events.pop("randomize_depth_camera_orientation")
  cfg.events["randomize_depth_camera_extrinsics"] = EventTermCfg(
    func=randomize_camera_between_calibrations,
    mode="reset",
    params={
      "first_position": DEPTH_CAMERA_POS,
      "first_quaternion": DEPTH_CAMERA_QUAT,
      "second_position": DEPTH_CAMERA_REFINED_POS,
      "second_quaternion": DEPTH_CAMERA_REFINED_QUAT,
      "position_residual_range": (-0.005, 0.005),
      "rotation_residual_range": (-math.radians(3.0), math.radians(3.0)),
      "asset_cfg": camera_cfg,
    },
  )
  cfg.events["randomize_depth_camera_fovy"] = EventTermCfg(
    func=envs_mdp.dr.cam_fovy,
    mode="reset",
    params={
      "ranges": (40.5, 44.5),
      "operation": "abs",
      "asset_cfg": camera_cfg,
    },
  )

  depth_term = cfg.observations["depth"].terms["image"]
  depth_term.params.update(
    {
      "mask_ball_when_sensor_hidden": False,
      "inpaint_ball_when_sensor_hidden": True,
      "depth_scale_range": (0.98, 1.02),
      "depth_bias_range": (-0.02, 0.02),
      "crop_shift_x_pixels": 4,
      "crop_shift_y_pixels": 4,
      "frame_repeat_probability": 0.10,
    }
  )
  depth_term.delay_min_lag = 0
  depth_term.delay_max_lag = 2
  depth_term.delay_hold_prob = 0.90
  return cfg


def unitree_g1_depth_temporal_calibrated_visual_dr_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Frozen-MLP distillation around the measured deployment calibration.

  Camera extrinsics/FOV and depth scale/bias/crop are sampled once per episode.
  The depth stream keeps independent pixel noise, but deliberately has no frame
  repetition, observation latency, or artificial long ball-loss episodes.
  """
  cfg = unitree_g1_depth_temporal_deployment_robust_v2_flat_env_cfg(play=play)
  depth_camera = next(
    sensor
    for sensor in cfg.scene.sensors or ()
    if sensor.name == DEPTH_SENSOR_NAME
  )
  assert isinstance(depth_camera, CameraSensorCfg)

  if play:
    depth_camera.pos = DEPTH_CAMERA_REFINED_POS
    depth_camera.quat = DEPTH_CAMERA_REFINED_QUAT
    depth_camera.fovy = 42.5
    return cfg

  # Center every episode on the measured deployment extrinsics, rather than
  # interpolating back toward the legacy estimate. The shared event adds one
  # fixed translation/RPY residual for the duration of that episode.
  extrinsics = cfg.events["randomize_depth_camera_extrinsics"]
  extrinsics.params["first_position"] = DEPTH_CAMERA_REFINED_POS
  extrinsics.params["first_quaternion"] = DEPTH_CAMERA_REFINED_QUAT
  extrinsics.params["second_position"] = DEPTH_CAMERA_REFINED_POS
  extrinsics.params["second_quaternion"] = DEPTH_CAMERA_REFINED_QUAT

  # Disable synthetic 2--6 second long ball loss. Natural FOV/occlusion
  # visibility remains synchronized between the depth image and Teacher.
  for term in cfg.observations["actor_history"].terms.values():
    term.params["transition_dropout_probability"] = 0.0
    term.params["transition_dropout_until_end_probability"] = 0.0

  depth_term = cfg.observations["depth"].terms["image"]
  depth_term.params["dropout_probability"] = 0.05
  depth_term.params["frame_repeat_probability"] = 0.0
  depth_term.delay_min_lag = 0
  depth_term.delay_max_lag = 0
  depth_term.delay_hold_prob = 0.0
  return cfg


def unitree_g1_depth_temporal_mount_range_visual_dr_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Depth Student with physically constrained adjustable-mount extrinsics."""
  cfg = unitree_g1_depth_temporal_calibrated_visual_dr_flat_env_cfg(play=play)
  depth_camera = next(
    sensor
    for sensor in cfg.scene.sensors or ()
    if sensor.name == DEPTH_SENSOR_NAME
  )
  assert isinstance(depth_camera, CameraSensorCfg)
  lower_fixed_y = (
    DEPTH_CAMERA_LOWER_ESTIMATE_POS[0],
    DEPTH_CAMERA_UPPER_POS[1],
    DEPTH_CAMERA_LOWER_ESTIMATE_POS[2],
  )
  if play:
    depth_camera.pos = lower_fixed_y
    depth_camera.quat = DEPTH_CAMERA_LOWER_ESTIMATE_QUAT
    depth_camera.fovy = 42.5
    return cfg

  camera_cfg = SceneEntityCfg("robot", camera_names=(DEPTH_SENSOR_NAME,))
  cfg.events["randomize_depth_camera_extrinsics"] = EventTermCfg(
    func=randomize_camera_between_uncertain_limits,
    mode="reset",
    params={
      "lower_position": DEPTH_CAMERA_LOWER_ESTIMATE_POS,
      "lower_quaternion": DEPTH_CAMERA_LOWER_ESTIMATE_QUAT,
      "upper_position": DEPTH_CAMERA_UPPER_POS,
      "upper_quaternion": DEPTH_CAMERA_UPPER_QUAT,
      # At 40.5 degrees FOV with a +/-4 raw-pixel crop, both feet reach the
      # image boundary by alpha=0.30. Keep a 0.05 pose-space safety margin.
      "alpha_range": (0.0, 0.25),
      "fixed_lateral_position": DEPTH_CAMERA_UPPER_POS[1],
      "lower_x_residual_range": (-0.03, 0.03),
      "lower_z_residual_range": (-0.01, 0.01),
      "lower_pitch_residual_range": (-math.radians(3.0), math.radians(3.0)),
      "asset_cfg": camera_cfg,
    },
  )
  # The real stream is 240 pixels high before resampling to 60 pixels, so a
  # +/-4 raw-pixel principal-point error is +/-1 policy-image pixel.
  depth_term = cfg.observations["depth"].terms["image"]
  depth_term.params["crop_shift_x_pixels"] = 1
  depth_term.params["crop_shift_y_pixels"] = 1
  return cfg


def unitree_g1_depth_temporal_mount_range_strong_visual_dr_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Broaden the physical mount range after real-camera validation."""
  cfg = unitree_g1_depth_temporal_mount_range_visual_dr_flat_env_cfg(play=play)
  if play:
    return cfg

  extrinsics = cfg.events["randomize_depth_camera_extrinsics"]
  extrinsics.params["alpha_range"] = (0.0, 0.35)
  extrinsics.params["lower_x_residual_range"] = (-0.035, 0.035)
  extrinsics.params["lower_z_residual_range"] = (-0.015, 0.015)
  extrinsics.params["lower_pitch_residual_range"] = (
    -math.radians(4.5),
    math.radians(4.5),
  )
  return cfg
