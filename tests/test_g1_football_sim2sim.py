from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest

from mjlab.scripts.sim2sim.d435_ball_observer import (
  D435_CAMERA_NAME,
  D435Config,
  YoloBallDetector,
  camera_point_to_yaw,
  project_bbox_bottom_to_ground_yaw,
)
from mjlab.scripts.sim2sim.g1_football import (
  B1_HISTORY_OBS_DIM,
  EXPECTED_ACTION_DIM,
  EXPECTED_OBS_DIM,
  EXPECTED_OBSERVATION_NAMES,
  FRAME_STACK,
  ISAACLAB_ALIGNED_OBS_DIM,
  PROPRIOCEPTIVE_OBS_DIM,
  PROPRIOCEPTIVE_OBSERVATION_NAMES,
  TEMPORAL_HISTORY_LENGTH,
  TEMPORAL_OBS_DIM,
  TEMPORAL_OBSERVATION_NAMES,
  TEMPORAL_TERM_DIMS,
  TERM_DIMS,
  KeyboardController,
  ModelBindings,
  ObservationAssembler,
  PolicyMetadata,
  Sim2SimCfg,
  build_model,
  compute_football_observation,
  configure_tracking_camera,
)


def make_terms(fill: float = 0.0) -> dict[str, np.ndarray]:
  return {name: np.full(dim, fill, dtype=np.float32) for name, dim in TERM_DIMS.items()}


def make_session(
  *, input_dim: int = EXPECTED_OBS_DIM, output_dim: int = EXPECTED_ACTION_DIM
) -> Any:
  observation_names = (
    TEMPORAL_OBSERVATION_NAMES
    if input_dim == ISAACLAB_ALIGNED_OBS_DIM
    else EXPECTED_OBSERVATION_NAMES
  )
  metadata = {
    "joint_names": ",".join(f"joint_{index}" for index in range(29)),
    "default_joint_pos": ",".join("0" for _ in range(29)),
    "action_scale": ",".join("0.5" for _ in range(29)),
    "observation_names": ",".join(observation_names),
    "observation_terms_history_length": ",".join(
      "5" for _ in observation_names
    ),
  }
  return SimpleNamespace(
    get_inputs=lambda: [SimpleNamespace(name="obs", shape=[1, input_dim])],
    get_outputs=lambda: [SimpleNamespace(name="actions", shape=[1, output_dim])],
    get_modelmeta=lambda: SimpleNamespace(custom_metadata_map=metadata),
  )


def make_temporal_session(
  history_length: int = TEMPORAL_HISTORY_LENGTH,
  history_dim: int = TEMPORAL_OBS_DIM,
  input_dim: int = TEMPORAL_OBS_DIM,
) -> Any:
  b1_stacked = input_dim == PROPRIOCEPTIVE_OBS_DIM
  observation_names = (
    PROPRIOCEPTIVE_OBSERVATION_NAMES if b1_stacked else TEMPORAL_OBSERVATION_NAMES
  )
  observation_history = "5" if b1_stacked else "0"
  metadata = {
    "joint_names": ",".join(f"joint_{index}" for index in range(29)),
    "default_joint_pos": ",".join("0" for _ in range(29)),
    "action_scale": ",".join("0.5" for _ in range(29)),
    "observation_names": ",".join(observation_names),
    "observation_terms_history_length": ",".join(
      observation_history for _ in observation_names
    ),
  }
  return SimpleNamespace(
    get_inputs=lambda: [
      SimpleNamespace(name="obs", shape=[1, input_dim]),
      SimpleNamespace(
        name="obs_history",
        shape=[1, history_length, history_dim],
      ),
    ],
    get_outputs=lambda: [
      SimpleNamespace(name="actions", shape=[1, EXPECTED_ACTION_DIM])
    ],
    get_modelmeta=lambda: SimpleNamespace(custom_metadata_map=metadata),
  )


def test_observation_assembler_uses_term_major_five_frame_history() -> None:
  assembler = ObservationAssembler()
  obs = assembler.reset(make_terms(1.0))

  assert obs.shape == (EXPECTED_OBS_DIM,)
  offset = 0
  for dim in TERM_DIMS.values():
    term_history = obs[offset : offset + FRAME_STACK * dim]
    np.testing.assert_array_equal(term_history, 1.0)
    offset += FRAME_STACK * dim


def test_observation_assembler_uses_latest_action_without_deployment_delay() -> None:
  assembler = ObservationAssembler()
  assembler.reset(make_terms())
  terms = make_terms()
  terms["actions"][:] = 1.0
  obs = assembler.append(terms)

  action_index = EXPECTED_OBSERVATION_NAMES.index("actions")
  action_offset = FRAME_STACK * sum(
    TERM_DIMS[name] for name in EXPECTED_OBSERVATION_NAMES[:action_index]
  )
  action_history = obs[
    action_offset : action_offset + FRAME_STACK * EXPECTED_ACTION_DIM
  ].reshape(FRAME_STACK, EXPECTED_ACTION_DIM)
  np.testing.assert_array_equal(action_history[-1], 1.0)
  np.testing.assert_array_equal(action_history[:-1], 0.0)


def test_football_observation_is_yaw_xy_and_points_from_ankles_to_ball() -> None:
  ball_pos, feet_to_ball = compute_football_observation(
    root_pos_w=(1.0, 2.0, 0.5),
    root_quat_w=(1.0, 0.0, 0.0, 0.0),
    ball_pos_w=(1.3, 2.0, 0.1),
    feet_pos_w=((1.1, 2.2, 0.0), (1.2, 1.8, 0.0)),
  )

  np.testing.assert_allclose(ball_pos, (0.3, 0.0), atol=1e-6)
  np.testing.assert_allclose(
    feet_to_ball,
    (0.2, -0.2, 0.1, 0.2),
    atol=1e-6,
  )


def test_policy_metadata_validates_dimensions_and_layout() -> None:
  metadata = PolicyMetadata.from_session(make_session())

  assert len(metadata.joint_names) == EXPECTED_ACTION_DIM
  assert metadata.observation_names == EXPECTED_OBSERVATION_NAMES


def test_isaaclab_aligned_policy_uses_stacked_visibility_mask() -> None:
  metadata = PolicyMetadata.from_session(
    make_session(input_dim=ISAACLAB_ALIGNED_OBS_DIM)
  )
  assembler = ObservationAssembler(metadata)
  terms = {
    name: np.ones(dim, dtype=np.float32) for name, dim in TEMPORAL_TERM_DIMS.items()
  }

  obs = assembler.reset(terms)
  inputs = assembler.policy_inputs(obs)

  assert not metadata.is_temporal
  assert metadata.observation_names == TEMPORAL_OBSERVATION_NAMES
  assert obs.shape == (ISAACLAB_ALIGNED_OBS_DIM,)
  assert inputs["obs"].shape == (1, ISAACLAB_ALIGNED_OBS_DIM)


def test_temporal_policy_metadata_and_assembler_build_dual_inputs() -> None:
  metadata = PolicyMetadata.from_session(make_temporal_session())
  assembler = ObservationAssembler(metadata)
  terms = {
    name: np.ones(dim, dtype=np.float32) for name, dim in TEMPORAL_TERM_DIMS.items()
  }

  obs = assembler.reset(terms)
  inputs = assembler.policy_inputs(obs)

  assert metadata.is_temporal
  assert inputs["obs"].shape == (1, TEMPORAL_OBS_DIM)
  assert inputs["obs_history"].shape == (
    1,
    TEMPORAL_HISTORY_LENGTH,
    TEMPORAL_OBS_DIM,
  )


def test_b1_policy_assembler_uses_only_seven_ball_history_features() -> None:
  metadata = PolicyMetadata.from_session(
    make_temporal_session(
      history_dim=B1_HISTORY_OBS_DIM,
      input_dim=PROPRIOCEPTIVE_OBS_DIM,
    )
  )
  assembler = ObservationAssembler(metadata)
  terms = {
    name: np.full(dim, index + 1, dtype=np.float32)
    for index, (name, dim) in enumerate(TEMPORAL_TERM_DIMS.items())
  }

  obs = assembler.reset(terms)
  inputs = assembler.policy_inputs(obs)

  assert inputs["obs"].shape == (1, PROPRIOCEPTIVE_OBS_DIM)
  assert inputs["obs_history"].shape == (
    1,
    TEMPORAL_HISTORY_LENGTH,
    B1_HISTORY_OBS_DIM,
  )
  expected_latest = np.concatenate(
    [
      terms["ball_pos_b"],
      terms["ball_to_feet_vectors_b"],
      terms["ball_visible_mask"],
    ]
  )
  np.testing.assert_array_equal(inputs["obs_history"][0, -1], expected_latest)
  assert metadata.observation_names == PROPRIOCEPTIVE_OBSERVATION_NAMES


@pytest.mark.parametrize("history_length", [5, 10, 20])
def test_temporal_policy_uses_history_length_from_onnx(
  history_length: int,
) -> None:
  metadata = PolicyMetadata.from_session(make_temporal_session(history_length))
  assembler = ObservationAssembler(metadata)
  terms = {
    name: np.ones(dim, dtype=np.float32) for name, dim in TEMPORAL_TERM_DIMS.items()
  }

  obs = assembler.reset(terms)
  inputs = assembler.policy_inputs(obs)

  assert metadata.temporal_history_length == history_length
  assert inputs["obs_history"].shape == (
    1,
    history_length,
    TEMPORAL_OBS_DIM,
  )


def test_policy_metadata_rejects_previous_535_observation_policy() -> None:
  with pytest.raises(ValueError, match="535 observations"):
    PolicyMetadata.from_session(make_session(input_dim=535))


def test_d435_optical_point_transforms_to_yaw_frame() -> None:
  point = camera_point_to_yaw(
    (0.2, 0.1, 1.0),
    (0.0, 0.0, 0.0),
    np.eye(3),
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
  )

  np.testing.assert_allclose(point, (0.2, -0.1, -1.0), atol=1e-6)


def test_d435_defaults_match_deployment_camera_contract() -> None:
  cfg = D435Config()

  assert cfg.width == 640
  assert cfg.height == 480
  assert cfg.rgb_fovy_deg == pytest.approx(42.5)
  assert cfg.depth_fovy_deg == pytest.approx(58.0)
  assert cfg.min_depth == pytest.approx(0.3)
  assert cfg.max_depth == pytest.approx(3.0)
  assert cfg.depth_roi_px == 4
  assert cfg.confidence_threshold == pytest.approx(0.5)
  assert cfg.camera_pos_torso == pytest.approx((0.0576235, 0.01753, 0.42987))
  assert cfg.camera_quat_torso_wxyz == pytest.approx(
    (0.6592524821, 0.2557071857, -0.2557071857, -0.6592524821)
  )


def test_sim2sim_defaults_to_robocup_visual_observation() -> None:
  cfg = Sim2SimCfg()

  assert cfg.ball_observer == "robocup"
  assert cfg.yolo_confidence is None


def test_robocup_yolo_preprocess_uses_top_left_black_padding() -> None:
  detector = object.__new__(YoloBallDetector)
  detector.cfg = D435Config(vision_mode="robocup")
  detector.input_width = 4
  detector.input_height = 4
  rgb = np.full((2, 4, 3), 255, dtype=np.uint8)

  tensor, scale, pad_x, pad_y = detector._preprocess(rgb)

  assert scale == pytest.approx(1.0)
  assert pad_x == 0.0
  assert pad_y == 0.0
  np.testing.assert_allclose(tensor[0, :, :2, :], 1.0)
  np.testing.assert_array_equal(tensor[0, :, 2:, :], 0.0)


def test_robocup_bbox_bottom_ray_intersects_ground_plane() -> None:
  position = project_bbox_bottom_to_ground_yaw(
    box=(40.0, 30.0, 60.0, 60.0),
    intrinsics=(100.0, 100.0, 50.0, 50.0),
    camera_pos_w=(0.0, 0.0, 1.0),
    camera_rotation_w=np.eye(3),
    root_pos_w=(0.0, 0.0, 0.0),
    root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
  )

  assert position is not None
  np.testing.assert_allclose(position, (0.0, -0.1, 0.0), atol=1.0e-6)


def test_d435_camera_is_attached_to_torso_with_rgb_fov() -> None:
  model, _, _ = build_model()
  camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, D435_CAMERA_NAME)
  torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")

  assert model.cam_bodyid[camera_id] == torso_id
  assert model.cam_fovy[camera_id] == pytest.approx(42.5)


def test_yolo_decoder_matches_deployment_direct_detection_layout() -> None:
  detector = object.__new__(YoloBallDetector)
  detector.cfg = D435Config(confidence_threshold=0.5)
  detections = detector._decode_output(
    np.asarray(
      [
        [
          [10.0, 20.0],
          [11.0, 21.0],
          [30.0, 40.0],
          [31.0, 41.0],
          [0.9, 0.4],
          [0.0, 0.0],
        ]
      ],
      dtype=np.float32,
    ),
    image_width=640,
    image_height=480,
    scale=1.0,
    pad_x=0.0,
    pad_y=0.0,
  )

  assert len(detections) == 1
  np.testing.assert_allclose(detections[0][0], (10.0, 11.0, 30.0, 31.0))
  assert detections[0][1] == pytest.approx(0.9)


def test_tracking_camera_follows_pelvis_with_fixed_view_angles() -> None:
  camera = mujoco.MjvCamera()

  configure_tracking_camera(
    camera,
    body_id=7,
    distance=3.0,
    azimuth=90.0,
    elevation=-5.0,
  )

  assert camera.type == mujoco.mjtCamera.mjCAMERA_TRACKING
  assert camera.trackbodyid == 7
  assert camera.distance == pytest.approx(3.0)
  assert camera.azimuth == pytest.approx(90.0)
  assert camera.elevation == pytest.approx(-5.0)


def test_numeric_keyboard_controls_velocity_command_and_stop() -> None:
  controller = KeyboardController(np.zeros(3, dtype=np.float32))

  for key in ("8", "4", "7"):
    controller(ord(key))
  np.testing.assert_allclose(controller.command, (0.1, 0.1, 0.1))

  for key in ("2", "6", "9"):
    controller(ord(key))
  np.testing.assert_allclose(controller.command, (0.0, 0.0, 0.0), atol=1e-7)

  controller(ord("8"))
  controller(ord("5"))
  np.testing.assert_array_equal(controller.command, 0.0)

  controller(ord("R"))
  assert controller.reset_requested


def test_keyboard_command_limits_match_full_training_range() -> None:
  controller = KeyboardController(np.zeros(3, dtype=np.float32))

  for _ in range(30):
    controller(ord("8"))
    controller(ord("4"))
    controller(ord("7"))
  np.testing.assert_allclose(controller.command, (2.0, 0.5, 1.0), atol=1e-7)

  for _ in range(40):
    controller(ord("2"))
    controller(ord("6"))
    controller(ord("9"))
  np.testing.assert_allclose(controller.command, (-0.5, -0.5, -1.0), atol=1e-7)


def test_sim2sim_rejects_initial_command_outside_training_range() -> None:
  with pytest.raises(ValueError, match="outside the trained range"):
    Sim2SimCfg(command_x=2.1)

  cfg = Sim2SimCfg(command_x=2.0, command_y=-0.5, command_yaw=1.0)
  assert cfg.command_x == pytest.approx(2.0)


def test_numeric_keypad_uses_the_same_command_mapping() -> None:
  controller = KeyboardController(np.zeros(3, dtype=np.float32))

  controller(328)  # GLFW_KEY_KP_8
  controller(324)  # GLFW_KEY_KP_4
  controller(327)  # GLFW_KEY_KP_7

  np.testing.assert_allclose(controller.command, (0.1, 0.1, 0.1))


def test_task_model_contains_policy_joints_ball_and_native_timing() -> None:
  model, timestep, decimation = build_model()
  joint_names = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
  )

  bindings = ModelBindings.from_model(model, joint_names)
  ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball/ball")
  ball_geom_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "ball/ball_collision"
  )
  terrain_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")

  assert bindings.joint_qpos_adr.shape == (29,)
  assert bindings.actuator_ids.shape == (29,)
  assert model.camera(D435_CAMERA_NAME).id == bindings.d435_camera_id
  assert timestep == pytest.approx(0.005)
  assert decimation == 4
  assert model.geom_type[terrain_geom_id] == mujoco.mjtGeom.mjGEOM_PLANE
  assert model.geom_size[ball_geom_id, 0] == pytest.approx(0.1098)
  assert model.body_mass[ball_body_id] == pytest.approx(0.43)
  np.testing.assert_allclose(model.geom_friction[ball_geom_id], (0.1, 0.005, 0.001))
