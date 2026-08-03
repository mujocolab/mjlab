from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest

from mjlab.scripts.sim2sim.d435_ball_observer import (
  D435_CAMERA_NAME,
  D435Config,
  camera_point_to_yaw,
)
from mjlab.scripts.sim2sim.g1_football import (
  B1_HISTORY_OBS_DIM,
  BALL_DISTURBANCE_INTERVAL_RANGE,
  BALL_DISTURBANCE_LINEAR_VELOCITY_RANGE,
  EXPECTED_ACTION_DIM,
  EXPECTED_OBS_DIM,
  EXPECTED_OBSERVATION_NAMES,
  FRAME_STACK,
  PROPRIOCEPTIVE_OBS_DIM,
  PROPRIOCEPTIVE_OBSERVATION_NAMES,
  TEMPORAL_HISTORY_LENGTH,
  TEMPORAL_OBS_DIM,
  TEMPORAL_OBSERVATION_NAMES,
  TEMPORAL_TERM_DIMS,
  TERM_DIMS,
  BallRelativeCommandGenerator,
  BallVelocityDisturbance,
  KeyboardController,
  ModelBindings,
  ObservationAssembler,
  PolicyMetadata,
  Sim2SimCfg,
  StopSkillCommandGenerator,
  StopSkillCommandGeneratorCfg,
  build_model,
  compute_football_observation,
  configure_tracking_camera,
)


def make_terms(fill: float = 0.0) -> dict[str, np.ndarray]:
  return {name: np.full(dim, fill, dtype=np.float32) for name, dim in TERM_DIMS.items()}


def make_session(
  *, input_dim: int = EXPECTED_OBS_DIM, output_dim: int = EXPECTED_ACTION_DIM
) -> Any:
  metadata = {
    "joint_names": ",".join(f"joint_{index}" for index in range(29)),
    "default_joint_pos": ",".join("0" for _ in range(29)),
    "action_scale": ",".join("0.5" for _ in range(29)),
    "observation_names": ",".join(EXPECTED_OBSERVATION_NAMES),
    "observation_terms_history_length": ",".join("5" for _ in TERM_DIMS),
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
    PROPRIOCEPTIVE_OBSERVATION_NAMES
    if b1_stacked
    else TEMPORAL_OBSERVATION_NAMES
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
  cfg = D435Config(
    camera_pos_pelvis=(0.0, 0.0, 0.0),
    camera_quat_pelvis_wxyz=(1.0, 0.0, 0.0, 0.0),
  )

  point = camera_point_to_yaw(
    (0.2, 0.1, 1.0),
    (1.0, 0.0, 0.0, 0.0),
    cfg,
  )

  np.testing.assert_allclose(point, (0.2, -0.1, -1.0), atol=1e-6)


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


def test_numeric_keypad_uses_the_same_command_mapping() -> None:
  controller = KeyboardController(np.zeros(3, dtype=np.float32))

  controller(328)  # GLFW_KEY_KP_8
  controller(324)  # GLFW_KEY_KP_4
  controller(327)  # GLFW_KEY_KP_7

  np.testing.assert_allclose(controller.command, (0.1, 0.1, 0.1))


def test_ball_velocity_disturbance_is_disabled_by_default() -> None:
  assert not Sim2SimCfg().ball_velocity_disturbance
  assert not Sim2SimCfg().ball_relative_command_generator
  assert Sim2SimCfg().stop_skill.enabled


def test_ball_relative_command_generator_decelerates_and_returns_to_user() -> None:
  generator = BallRelativeCommandGenerator()
  moving = np.asarray([0.8, 0.0, 0.0], dtype=np.float32)
  stopped = np.zeros(3, dtype=np.float32)
  anchor = np.asarray([0.25, 0.0], dtype=np.float32)
  generator.reset(moving, anchor)

  first = generator.update(stopped, anchor, dt=0.1)
  np.testing.assert_allclose(first, (0.76, 0.0, 0.0), atol=1e-6)

  output = first
  for _ in range(24):
    output = generator.update(stopped, anchor, dt=0.1)
  np.testing.assert_allclose(output, stopped, atol=1e-6)


def test_ball_relative_command_generator_chases_ball_moving_forward() -> None:
  generator = BallRelativeCommandGenerator()
  moving = np.asarray([0.8, 0.0, 0.0], dtype=np.float32)
  stopped = np.zeros(3, dtype=np.float32)
  generator.reset(moving, np.asarray([0.25, 0.0], dtype=np.float32))

  output = generator.update(
    stopped,
    np.asarray([0.80, 0.0], dtype=np.float32),
    dt=0.1,
  )

  assert output[0] > moving[0]


def test_stop_skill_generator_rises_then_falls_to_keyboard_target() -> None:
  generator = StopSkillCommandGenerator(StopSkillCommandGeneratorCfg())
  moving = np.asarray([0.8, 0.0, 0.0], dtype=np.float32)
  stopped = np.zeros(3, dtype=np.float32)
  generator.reset(moving)

  first = generator.update(stopped, dt=0.02)
  np.testing.assert_allclose(first, moving)

  references = [generator.update(stopped, dt=0.02) for _ in range(60)]
  forward_references = np.asarray(references)[:, 0]

  assert np.max(forward_references) > moving[0]
  assert generator.state == StopSkillCommandGenerator.IDLE
  np.testing.assert_allclose(references[-1], stopped, atol=1e-6)
  np.testing.assert_allclose(generator.target_reference, stopped, atol=1e-6)


def test_stop_skill_generator_ignores_slow_keyboard_deceleration() -> None:
  generator = StopSkillCommandGenerator(StopSkillCommandGeneratorCfg())
  command = np.asarray([0.8, 0.0, 0.0], dtype=np.float32)
  generator.reset(command)

  for forward_command in np.linspace(0.79, 0.60, 20):
    command[0] = forward_command
    output = generator.update(command, dt=0.02)
    np.testing.assert_allclose(output, command)

  assert generator.state == StopSkillCommandGenerator.IDLE


def test_ball_velocity_disturbance_adds_xyz_velocity_without_angular_kick() -> None:
  model, _, _ = build_model()
  data = mujoco.MjData(model)
  ball_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "ball/ball_freejoint"
  )
  ball_dof_adr = int(model.jnt_dofadr[ball_joint_id])
  disturbance = BallVelocityDisturbance(ball_dof_adr, rng=np.random.default_rng(7))
  initial_velocity = np.asarray((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dtype=np.float64)
  data.qvel[ball_dof_adr : ball_dof_adr + 6] = initial_velocity
  disturbance.reset(float(data.time))

  assert BALL_DISTURBANCE_INTERVAL_RANGE[0] <= disturbance.next_trigger_time
  assert disturbance.next_trigger_time <= BALL_DISTURBANCE_INTERVAL_RANGE[1]
  data.time = disturbance.next_trigger_time
  delta = disturbance.update(data)

  assert delta is not None
  for value, bounds in zip(delta, BALL_DISTURBANCE_LINEAR_VELOCITY_RANGE, strict=True):
    assert bounds[0] <= value <= bounds[1]
  np.testing.assert_allclose(
    data.qvel[ball_dof_adr : ball_dof_adr + 3], initial_velocity[:3] + delta
  )
  np.testing.assert_array_equal(
    data.qvel[ball_dof_adr + 3 : ball_dof_adr + 6], initial_velocity[3:]
  )
  assert data.time + 5.0 <= disturbance.next_trigger_time <= data.time + 6.0


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
