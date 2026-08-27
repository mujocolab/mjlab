from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mjlab.scripts.sim2sim.g1_football import EXPECTED_ACTION_DIM, KeyboardController
from mjlab.scripts.sim2sim.g1_football_depth import (
  DEPLOYED_COMMAND_MAX,
  DEPLOYED_COMMAND_MIN,
  DepthPolicyMetadata,
  Sim2SimCfg,
  TrainingActionProcessor,
)


def make_metadata(*, default: float = 0.0, scale: float = 1.0) -> DepthPolicyMetadata:
  return DepthPolicyMetadata(
    joint_names=tuple(f"joint_{index}" for index in range(EXPECTED_ACTION_DIM)),
    default_joint_pos=np.full(EXPECTED_ACTION_DIM, default, dtype=np.float32),
    action_scale=np.full(EXPECTED_ACTION_DIM, scale, dtype=np.float32),
    depth_history_length=10,
    depth_height=30,
    depth_width=40,
  )


def make_session() -> Any:
  metadata = {
    "joint_names": ",".join(f"joint_{index}" for index in range(29)),
    "default_joint_pos": ",".join("0" for _ in range(29)),
    "action_scale": ",".join("0.5" for _ in range(29)),
    "observation_names": ",".join(
      (
        "base_ang_vel",
        "projected_gravity",
        "command",
        "phase",
        "joint_pos",
        "joint_vel",
        "actions",
      )
    ),
    "observation_terms_history_length": ",".join("5" for _ in range(7)),
    "depth_history_length": "10",
    "depth_height": "30",
    "depth_width": "40",
  }
  return SimpleNamespace(
    get_inputs=lambda: [
      SimpleNamespace(name="proprio", shape=[1, 490]),
      SimpleNamespace(name="depth", shape=[1, 10, 30, 40]),
    ],
    get_outputs=lambda: [SimpleNamespace(name="actions", shape=[1, 29])],
    get_modelmeta=lambda: SimpleNamespace(custom_metadata_map=metadata),
  )


def test_depth_metadata_accepts_deployment_two_input_contract() -> None:
  metadata = DepthPolicyMetadata.from_session(make_session())

  assert metadata.depth_history_length == 10
  assert (metadata.depth_height, metadata.depth_width) == (30, 40)


def test_training_action_processor_applies_no_clamps() -> None:
  metadata = make_metadata(default=0.5, scale=0.25)
  processor = TrainingActionProcessor(metadata)
  raw_action = np.full(29, 20.0, dtype=np.float32)

  np.testing.assert_allclose(processor.reset(), 0.5)
  np.testing.assert_allclose(
    processor.process(raw_action),
    metadata.default_joint_pos + metadata.action_scale * raw_action,
  )


def test_depth_sim2sim_uses_deployment_command_envelope() -> None:
  Sim2SimCfg(command_x=1.0, command_y=-0.25, command_yaw=1.0)

  with pytest.raises(ValueError, match="deployment range"):
    Sim2SimCfg(command_x=1.01)
  with pytest.raises(ValueError, match="deployment range"):
    Sim2SimCfg(command_y=-0.26)
  with pytest.raises(ValueError, match="camera_position_jitter_meters"):
    Sim2SimCfg(camera_position_jitter_meters=-0.001)


def test_keyboard_controller_can_use_deployment_command_envelope() -> None:
  keyboard = KeyboardController(
    np.asarray([1.0, 0.25, 1.0], dtype=np.float32),
    command_min=DEPLOYED_COMMAND_MIN,
    command_max=DEPLOYED_COMMAND_MAX,
  )

  keyboard(ord("8"))
  keyboard(ord("4"))
  keyboard(ord("7"))

  np.testing.assert_array_equal(keyboard.command, DEPLOYED_COMMAND_MAX)
