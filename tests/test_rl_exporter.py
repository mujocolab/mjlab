"""Tests for RL exporter utilities."""

import os
import tempfile

import mujoco
import onnx
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
  list_to_csv_str,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg


def test_list_to_csv_str():
  """Test CSV string conversion utility."""
  # Test with floats.
  result = list_to_csv_str([1.23456, 2.34567, 3.45678], decimals=3)
  assert result == "1.235,2.346,3.457"

  # Test with integers.
  result = list_to_csv_str([1, 2, 3], decimals=2)
  assert result == "1.00,2.00,3.00"

  # Test with mixed types.
  result = list_to_csv_str([1.5, "hello", 2.5], decimals=1)
  assert result == "1.5,hello,2.5"

  # Test custom delimiter.
  result = list_to_csv_str([1.0, 2.0, 3.0], decimals=1, delimiter=";")
  assert result == "1.0;2.0;3.0"

  # Test nested sequence entries (e.g. per-term clip ranges or vector scales):
  # each entry is joined with the sub-delimiter so the outer delimiter stays
  # unambiguous and round-trips via a plain split.
  result = list_to_csv_str([[1.0, 2.0], 3.0, [4.0, 5.0]], decimals=1)
  assert result == "1.0;2.0,3.0,4.0;5.0"
  entries = result.split(",")
  assert entries == ["1.0;2.0", "3.0", "4.0;5.0"]
  assert entries[0].split(";") == ["1.0", "2.0"]


def test_attach_metadata_to_onnx():
  """Test that metadata can be attached to ONNX models."""
  # Create a dummy ONNX model.
  with tempfile.TemporaryDirectory() as tmpdir:
    onnx_path = os.path.join(tmpdir, "test_policy.onnx")

    # Create minimal ONNX model.
    input_tensor = onnx.helper.make_tensor_value_info(
      "input", onnx.TensorProto.FLOAT, [1, 2]
    )
    output_tensor = onnx.helper.make_tensor_value_info(
      "output", onnx.TensorProto.FLOAT, [1, 2]
    )
    node = onnx.helper.make_node("Identity", ["input"], ["output"])
    graph = onnx.helper.make_graph(
      [node], "test_graph", [input_tensor], [output_tensor]
    )
    model = onnx.helper.make_model(graph)
    onnx.save(model, onnx_path)

    # Attach metadata.
    metadata = {
      "run_path": "test/run/path",
      "joint_names": ["joint_a", "joint_b"],
      "joint_stiffness": [20.0, 10.0],
      "joint_damping": [1.0, 1.0],
      "extra_field": "extra_value",
    }
    attach_metadata_to_onnx(onnx_path, metadata)

    # Load and verify metadata was attached.
    loaded_model = onnx.load(onnx_path)
    metadata_props = {prop.key: prop.value for prop in loaded_model.metadata_props}

    # Check all metadata fields are present.
    assert "run_path" in metadata_props
    assert "joint_names" in metadata_props
    assert "joint_stiffness" in metadata_props
    assert "extra_field" in metadata_props

    # Check values are correct.
    assert metadata_props["run_path"] == "test/run/path"
    assert metadata_props["extra_field"] == "extra_value"

    # Check list was converted to CSV string.
    joint_names = metadata_props["joint_names"].split(",")
    assert len(joint_names) == 2
    assert "joint_a" in joint_names
    assert "joint_b" in joint_names

    # Check stiffness values are in natural joint order.
    stiffness_values = [float(x) for x in metadata_props["joint_stiffness"].split(",")]
    assert stiffness_values == [20.0, 10.0]  # Natural order: joint_a (20), joint_b (10)


def test_attach_metadata_to_onnx_nested_clip_and_scale():
  """Per-term clip ranges and vector scales must round-trip through a plain
  comma split without corrupting neighboring entries."""
  with tempfile.TemporaryDirectory() as tmpdir:
    onnx_path = os.path.join(tmpdir, "test_policy.onnx")

    input_tensor = onnx.helper.make_tensor_value_info(
      "input", onnx.TensorProto.FLOAT, [1, 2]
    )
    output_tensor = onnx.helper.make_tensor_value_info(
      "output", onnx.TensorProto.FLOAT, [1, 2]
    )
    node = onnx.helper.make_node("Identity", ["input"], ["output"])
    graph = onnx.helper.make_graph(
      [node], "test_graph", [input_tensor], [output_tensor]
    )
    model = onnx.helper.make_model(graph)
    onnx.save(model, onnx_path)

    metadata = {
      "observation_terms_clip": [
        [float("-inf"), float("inf")],
        [-1.0, 1.0],
        [float("-inf"), float("inf")],
      ],
      "observation_terms_scale": [1.0, [0.5, 1.2, 0.3], 2.0],
    }
    attach_metadata_to_onnx(onnx_path, metadata)

    loaded_model = onnx.load(onnx_path)
    metadata_props = {prop.key: prop.value for prop in loaded_model.metadata_props}

    clip_entries = metadata_props["observation_terms_clip"].split(",")
    assert clip_entries == ["-inf;inf", "-1.000;1.000", "-inf;inf"]
    assert [float(x) for x in clip_entries[1].split(";")] == [-1.0, 1.0]

    scale_entries = metadata_props["observation_terms_scale"].split(",")
    assert scale_entries == ["1.000", "0.500;1.200;0.300", "2.000"]
    assert [float(x) for x in scale_entries[1].split(";")] == [0.5, 1.2, 0.3]


# Robot with 2 joints but only 1 actuator (underactuated).
ROBOT_XML_UNDERACTUATED = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
      <body name="link2" pos="0 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link2_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="actuator1" joint="joint2" gear="1.0"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def test_get_base_metadata_skips_non_actuated_joints(device):
  """get_base_metadata handles non-actuated joints without KeyError."""
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_UNDERACTUATED),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=1.0,
      entities={"robot": robot_cfg},
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "joint_pos": ObservationTermCfg(
            func=lambda env: env.scene["robot"].data.joint_pos,
            history_length=5,
            scale=2.0,
          ),
        },
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=1,
    episode_length_s=1.0,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  metadata = get_base_metadata(env, run_path="dummy/run")

  robot = env.scene["robot"]

  # All joints (including non-actuated) should be listed in joint_names metadata.
  joint_names_meta = metadata["joint_names"]
  assert isinstance(joint_names_meta, list)
  assert joint_names_meta == list(robot.joint_names)
  assert "joint1" in joint_names_meta
  assert "joint2" in joint_names_meta

  # Stiffness/damping are only defined for actuated joints, in natural joint order.
  stiffness_meta = metadata["joint_stiffness"]
  damping_meta = metadata["joint_damping"]
  assert isinstance(stiffness_meta, list)
  assert isinstance(damping_meta, list)
  assert len(stiffness_meta) == len(robot.spec.actuators)
  assert len(damping_meta) == len(robot.spec.actuators)

  observation_names = metadata["observation_names"]
  assert isinstance(observation_names, list)
  assert "joint_pos" in observation_names

  observation_terms_scale = metadata["observation_terms_scale"]
  assert isinstance(observation_terms_scale, list)
  assert len(observation_terms_scale) == len(observation_names)
  assert observation_terms_scale[0] == 2.0

  observation_history_length = metadata["observation_terms_history_length"]
  assert isinstance(observation_history_length, list)
  assert len(observation_history_length) == len(observation_names)
  assert observation_history_length[0] == 5

  observation_flatten_history_dim = metadata["observation_terms_flatten_history_dim"]
  assert isinstance(observation_flatten_history_dim, list)
  assert len(observation_flatten_history_dim) == len(observation_names)
  # Default flatten_history_dim is 1.0 (True)
  assert observation_flatten_history_dim[0] is True

  observation_terms_clip = metadata["observation_terms_clip"]
  assert isinstance(observation_terms_clip, list)
  assert len(observation_terms_clip) == len(observation_names)
  # Default clip is [-inf, inf]
  assert observation_terms_clip[0] == [float("-inf"), float("inf")]

  env.close()


def test_get_base_metadata_multiple_terms_scale_clip(device):
  """get_base_metadata preserves obs term ordering and handles scalar/vector tensor scale, tuple scale, and non-default clip."""
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_UNDERACTUATED),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=1.0,
      entities={"robot": robot_cfg},
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "joint_pos": ObservationTermCfg(
            func=lambda env: env.scene["robot"].data.joint_pos,
            scale=torch.tensor(
              2.0
            ),  # tensor scale -> should be converted via .tolist()
            history_length=1,
          ),
          "joint_vel": ObservationTermCfg(
            func=lambda env: env.scene["robot"].data.joint_vel,
            scale=(0.5, 1.5),  # tuple scale -> stored as-is
            clip=(-1.0, 1.0),  # non-default clip
            history_length=1,
          ),
          "joint_pos_scaled": ObservationTermCfg(
            func=lambda env: env.scene["robot"].data.joint_pos,
            scale=torch.tensor([1.0, 2.0]),  # per-element tensor -> list of floats
            history_length=1,
          ),
        },
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=1,
    episode_length_s=1.0,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  metadata = get_base_metadata(env, run_path="dummy/run")

  observation_names = metadata["observation_names"]
  scales = metadata["observation_terms_scale"]
  clips = metadata["observation_terms_clip"]
  assert isinstance(observation_names, list)
  assert isinstance(scales, list)
  assert isinstance(clips, list)

  # Terms appear in definition order.
  assert observation_names == ["joint_pos", "joint_vel", "joint_pos_scaled"]

  # Scalar tensor scale is unwrapped to a plain float.
  assert scales[0] == 2.0

  # Tuple scale is stored as list.
  assert scales[1] == [0.5, 1.5]

  # Per-element tensor scale is converted to a list of floats.
  assert scales[2] == [1.0, 2.0]

  # joint_pos has default clip.
  assert clips[0] == [float("-inf"), float("inf")]

  # joint_vel has explicit clip converted to list.
  assert clips[1] == [-1.0, 1.0]

  # joint_pos_scaled has default clip.
  assert clips[2] == [float("-inf"), float("inf")]

  env.close()
