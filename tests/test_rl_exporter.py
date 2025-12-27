"""Tests for RL exporter utilities."""

import os
import tempfile

import mujoco
import onnx
import pytest
from conftest import get_test_device

from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.manager_term_config import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
  list_to_csv_str,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg


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
      actuators=(XmlMotorActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),
      num_envs=1,
      extent=1.0,
      entities={"robot": robot_cfg},
    ),
    observations={
      "policy": ObservationGroupCfg(
        terms={
          "joint_pos": ObservationTermCfg(
            func=lambda env: env.scene["robot"].data.joint_pos
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

  env.close()
