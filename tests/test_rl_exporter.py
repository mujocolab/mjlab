"""Tests for RL exporter utilities."""

import math
import os
import tempfile
from dataclasses import dataclass

import mujoco
import onnx
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import (
  BuiltinMotorActuatorCfg,
  BuiltinPdActuatorCfg,
  BuiltinPositionActuatorCfg,
  BuiltinVelocityActuatorCfg,
  IdealPdActuatorCfg,
  XmlActuatorCfg,
)
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
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
    attach_metadata_to_onnx(onnx_path, metadata)  # ty: ignore[invalid-argument-type]

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
    attach_metadata_to_onnx(onnx_path, metadata)  # ty: ignore[invalid-argument-type]

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
    {actuator}
  </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.mark.parametrize(
  "actuator_xml,expected_stiffness,expected_damping",
  [
    pytest.param('<motor joint="joint2" gear="1.0"/>', 1.0, 0.0, id="motor"),
    pytest.param('<position joint="joint2" kp="20" kv="2"/>', 20.0, 2.0, id="position"),
  ],
)
def test_get_base_metadata_skips_non_actuated_joints(
  device, actuator_xml, expected_stiffness, expected_damping
):
  """get_base_metadata handles non-actuated joints without KeyError."""
  xml = ROBOT_XML_UNDERACTUATED.format(actuator=actuator_xml)
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(xml),
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

  # Gains are only listed for actuated joints (joint2). XML actuators have no gains
  # in their config, so they are read from the MuJoCo gain and bias parameters.
  assert metadata["joint_stiffness"] == pytest.approx([expected_stiffness])
  assert metadata["joint_damping"] == pytest.approx([expected_damping])

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

  # Two joints with a flattened history of 5.
  assert metadata["observation_terms_dim"] == [10]

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
    spec_fn=lambda: mujoco.MjSpec.from_string(
      ROBOT_XML_UNDERACTUATED.format(actuator='<motor joint="joint2" gear="1.0"/>')
    ),
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


# Robot with joint, tendon and site transmissions. The spare tendon and site are
# never actuated, so they are not exported.
ROBOT_XML_MIXED_TRANSMISSIONS = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <site name="tip" pos="0 0 0.1"/>
      <site name="spare_site" pos="0 0 -0.1"/>
      <body name="link1">
        <joint name="j1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
      <body name="link2">
        <joint name="j2" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link2_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="ten">
      <joint joint="j1" coef="1"/>
      <joint joint="j2" coef="-1"/>
    </fixed>
    <fixed name="spare_ten">
      <joint joint="j1" coef="1"/>
    </fixed>
  </tendon>
</mujoco>
"""

JOINT_ACTUATORS = (
  BuiltinPositionActuatorCfg(target_names_expr=("j1",), stiffness=20, damping=2),
  BuiltinVelocityActuatorCfg(target_names_expr=("j2",), damping=1),
)
TENDON_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=("ten",),
    transmission_type=TransmissionType.TENDON,
    stiffness=30,
    damping=3,
  ),
)
SITE_ACTUATORS = (
  BuiltinMotorActuatorCfg(
    target_names_expr=("tip",),
    transmission_type=TransmissionType.SITE,
    effort_limit=5.0,
  ),
)
ALL_ACTUATORS = JOINT_ACTUATORS + TENDON_ACTUATORS + SITE_ACTUATORS

TRANSMISSION_KEYS = {
  "joint": {"joint_names", "joint_stiffness", "joint_damping", "default_joint_pos"},
  "tendon": {
    "tendon_names",
    "tendon_stiffness",
    "tendon_damping",
    "default_tendon_pos",
  },
  "site": {"site_names", "site_stiffness", "site_damping"},
}


def _make_mixed_transmission_env(
  device, actions: dict, actuators: tuple = ALL_ACTUATORS
) -> ManagerBasedRlEnv:
  # Nonzero default joint positions and j2 velocity make offsets taken from the
  # defaults distinguishable from zero.
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_MIXED_TRANSMISSIONS),
    init_state=EntityCfg.InitialStateCfg(
      joint_pos={"j1": 0.2, "j2": -0.1}, joint_vel={"j1": 0.0, "j2": 0.5}
    ),
    articulation=EntityArticulationInfoCfg(actuators=actuators),
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
            func=lambda env: env.scene["robot"].data.joint_pos
          ),
        },
      ),
    },
    actions=actions,
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=1,
    episode_length_s=1.0,
  )
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def _transmission_keys(metadata: dict) -> set[str]:
  return {k for keys in TRANSMISSION_KEYS.values() for k in keys if k in metadata}


@pytest.mark.parametrize(
  "actuators,actions,expected",
  [
    pytest.param(
      JOINT_ACTUATORS,
      {
        "joint_pos": mdp.JointPositionActionCfg(
          entity_name="robot", actuator_names=(".*",)
        )
      },
      {
        "joint_names": ["j1", "j2"],
        # Velocity actuators have damping but no stiffness.
        "joint_stiffness": [20.0, float("nan")],
        "joint_damping": [2.0, 1.0],
        "default_joint_pos": [0.2, -0.1],
      },
      id="joint",
    ),
    pytest.param(
      TENDON_ACTUATORS,
      {
        "tendon_len": mdp.TendonLengthActionCfg(
          entity_name="robot", actuator_names=("ten",), offset=0.3
        )
      },
      {
        "tendon_names": ["ten"],
        "tendon_stiffness": [30.0],
        "tendon_damping": [3.0],
        "default_tendon_pos": [0.3],
      },
      id="tendon_length",
    ),
    pytest.param(
      TENDON_ACTUATORS,
      {
        "tendon_vel": mdp.TendonVelocityActionCfg(
          entity_name="robot", actuator_names=("ten",), offset=0.3
        )
      },
      # Only a length action defines the tendon's default length.
      {
        "tendon_names": ["ten"],
        "tendon_stiffness": [30.0],
        "tendon_damping": [3.0],
        "default_tendon_pos": [float("nan")],
      },
      id="tendon_velocity",
    ),
    pytest.param(
      SITE_ACTUATORS,
      {
        "site_effort": mdp.SiteEffortActionCfg(
          entity_name="robot", actuator_names=("tip",)
        )
      },
      # Motors have no PD gains, and sites have no default position.
      {
        "site_names": ["tip"],
        "site_stiffness": [float("nan")],
        "site_damping": [float("nan")],
      },
      id="site",
    ),
    pytest.param(
      ALL_ACTUATORS,
      {
        "joint_pos": mdp.JointPositionActionCfg(
          entity_name="robot", actuator_names=(".*",)
        ),
        "tendon_len": mdp.TendonLengthActionCfg(
          entity_name="robot", actuator_names=("ten",), offset=0.3
        ),
        "site_effort": mdp.SiteEffortActionCfg(
          entity_name="robot", actuator_names=("tip",)
        ),
      },
      {
        "joint_names": ["j1", "j2"],
        "joint_stiffness": [20.0, float("nan")],
        "joint_damping": [2.0, 1.0],
        "default_joint_pos": [0.2, -0.1],
        "tendon_names": ["ten"],
        "tendon_stiffness": [30.0],
        "tendon_damping": [3.0],
        "default_tendon_pos": [0.3],
        "site_names": ["tip"],
        "site_stiffness": [float("nan")],
        "site_damping": [float("nan")],
      },
      id="mixed",
    ),
  ],
)
def test_get_base_metadata_transmission_models(device, actuators, actions, expected):
  """Only transmission types driven by an action term export their keys."""
  env = _make_mixed_transmission_env(device, actions, actuators)
  metadata = get_base_metadata(env, run_path="dummy/run")

  assert _transmission_keys(metadata) == set(expected)
  for key, value in expected.items():
    if key.endswith("_names"):
      assert metadata[key] == value
    else:
      assert metadata[key] == pytest.approx(value, nan_ok=True)

  env.close()


@pytest.mark.parametrize(
  "actuator,expected_stiffness,expected_damping",
  [
    pytest.param(
      BuiltinPositionActuatorCfg(
        target_names_expr=("j1", "j2"), stiffness=20, damping=2
      ),
      20.0,
      2.0,
      id="builtin_position",
    ),
    pytest.param(
      BuiltinPdActuatorCfg(target_names_expr=("j1", "j2"), stiffness=20, damping=2),
      20.0,
      2.0,
      id="builtin_pd",
    ),
    pytest.param(
      IdealPdActuatorCfg(target_names_expr=("j1", "j2"), stiffness=20, damping=2),
      20.0,
      2.0,
      id="ideal_pd",
    ),
    pytest.param(
      BuiltinVelocityActuatorCfg(target_names_expr=("j1", "j2"), damping=2),
      float("nan"),
      2.0,
      id="builtin_velocity",
    ),
    pytest.param(
      BuiltinMotorActuatorCfg(target_names_expr=("j1", "j2"), effort_limit=5.0),
      float("nan"),
      float("nan"),
      id="builtin_motor",
    ),
  ],
)
def test_get_base_metadata_actuator_gains(
  device, actuator, expected_stiffness, expected_damping
):
  """PD gains come from the actuator config, NaN when the config has none."""
  env = _make_mixed_transmission_env(
    device,
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",)
      )
    },
    actuators=(actuator,),
  )
  metadata = get_base_metadata(env, run_path="dummy/run")

  assert metadata["joint_stiffness"] == pytest.approx(
    [expected_stiffness] * 2, nan_ok=True
  )
  assert metadata["joint_damping"] == pytest.approx([expected_damping] * 2, nan_ok=True)

  env.close()


@pytest.mark.parametrize(
  "action,target,expected_offset",
  [
    pytest.param(
      mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=("j2",), scale=2.0
      ),
      "j2",
      -0.1,  # Default joint position.
      id="joint_position",
    ),
    pytest.param(
      mdp.JointVelocityActionCfg(
        entity_name="robot", actuator_names=("j2",), scale=2.0
      ),
      "j2",
      0.5,  # Default joint velocity.
      id="joint_velocity",
    ),
    pytest.param(
      mdp.RelativeJointPositionActionCfg(
        entity_name="robot", actuator_names=("j2",), scale=2.0
      ),
      "j2",
      0.0,
      id="relative_joint_position",
    ),
    pytest.param(
      mdp.TendonEffortActionCfg(
        entity_name="robot", actuator_names=("ten",), scale=2.0, offset=0.1
      ),
      "ten",
      0.1,
      id="explicit_offset",
    ),
  ],
)
def test_get_base_metadata_action_scale_and_offset(
  device, action, target, expected_offset
):
  """Scale and offset are read from the action term, including its defaults."""
  env = _make_mixed_transmission_env(device, {"action": action})
  metadata = get_base_metadata(env, run_path="dummy/run")

  assert metadata["action_target_names"] == [target]
  assert metadata["action_scale"] == pytest.approx([2.0])
  assert metadata["action_offset"] == pytest.approx([expected_offset])

  env.close()


def test_get_base_metadata_action_order(device):
  """Action metadata follows action term order, not transmission order."""
  env = _make_mixed_transmission_env(
    device,
    actions={
      "site_effort": mdp.SiteEffortActionCfg(
        entity_name="robot", actuator_names=("tip",), scale=3.0, offset=0.3
      ),
      "tendon_len": mdp.TendonLengthActionCfg(
        entity_name="robot", actuator_names=("ten",), scale=2.0, offset=0.2
      ),
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale={"j1": 0.5, "j2": 0.25}
      ),
    },
  )
  metadata = get_base_metadata(env, run_path="dummy/run")

  assert metadata["action_names"] == ["site_effort", "tendon_len", "joint_pos"]
  assert metadata["action_target_names"] == ["tip", "ten", "j1", "j2"]
  assert metadata["action_terms_dim"] == [1, 1, 2]
  assert metadata["action_scale"] == pytest.approx([3.0, 2.0, 0.5, 0.25])
  assert metadata["action_offset"] == pytest.approx([0.3, 0.2, 0.2, -0.1])

  env.close()


@pytest.mark.parametrize(
  "tendon_value,expected", [(0.5, [0.5] * 3), (0.7, [0.5, 0.5, 0.7])]
)
def test_get_base_metadata_scale_and_offset_are_lists(device, tendon_value, expected):
  """Scale and offset are per-dim lists, even when all dims share one value."""
  env = _make_mixed_transmission_env(
    device,
    actions={
      # Dict scale and offset are stored as tensors, float ones as floats.
      "joint_effort": mdp.JointEffortActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale={".*": 0.5},
        offset={".*": 0.5},
      ),
      "tendon_effort": mdp.TendonEffortActionCfg(
        entity_name="robot",
        actuator_names=("ten",),
        scale=tendon_value,
        offset=tendon_value,
      ),
    },
  )
  metadata = get_base_metadata(env, run_path="dummy/run")

  for key in ("action_scale", "action_offset"):
    assert isinstance(metadata[key], list)
    assert metadata[key] == pytest.approx(expected)

  env.close()


@dataclass(kw_only=True)
class _NonBaseActionCfg(ActionTermCfg):
  def build(self, env: ManagerBasedRlEnv) -> "_NonBaseAction":
    return _NonBaseAction(self, env)


class _NonBaseAction(ActionTerm):
  """Action term that does not derive from BaseAction."""

  def __init__(self, cfg: _NonBaseActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)

  @property
  def action_dim(self) -> int:
    return 2

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions

  def apply_actions(self) -> None:
    pass


def test_get_base_metadata_non_base_action_padded_with_nan(device):
  """A non-BaseAction term keeps its names, with NaN scale and offset."""
  env = _make_mixed_transmission_env(
    device,
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=("j1",), scale=0.5
      ),
      "custom": _NonBaseActionCfg(entity_name="robot"),
    },
  )
  with pytest.warns(UserWarning, match="Action term 'custom'"):
    metadata = get_base_metadata(env, run_path="dummy/run")

  assert metadata["action_names"] == ["joint_pos", "custom"]
  assert metadata["action_target_names"] == ["j1", "custom_0", "custom_1"]
  assert metadata["action_terms_dim"] == [1, 2]
  for key in ("action_scale", "action_offset"):
    values = metadata[key]
    assert isinstance(values, list)
    assert not math.isnan(values[0])
    assert all(math.isnan(v) for v in values[1:])
  # The rest of the export is unaffected.
  assert _transmission_keys(metadata) == TRANSMISSION_KEYS["joint"]

  env.close()


@pytest.mark.parametrize("num_terms", [1, 2])
def test_get_base_metadata_only_non_base_actions(device, num_terms):
  """With only non-BaseAction terms, scale and offset are lists of NaN."""
  env = _make_mixed_transmission_env(
    device,
    actions={
      f"custom{i}": _NonBaseActionCfg(entity_name="robot") for i in range(num_terms)
    },
  )
  with pytest.warns(UserWarning, match="is not a BaseAction"):
    metadata = get_base_metadata(env, run_path="dummy/run")

  assert metadata["action_terms_dim"] == [2] * num_terms
  for key in ("action_scale", "action_offset"):
    values = metadata[key]
    assert isinstance(values, list)
    assert len(values) == 2 * num_terms
    assert all(math.isnan(v) for v in values)
  # No term drives a transmission, so no transmission fields are exported.
  assert _transmission_keys(metadata) == set()

  env.close()
