"""Tests for XML actuator wrappers."""

import mujoco
import pytest
from conftest import get_test_device

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

# Robot with 2 joints but only 1 actuator defined (underactuated).
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


def test_xml_actuator_underactuated_with_wildcard():
  """XmlActuator filters to joints with XML actuators when using wildcard."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_UNDERACTUATED),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)
    ),
  )
  entity = Entity(cfg)
  entity.compile()

  # Should only control joint2 (which has an XML actuator), not joint1.
  assert len(entity._actuators) == 1
  actuator = entity._actuators[0]
  assert actuator._target_names == ["joint2"]


def test_xml_actuator_no_matching_actuators_raises_error():
  """XmlActuator raises error when no targets have matching XML actuators."""
  with pytest.raises(
    ValueError, match="No XML actuators found for any targets matching the patterns"
  ):
    cfg = EntityCfg(
      spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_UNDERACTUATED),
      articulation=EntityArticulationInfoCfg(
        actuators=(XmlActuatorCfg(target_names_expr=("joint1",)),)
      ),
    )
    entity = Entity(cfg)
    entity.compile()


def test_joint_action_underactuated_with_wildcard(device):
  """JointAction with wildcard pattern matches only actuated joints."""
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
            func=lambda env: env.scene["robot"].data.joint_pos
          ),
        },
      ),
    },
    actions={
      "joint_effort": mdp.JointEffortActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=1,
    episode_length_s=1.0,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  action_term = env.action_manager._terms["joint_effort"]
  assert isinstance(action_term, mdp.JointEffortAction)

  # Wildcard should resolve to only actuated joint (joint2), not all joints.
  assert action_term.action_dim == 1
  assert action_term.target_names == ["joint2"]
  assert action_term.target_ids.tolist() == [1]

  env.close()


# Robot with a "general" actuator (hand-tuned gainprm/biasprm that don't match
# any standard pattern). Strict auto-detection would reject this.
ROBOT_XML_GENERAL_ACTUATOR = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="custom_act" joint="joint1"
             gaintype="fixed" biastype="affine"
             gainprm="10" biasprm="5 -3 -1"/>
  </actuator>
</mujoco>
"""


def test_xml_actuator_explicit_command_field_bypasses_detection():
  """Explicit command_field allows non-standard XML actuators."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_GENERAL_ACTUATOR),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlActuatorCfg(
          target_names_expr=("joint1",),
          command_field="effort",
        ),
      )
    ),
  )
  entity = Entity(cfg)
  entity.compile()

  actuator = entity._actuators[0]
  assert actuator.command_field == "effort"
  assert actuator._target_names == ["joint1"]


def test_xml_actuator_auto_detection_rejects_general_actuator():
  """Auto-detection raises for non-standard actuators when command_field is None."""
  with pytest.raises(ValueError, match="Cannot determine command field"):
    cfg = EntityCfg(
      spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_GENERAL_ACTUATOR),
      articulation=EntityArticulationInfoCfg(
        actuators=(XmlActuatorCfg(target_names_expr=("joint1",)),)
      ),
    )
    entity = Entity(cfg)
    entity.compile()
