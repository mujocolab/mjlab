"""Tests for XML actuator wrappers."""

import mujoco
import pytest
from conftest import get_test_device

from mjlab.actuator import XmlMotorActuatorCfg, XmlMuscleActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.manager_term_config import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg

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

# Robot with 2 tendons but only 1 muscle defined (underactuated).
ROBOT_XML_WITH_UNDERACTUATED_TENDONS = """
<mujoco>
    <worldbody>
        <body name="link1" pos="0 0 1">
            <geom name="link1_geom" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0" />
            <site name="site1" pos="0 0 0.1" size="0.02" type="sphere" />
            <site name="site2" pos="0 0 -0.1" size="0.02" type="sphere" />
            <geom name="wrapper_geom" type="cylinder" size="0.1 0.05" euler="1.57 0 0" pos="0.5 0 0" mass="0" />
            <site name="site3" pos="0.5 0 0.15" size="0.02" type="sphere" />
            <site name="site4" pos="0.5 0 -0.15" size="0.02" type="sphere" />
            <body name="link2" pos="0.5 0 0">
                <joint name="joint1" type="hinge" axis="0 1 0" range="-1.57 1.57" />
                <geom name="link2_geom" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="0.1" />
                <site name="site5" pos="0.5 0 0.1" size="0.02" type="sphere" />
                <site name="site6" pos="0.5 0 -0.1" size="0.02" type="sphere" />
            </body>
        </body>
    </worldbody>
    <tendon>
        <spatial name="tendon1">
            <site site="site1" />
            <geom geom="wrapper_geom" sidesite="site3" />
            <site site="site5" />
        </spatial>
        <spatial name="tendon2">
            <site site="site2" />
            <geom geom="wrapper_geom" sidesite="site4" />
            <site site="site6" />
        </spatial>
    </tendon>
    <actuator>
        <muscle name="muscle1" tendon="tendon2" />
    </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.mark.parametrize(
  "robot_xml_string, xml_actuator_cfg_cls, excepted_actuated_names",
  [
    (ROBOT_XML_UNDERACTUATED, XmlMotorActuatorCfg, ["joint2"]),
    (ROBOT_XML_WITH_UNDERACTUATED_TENDONS, XmlMuscleActuatorCfg, ["tendon2"]),
  ],
)
def test_xml_actuator_underactuated_with_wildcard(
  robot_xml_string: str,
  xml_actuator_cfg_cls: type[XmlMotorActuatorCfg] | type[XmlMuscleActuatorCfg],
  excepted_actuated_names: list[str],
):
  """XmlActuator filters to joints with XML actuators when using wildcard."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml_string),
    articulation=EntityArticulationInfoCfg(
      actuators=(xml_actuator_cfg_cls(actuated_names_expr=(".*",)),)
    ),
  )
  entity = Entity(cfg)
  entity.compile()

  # Should only control joint2 or tendon2 (which has an XML actuator), not joint1/tendon1.
  assert len(entity._actuators) == 1
  actuator = entity._actuators[0]
  assert actuator._actuated_names == excepted_actuated_names


@pytest.mark.parametrize(
  "robot_xml_string, xml_actuator_cfg_cls, actuated_names_expr",
  [
    (ROBOT_XML_UNDERACTUATED, XmlMotorActuatorCfg, ("joint1",)),
    (ROBOT_XML_WITH_UNDERACTUATED_TENDONS, XmlMuscleActuatorCfg, ("tendon1",)),
  ],
)
def test_xml_actuator_no_matching_actuators_raises_error(
  robot_xml_string: str,
  xml_actuator_cfg_cls: type[XmlMotorActuatorCfg] | type[XmlMuscleActuatorCfg],
  actuated_names_expr: tuple[str, ...],
):
  """XmlActuator raises error when no joints have matching XML actuators."""
  with pytest.raises(
    ValueError,
    match="No XML actuators found for any joints or any tendons matching the patterns",
  ):
    cfg = EntityCfg(
      spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml_string),
      articulation=EntityArticulationInfoCfg(
        actuators=(xml_actuator_cfg_cls(actuated_names_expr=actuated_names_expr),)
      ),
    )
    entity = Entity(cfg)
    entity.compile()


def test_joint_action_underactuated_with_wildcard(device):
  """JointAction with wildcard pattern matches only actuated joints."""
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML_UNDERACTUATED),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlMotorActuatorCfg(actuated_names_expr=(".*",)),)
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
      "joint_effort": mdp.JointEffortActionCfg(
        asset_name="robot", actuator_names=(".*",), scale=1.0
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
  assert action_term._joint_names == ["joint2"]
  assert action_term._joint_ids.tolist() == [1]

  env.close()
