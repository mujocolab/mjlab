"""Tests for spec_config.py.

Tests the configuration system for building MuJoCo models, including:
- Actuator creation and parameter validation (PD control, effort limits)
- Collision geom property modification (friction, contact dimensions, regex matching)
- Sensor addition (inertial sensors), validation
- Visual elements (textures, materials, lights, cameras)
"""

import mujoco
import pytest

from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import (
  ActuatorCfg,
  CameraCfg,
  CollisionCfg,
  LightCfg,
  MaterialCfg,
  TextureCfg,
)


@pytest.fixture
def simple_robot_xml():
  """Minimal robot XML for testing."""
  return """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
          <body name="link1" pos="0 0 0">
            <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
          </body>
          <body name="link2" pos="0 0 0">
            <joint name="joint2" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="link2_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """


@pytest.fixture
def robot_with_actuators_xml():
  """Robot XML with pre-existing actuators."""
  return """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
          <body name="link1" pos="0 0 0">
            <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position name="joint1" joint="joint1" kp="10.0" kv="1.0"/>
      </actuator>
    </mujoco>
    """


# Actuator Tests


def test_existing_actuators_parsed(robot_with_actuators_xml):
  """Existing actuators in XML should be parsed correctly."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_actuators_xml),
  )
  entity = Entity(cfg)

  assert entity.num_actuators == 1
  assert entity.actuator_names == ["joint1"]
  assert entity.is_actuated

  act = entity.spec.actuator("joint1")
  assert act.gainprm[0] == 10.0
  assert act.biasprm[2] == -1.0


def test_actuator_cfg_creates_actuators(simple_robot_xml):
  """ActuatorCfg should create actuators for matched joints."""
  actuator_cfg = ActuatorCfg(
    joint_names_expr=["joint1", "joint2"],
    effort_limit=2.0,
    stiffness=15.0,
    damping=2.0,
    frictionloss=0.1,
    armature=0.01,
  )

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(simple_robot_xml),
    articulation=EntityArticulationInfoCfg(actuators=(actuator_cfg,)),
  )
  entity = Entity(cfg)

  assert entity.num_actuators == 2
  assert set(entity.actuator_names) == {"joint1", "joint2"}

  act = entity.spec.actuator("joint1")
  assert act.gainprm[0] == 15.0
  assert act.biasprm[1] == -15.0
  assert act.biasprm[2] == -2.0
  assert act.forcerange[0] == -2.0
  assert act.forcerange[1] == 2.0

  joint = entity.spec.joint("joint1")
  assert joint.frictionloss == 0.1
  assert joint.armature == 0.01


def test_actuator_cfg_regex_matching(simple_robot_xml):
  """ActuatorCfg should support regex pattern matching."""
  actuator_cfg = ActuatorCfg(
    joint_names_expr=["joint.*"],
    effort_limit=1.5,
    stiffness=8.0,
    damping=1.5,
  )

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(simple_robot_xml),
    articulation=EntityArticulationInfoCfg(actuators=(actuator_cfg,)),
  )
  entity = Entity(cfg)

  assert entity.num_actuators == 2
  assert set(entity.actuator_names) == {"joint1", "joint2"}


def test_multiple_actuator_configs(simple_robot_xml):
  """Multiple ActuatorCfg instances should apply different parameters."""
  actuator_cfgs = (
    ActuatorCfg(
      joint_names_expr=["joint1"], effort_limit=3.0, stiffness=20.0, damping=3.0
    ),
    ActuatorCfg(
      joint_names_expr=["joint2"], effort_limit=1.0, stiffness=10.0, damping=1.0
    ),
  )

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(simple_robot_xml),
    articulation=EntityArticulationInfoCfg(actuators=actuator_cfgs),
  )
  entity = Entity(cfg)

  joint1_act = entity.spec.actuator("joint1")
  assert joint1_act.gainprm[0] == 20.0
  assert joint1_act.forcerange[1] == 3.0

  joint2_act = entity.spec.actuator("joint2")
  assert joint2_act.gainprm[0] == 10.0
  assert joint2_act.forcerange[1] == 1.0


def test_no_matching_joints_raises_error(simple_robot_xml):
  """ActuatorCfg should raise error when no joints match."""
  actuator_cfg = ActuatorCfg(
    joint_names_expr=["nonexistent.*"],
    effort_limit=1.0,
    stiffness=1.0,
    damping=1.0,
  )

  with pytest.raises(
    ValueError,
    match="No joints matched actuator patterns 'nonexistent.*'",
  ):
    cfg = EntityCfg(
      spec_fn=lambda: mujoco.MjSpec.from_string(simple_robot_xml),
      articulation=EntityArticulationInfoCfg(actuators=(actuator_cfg,)),
    )
    Entity(cfg)


# fmt: off
@pytest.mark.parametrize(
  "param,value,expected_error",
  [
    ("effort_limit", -1.0, "effort_limit must be positive"),
    ("stiffness", -1.0, "stiffness must be non-negative"),
    ("damping", -1.0, "damping must be non-negative"),
    ("frictionloss", -0.1, "frictionloss must be non-negative"),
    ("armature", -0.01, "armature must be non-negative"),
  ],
)
# fmt: on
def test_actuator_validation(simple_robot_xml, param, value, expected_error):
  """ActuatorCfg should validate parameters."""
  with pytest.raises(ValueError, match=expected_error):
    params = {
      "joint_names_expr": ["joint1"],
      "effort_limit": 1.0,
      "stiffness": 1.0,
      "damping": 1.0,
      "frictionloss": 0.0,
      "armature": 0.0,
    }
    params[param] = value

    actuator_cfg = ActuatorCfg(
      joint_names_expr=params["joint_names_expr"],
      effort_limit=params["effort_limit"],
      stiffness=params["stiffness"],
      damping=params["damping"],
      frictionloss=params["frictionloss"],
      armature=params["armature"],
    )
    cfg = EntityCfg(
      spec_fn=lambda: mujoco.MjSpec.from_string(simple_robot_xml),
      articulation=EntityArticulationInfoCfg(actuators=(actuator_cfg,)),
    )
    Entity(cfg)


def test_actuator_ordering_preserved(simple_robot_xml):
  """Actuators should be created in spec joint order, not config order."""
  # Match joint2 first in config, then joint1.
  actuator_cfgs = (
    ActuatorCfg(
      joint_names_expr=["joint2"], effort_limit=1.0, stiffness=1.0, damping=1.0
    ),
    ActuatorCfg(
      joint_names_expr=["joint1"], effort_limit=2.0, stiffness=2.0, damping=2.0
    ),
  )

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(simple_robot_xml),
    articulation=EntityArticulationInfoCfg(actuators=actuator_cfgs),
  )
  entity = Entity(cfg)

  # Actuators should be in spec order (joint1, joint2), not config order
  assert entity.actuator_names == ["joint1", "joint2"]


# Collision Tests


@pytest.fixture
def multi_geom_spec():
  """Spec with multiple geoms for collision testing."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="test_body")
  body.add_geom(
    name="left_foot1_collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1]
  )
  body.add_geom(
    name="right_foot3_collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1]
  )
  body.add_geom(
    name="arm_collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1]
  )
  return spec


def test_collision_basic_properties(multi_geom_spec):
  """CollisionCfg should set basic collision properties."""
  collision_cfg = CollisionCfg(
    geom_names_expr=["arm_collision"], contype=2, conaffinity=3, condim=4, priority=1
  )
  collision_cfg.edit_spec(multi_geom_spec)

  geom = multi_geom_spec.geom("arm_collision")
  assert geom.contype == 2
  assert geom.conaffinity == 3
  assert geom.condim == 4
  assert geom.priority == 1


def test_collision_regex_matching(multi_geom_spec):
  """CollisionCfg should support regex pattern matching."""
  collision_cfg = CollisionCfg(
    geom_names_expr=[r"^(left|right)_foot\d_collision$"],
    condim=3,
    priority=1,
    friction=(0.6,),
    disable_other_geoms=False,
  )
  collision_cfg.edit_spec(multi_geom_spec)

  left_foot = multi_geom_spec.geom("left_foot1_collision")
  assert left_foot.condim == 3
  assert left_foot.priority == 1
  assert left_foot.friction[0] == 0.6

  right_foot = multi_geom_spec.geom("right_foot3_collision")
  assert right_foot.condim == 3

  arm = multi_geom_spec.geom("arm_collision")
  assert arm.condim == 3  # Default unchanged.


def test_collision_dict_field_resolution(multi_geom_spec):
  """CollisionCfg should support dict-based field resolution."""
  collision_cfg = CollisionCfg(
    geom_names_expr=[r".*_foot\d_collision$", "arm_collision"],
    condim={r".*_foot\d_collision$": 3, "arm_collision": 1},
    priority={r".*_foot\d_collision$": 2, "arm_collision": 0},
  )
  collision_cfg.edit_spec(multi_geom_spec)

  left_foot = multi_geom_spec.geom("left_foot1_collision")
  assert left_foot.condim == 3
  assert left_foot.priority == 2

  arm = multi_geom_spec.geom("arm_collision")
  assert arm.condim == 1
  assert arm.priority == 0


def test_collision_disable_other_geoms(multi_geom_spec):
  """CollisionCfg should disable non-matching geoms when requested."""
  collision_cfg = CollisionCfg(
    geom_names_expr=["left_foot1_collision"], contype=2, disable_other_geoms=True
  )
  collision_cfg.edit_spec(multi_geom_spec)

  left_foot = multi_geom_spec.geom("left_foot1_collision")
  assert left_foot.contype == 2

  right_foot = multi_geom_spec.geom("right_foot3_collision")
  assert right_foot.contype == 0
  assert right_foot.conaffinity == 0

  arm = multi_geom_spec.geom("arm_collision")
  assert arm.contype == 0


# fmt: off
@pytest.mark.parametrize(
  "param,value,expected_error",
  [
    ("condim", -1, "condim must be one of"),
    ("condim", 2, "condim must be one of"),
    ("contype", -1, "contype must be non-negative"),
    ("conaffinity", -1, "conaffinity must be non-negative"),
    ("priority", -1, "priority must be non-negative"),
  ],
)
# fmt: on
def test_collision_validation(param, value, expected_error):
  """CollisionCfg should validate parameters."""
  with pytest.raises(ValueError, match=expected_error):
    params: dict = {"geom_names_expr": ["test"]}
    params[param] = value
    cfg = CollisionCfg(
      geom_names_expr=params["geom_names_expr"],
      contype=params.get("contype", 1),
      conaffinity=params.get("conaffinity", 1),
      condim=params.get("condim", 3),
      priority=params.get("priority", 0),
    )
    cfg.validate()


# Visual Element Tests


def test_texture_cfg():
  """TextureCfg should add textures to spec."""
  spec = mujoco.MjSpec()
  texture_cfg = TextureCfg(
    name="test_texture",
    type="2d",
    builtin="checker",
    rgb1=(1.0, 0.0, 0.0),
    rgb2=(0.0, 1.0, 0.0),
    width=64,
    height=64,
  )
  texture_cfg.edit_spec(spec)

  texture = spec.texture("test_texture")
  assert texture.name == "test_texture"


def test_material_cfg():
  """MaterialCfg should add materials to spec."""
  spec = mujoco.MjSpec()
  material_cfg = MaterialCfg(
    name="test_material",
    texuniform=True,
    texrepeat=(2, 2),
    reflectance=0.5,
  )
  material_cfg.edit_spec(spec)

  material = spec.material("test_material")
  assert material.name == "test_material"


def test_light_cfg():
  """LightCfg should add lights to spec."""
  spec = mujoco.MjSpec()
  light_cfg = LightCfg(
    name="test_light",
    body="world",
    type="spot",
    pos=(1.0, 2.0, 3.0),
    dir=(0.0, 0.0, -1.0),
  )
  light_cfg.edit_spec(spec)

  light = spec.light("test_light")
  assert light.name == "test_light"


def test_camera_cfg():
  """CameraCfg should add cameras to spec."""
  spec = mujoco.MjSpec()
  camera_cfg = CameraCfg(
    name="test_camera", body="world", fovy=60.0, pos=(0.0, 0.0, 5.0)
  )
  camera_cfg.edit_spec(spec)

  camera = spec.camera("test_camera")
  assert camera.name == "test_camera"
