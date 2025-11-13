"""Tests for spec_config.py."""

import mujoco
import pytest

from mjlab.utils.spec_config import (
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
    geom_names_expr=("arm_collision",), contype=2, conaffinity=3, condim=4, priority=1
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
    geom_names_expr=(r"^(left|right)_foot\d_collision$",),
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
    geom_names_expr=(r".*_foot\d_collision$", "arm_collision"),
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
    geom_names_expr=("left_foot1_collision",), contype=2, disable_other_geoms=True
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
    geom_names_expr = ("test",)
    contype = value if param == "contype" else 1
    conaffinity = value if param == "conaffinity" else 1
    condim = value if param == "condim" else 3
    priority = value if param == "priority" else 0

    cfg = CollisionCfg(
      geom_names_expr=geom_names_expr,
      contype=contype,
      conaffinity=conaffinity,
      condim=condim,
      priority=priority,
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
