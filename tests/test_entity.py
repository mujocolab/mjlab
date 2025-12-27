"""Tests for entity module."""

from dataclasses import dataclass

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device, load_fixture_xml

from mjlab.actuator import BuiltinPositionActuatorCfg, XmlMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg

FIXED_BASE_XML = """
<mujoco>
  <worldbody>
    <body name="object" pos="0 0 0.5">
      <geom name="object_geom" type="box" size="0.1 0.1 0.1" rgba="0.8 0.3 0.3 1"/>
    </body>
  </worldbody>
</mujoco>
"""

FLOATING_BASE_XML = """
<mujoco>
  <worldbody>
    <body name="object" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="object_geom" type="box" size="0.1 0.1 0.1" rgba="0.3 0.3 0.8 1" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

FIXED_BASE_ARTICULATED_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="base_geom" type="cylinder" size="0.1 0.05" mass="5.0"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="link1_geom" type="box" size="0.05 0.05 0.2" mass="1.0"/>
        <body name="link2" pos="0 0 0.4">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <geom name="link2_geom" type="box" size="0.05 0.05 0.15" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

FLOATING_BASE_ARTICULATED_XML = load_fixture_xml("floating_base_articulated")

ACTUATOR_ORDER_TEST_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0">
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
      <joint name="joint_a" axis="1 0 0" range="-1 1"/>
      <body name="link1" pos="0.2 0 0">
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
        <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
        <joint name="joint_b" axis="0 1 0" range="-1 1"/>
        <body name="link2" pos="0.2 0 0">
          <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
          <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
          <joint name="joint_c" axis="0 0 1" range="-1 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_c" joint="joint_c" kp="10"/>
    <position name="act_b" joint="joint_b" kp="10"/>
    <position name="act_a" joint="joint_a" kp="10"/>
  </actuator>
</mujoco>
"""

UNDERACTUATED_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0">
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
      <joint name="joint_a" axis="1 0 0" range="-1 1"/>
      <body name="link1" pos="0.2 0 0">
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
        <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
        <joint name="joint_b" axis="0 1 0" range="-1 1"/>
        <body name="link2" pos="0.2 0 0">
          <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
          <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
          <joint name="joint_c" axis="0 0 1" range="-1 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_c" joint="joint_c" kp="10"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


def create_fixed_base_entity():
  """Create a simple fixed-base entity."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(FIXED_BASE_XML))
  return Entity(cfg)


def create_floating_base_entity():
  """Create a floating-base entity."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(FLOATING_BASE_XML))
  return Entity(cfg)


def create_fixed_articulated_entity():
  """Create a fixed-base articulated entity (e.g., robot arm)."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(FIXED_BASE_ARTICULATED_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("joint1", "joint2"),
          effort_limit=1.0,
          stiffness=1.0,
          damping=1.0,
        ),
      )
    ),
  )
  return Entity(cfg)


def create_floating_articulated_entity():
  """Create a floating-base articulated entity."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(FLOATING_BASE_ARTICULATED_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("joint1", "joint2"),
          effort_limit=1.0,
          stiffness=1.0,
          damping=1.0,
        ),
      )
    ),
  )
  return Entity(cfg)


def initialize_entity_with_sim(entity, device, num_envs=1):
  """Initialize an entity with a simulation."""
  model = entity.compile()
  sim_cfg = SimulationCfg(njmax=75)
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


@pytest.mark.parametrize(
  "entity_fn,expected",
  [
    (
      create_fixed_base_entity,
      {
        "is_fixed_base": True,
        "is_articulated": False,
        "is_actuated": False,
        "num_bodies": 2,  # mocap_base wrapper + object
        "num_joints": 0,
        "num_actuators": 0,
      },
    ),
    (
      create_floating_base_entity,
      {
        "is_fixed_base": False,
        "is_articulated": False,
        "is_actuated": False,
        "num_bodies": 1,
        "num_joints": 0,
        "num_actuators": 0,
      },
    ),
    (
      create_fixed_articulated_entity,
      {
        "is_fixed_base": True,
        "is_articulated": True,
        "is_actuated": True,
        "num_bodies": 4,  # mocap_base wrapper + base + link1 + link2
        "num_joints": 2,
        "num_actuators": 2,
      },
    ),
    (
      create_floating_articulated_entity,
      {
        "is_fixed_base": False,
        "is_articulated": True,
        "is_actuated": True,
        "num_bodies": 3,
        "num_joints": 2,
        "num_actuators": 2,
      },
    ),
  ],
)
def test_entity_properties(entity_fn, expected):
  """Test entity type properties and element counts."""
  entity = entity_fn()
  for prop, value in expected.items():
    assert getattr(entity, prop) == value


def test_find_methods():
  """Test find methods with exact and regex matches."""
  entity = create_floating_articulated_entity()

  # Test exact matches.
  assert entity.find_bodies("base")[1] == ["base"]
  assert entity.find_joints("joint1")[1] == ["joint1"]
  assert entity.find_sites("site1")[1] == ["site1"]

  # Test regex matches.
  assert entity.find_bodies("link.*")[1] == ["link1", "link2"]
  assert entity.find_joints("joint.*")[1] == ["joint1", "joint2"]


def test_find_with_subset_filtering():
  """Test find methods with subset filtering."""
  entity = create_floating_articulated_entity()

  # Test subset filtering.
  _, names = entity.find_joints("joint1", joint_subset=["joint1", "joint2"])
  assert names == ["joint1"]

  # Test error on invalid subset.
  with pytest.raises(ValueError, match="Not all regular expressions are matched"):
    entity.find_joints("joint1", joint_subset=["joint2"])


def test_root_state_read_write(device):
  """Test root state can be written and read from simulation."""
  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  # fmt: off
  root_state = torch.tensor([
      1.0, 2.0, 3.0,           # position
      1.0, 0.0, 0.0, 0.0,      # quaternion (identity)
      0.5, 0.0, 0.0,           # linear velocity in X
      0.0, 0.0, 0.2            # angular velocity around Z
  ], device=device).unsqueeze(0)
  # fmt: on

  entity.write_root_state_to_sim(root_state)

  # Verify the state was actually written.
  q_slice = entity.data.indexing.free_joint_q_adr
  v_slice = entity.data.indexing.free_joint_v_adr
  assert torch.allclose(sim.data.qpos[:, q_slice], root_state[:, :7])
  assert torch.allclose(sim.data.qvel[:, v_slice], root_state[:, 7:])


def test_external_force_and_torque(device):
  """Test forces translate, torques rotate, and forces can be cleared."""
  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  # Apply force in X, torque around Z.
  entity.write_external_wrench_to_sim(
    forces=torch.tensor([[5.0, 0.0, 0.0]], device=sim.device),
    torques=torch.tensor([[0.0, 0.0, 3.0]], device=sim.device),
  )

  initial_pos = sim.data.qpos[0, :3].clone()
  initial_quat = sim.data.qpos[0, 3:7].clone()

  for _ in range(10):
    sim.step()

  # Verify X translation and rotation occurred.
  assert sim.data.qpos[0, 0] > initial_pos[0], "Force should cause X translation"
  assert not torch.allclose(sim.data.qpos[0, 3:7], initial_quat), (
    "Torque should cause rotation"
  )

  # Verify angular velocity is primarily around Z.
  angular_vel = sim.data.qvel[0, 3:6]
  z_rotation = abs(angular_vel[2])
  xy_rotation = abs(angular_vel[0]) + abs(angular_vel[1])
  assert z_rotation > xy_rotation * 5, "Rotation should be primarily around Z axis"


def test_external_force_clearing(device):
  """Test external forces can be cleared."""
  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  # Apply force.
  entity.write_external_wrench_to_sim(
    forces=torch.tensor([[5.0, 0.0, 0.0]], device=sim.device),
    torques=torch.tensor([[0.0, 0.0, 3.0]], device=sim.device),
  )

  # Clear forces.
  entity.write_external_wrench_to_sim(
    forces=torch.zeros((1, 3), device=sim.device),
    torques=torch.zeros((1, 3), device=sim.device),
  )

  body_id = entity.indexing.body_ids[0]
  assert torch.allclose(
    sim.data.xfrc_applied[:, body_id, :], torch.zeros(6, device=sim.device)
  )


def test_external_force_on_specific_body(device):
  """Test applying force to specific body in articulated system."""
  entity = create_floating_articulated_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  # Apply force only to link1.
  body_ids = entity.find_bodies("link1")[0]
  entity.write_external_wrench_to_sim(
    forces=torch.tensor([[3.0, 0.0, 0.0]], device=sim.device),
    torques=torch.zeros((1, 3), device=sim.device),
    body_ids=body_ids,
  )

  # Verify force applied only to link1.
  link1_id = sim.mj_model.body("link1").id
  base_id = sim.mj_model.body("base").id
  assert torch.allclose(
    sim.data.xfrc_applied[0, link1_id, :3],
    torch.tensor([3.0, 0.0, 0.0], device=sim.device),
  )
  assert torch.allclose(
    sim.data.xfrc_applied[0, base_id, :3], torch.zeros(3, device=sim.device)
  )

  # Verify motion occurs.
  initial_pos = sim.data.xpos[0, link1_id, :].clone()
  for _ in range(10):
    sim.step()
  assert not torch.allclose(sim.data.xpos[0, link1_id, :], initial_pos)


def test_fixed_base_initial_position():
  """Test fixed-base entity's initial pos/rot are applied to the mocap wrapper."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(FIXED_BASE_XML),
    init_state=EntityCfg.InitialStateCfg((1.0, 2.0, 3.0), (0.7071, 0.7071, 0.0, 0.0)),
  )
  entity = Entity(cfg)
  model = entity.compile()

  # init_state is applied to the auto-generated mocap_base wrapper body.
  mocap_body = model.body("mocap_base")
  np.testing.assert_allclose(mocap_body.pos, [1.0, 2.0, 3.0], rtol=1e-6)
  np.testing.assert_allclose(mocap_body.quat, [0.7071, 0.7071, 0.0, 0.0], atol=1e-4)


def test_keyframe_ctrl_maps_joint_pos_to_actuators():
  """Test keyframe ctrl values match init_state joint positions."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(FLOATING_BASE_ARTICULATED_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=(
            "joint1",
            "joint2",
          ),
          effort_limit=1.0,
          stiffness=1.0,
          damping=1.0,
        ),
      )
    ),
    init_state=EntityCfg.InitialStateCfg(joint_pos={"joint1": 0.5, "joint2": -0.25}),
  )
  model = Entity(cfg).compile()

  assert model.nkey == 1
  assert model.nu == 2
  assert list(model.key("init_state").ctrl) == [0.5, -0.25]


def test_keyframe_ctrl_underactuated():
  """Test ctrl is correctly constructed for an underactuated system."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(FLOATING_BASE_ARTICULATED_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("joint1",),  # Only one actuator.
          effort_limit=1.0,
          stiffness=1.0,
          damping=1.0,
        ),
      )
    ),
    init_state=EntityCfg.InitialStateCfg(joint_pos={"joint1": 0.42, "joint2": -0.99}),
  )
  model = Entity(cfg).compile()

  assert model.nu == 1
  assert model.key_ctrl[0, 0] == 0.42


def test_fixed_base_mocap_runtime_pose_change(device):
  """Test fixed-base mocap entity can have its pose changed at runtime."""

  def spec_fn():
    spec = mujoco.MjSpec.from_string(FIXED_BASE_ARTICULATED_XML)
    spec.worldbody.first_body().mocap = True
    return spec

  cfg = EntityCfg(
    spec_fn=spec_fn,
    init_state=EntityCfg.InitialStateCfg((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0)),
  )
  entity = Entity(cfg)
  entity, sim = initialize_entity_with_sim(entity, device)

  assert entity.indexing.mocap_id is not None
  assert entity.is_mocap is True

  # fmt: off
  new_pose = torch.tensor([
    5.0, 6.0, 7.0,
    1.0, 0.0, 0.0, 0.0,
  ], device=device).unsqueeze(0)
  # fmt: on
  entity.write_mocap_pose_to_sim(new_pose)

  sim.forward()
  assert torch.allclose(entity.data.root_link_pose_w, new_pose, atol=1e-5)


def test_find_joints_by_actuator_names_preserves_natural_order(device):
  """Test that find_joints_by_actuator_names returns joints in natural joint order.

  This is a regression test for a bug where joints were returned in actuator
  definition order instead of natural joint order, breaking motion tracking tasks.
  """
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ACTUATOR_ORDER_TEST_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlMotorActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

  robot = Entity(robot_cfg)
  robot.compile()

  # Natural joint order should be: joint_a, joint_b, joint_c.
  assert list(robot.joint_names) == ["joint_a", "joint_b", "joint_c"]

  # Actuator order is: act_c, act_b, act_a (reverse).
  # But find_joints_by_actuator_names should still return joints in natural order.
  joint_ids, joint_names = robot.find_joints_by_actuator_names(".*")

  # Critical: joints must be in natural order, not actuator order.
  assert joint_names == ["joint_a", "joint_b", "joint_c"]
  assert joint_ids == [0, 1, 2]

  # Verify this differs from actuator order (which is reverse).
  assert list(robot.actuator_names) == ["act_c", "act_b", "act_a"]


def test_ctrl_ids_follow_natural_joint_order(device):
  """Test that entity.indexing.ctrl_ids are in actuator definition order.

  ctrl_ids follow actuator definition order for simplicity. ONNX export builds
  the natural joint order mapping where needed.
  """
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ACTUATOR_ORDER_TEST_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlMotorActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

  robot = Entity(robot_cfg)
  mj_model = robot.compile()

  # Create simulation to initialize entity.
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=mj_model, device=device)
  robot.initialize(sim.mj_model, sim.model, sim.data, device)

  # Natural joint order: joint_a, joint_b, joint_c.
  assert list(robot.joint_names) == ["joint_a", "joint_b", "joint_c"]

  # Actuator definition order (from XML): act_c, act_b, act_a.
  assert list(robot.actuator_names) == ["act_c", "act_b", "act_a"]

  # ctrl_ids should be in actuator definition order (c, b, a).
  ctrl_ids = robot.indexing.ctrl_ids.cpu().tolist()

  # Map actuator names to their MuJoCo IDs in the compiled model.
  actuator_name_to_id = {
    mj_model.actuator(i).name.split("/")[-1]: i for i in range(mj_model.nu)
  }

  # ctrl_ids should be ordered as: act_c, act_b, act_a (actuator definition order).
  expected_ctrl_ids = [
    actuator_name_to_id["act_c"],
    actuator_name_to_id["act_b"],
    actuator_name_to_id["act_a"],
  ]
  assert ctrl_ids == expected_ctrl_ids


def test_find_joints_by_actuator_names_returns_entity_local_indices():
  """Test that find_joints_by_actuator_names returns entity-local indices."""
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(UNDERACTUATED_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlMotorActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

  robot = Entity(robot_cfg)
  robot.compile()

  # Natural joint order: joint_a (0), joint_b (1), joint_c (2).
  assert list(robot.joint_names) == ["joint_a", "joint_b", "joint_c"]

  # Only joint_c has an actuator.
  joint_ids, joint_names = robot.find_joints_by_actuator_names(".*")

  # Should return entity-local index [2], not subset-local [0].
  assert joint_names == ["joint_c"]
  assert joint_ids == [2]  # Index of joint_c in self.joint_names.


@dataclass
class CustomEntityCfg(EntityCfg):
  """Custom entity config with additional fields."""

  custom_threshold: float = 0.5

  def build(self) -> "CustomEntity":
    return CustomEntity(self)


class CustomEntity(Entity):
  """Custom entity with additional properties."""

  cfg: CustomEntityCfg

  @property
  def custom_value(self) -> float:
    """Custom property that uses config field."""
    return self.cfg.custom_threshold * 2


def test_custom_entity_subclass(device):
  """Test that custom Entity subclasses work through the scene."""
  scene_cfg = SceneCfg(
    num_envs=1,
    entities={
      "custom": CustomEntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(FIXED_BASE_XML),
        custom_threshold=0.9,
      ),
    },
  )
  scene = Scene(scene_cfg, device)

  # Scene should have instantiated our custom entity type.
  custom_entity = scene.entities["custom"]
  assert isinstance(custom_entity, CustomEntity)
  assert custom_entity.cfg.custom_threshold == 0.9
  assert custom_entity.custom_value == 1.8


# ============================================================================
# Keyframe Fallback Tests
# ============================================================================

XML_WITH_KEYFRAME = """
<mujoco>
  <worldbody>
    <body name="robot">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="link" pos="0.2 0 0">
        <joint name="joint1" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0 0 1 1 0 0 0 0.5"/>
  </keyframe>
</mujoco>
"""

XML_WITHOUT_KEYFRAME = """
<mujoco>
  <worldbody>
    <body name="robot">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="link" pos="0.2 0 0">
        <joint name="joint1" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_joint_pos_none_uses_model_keyframe():
  """Test that joint_pos=None uses the model's existing keyframe."""
  cfg = EntityCfg(
    init_state=EntityCfg.InitialStateCfg(joint_pos=None),
    spec_fn=lambda: mujoco.MjSpec.from_string(XML_WITH_KEYFRAME),
  )
  entity = Entity(cfg)
  model = entity.spec.compile()

  assert model.nkey == 1
  # Keyframe: qpos="0 0 1 1 0 0 0 0.5" (root pos/quat + joint1)
  assert model.key(0).qpos[7] == 0.5  # joint1 position


def test_joint_pos_none_errors_without_keyframe():
  """Test that joint_pos=None raises error if model has no keyframe."""
  cfg = EntityCfg(
    init_state=EntityCfg.InitialStateCfg(joint_pos=None),
    spec_fn=lambda: mujoco.MjSpec.from_string(XML_WITHOUT_KEYFRAME),
  )
  with pytest.raises(ValueError, match="requires the model to have a keyframe"):
    Entity(cfg)


XML_FIXED_BASE_WITH_KEYFRAME = """
<mujoco>
  <worldbody>
    <body name="arm">
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0.5"/>
  </keyframe>
</mujoco>
"""


def test_joint_pos_none_fixed_base_uses_keyframe():
  """Test that joint_pos=None works for fixed-base entities with keyframes."""
  cfg = EntityCfg(
    init_state=EntityCfg.InitialStateCfg(joint_pos=None),
    spec_fn=lambda: mujoco.MjSpec.from_string(XML_FIXED_BASE_WITH_KEYFRAME),
  )
  entity = Entity(cfg)
  model = entity.spec.compile()

  assert model.nkey == 1
  assert model.key(0).qpos[0] == 0.5


XML_WITH_SITES_AND_TENDONS = """
<mujoco>
  <worldbody>
    <body name="base">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1.0"/>
      <site name="site1" pos="0.1 0 0"/>
      <site name="site2" pos="0 0.1 0"/>
      <body name="link1" pos="0 0 0.2">
        <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom type="box" size="0.05 0.05 0.1" mass="0.5"/>
        <site name="site3" pos="0 0 0.1"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="tendon1">
      <joint joint="joint1" coef="1.0"/>
    </fixed>
    <fixed name="tendon2">
      <joint joint="joint2" coef="1.0"/>
    </fixed>
  </tendon>
</mujoco>
"""


def test_tendon_and_site_targets_only_allocated_when_needed(device):
  """Test that tendon/site targets are only allocated when actuators use them."""
  from mjlab.actuator import BuiltinMotorActuatorCfg
  from mjlab.actuator.actuator import TransmissionType

  # Entity with sites and tendons but NO site/tendon actuators.
  # Should allocate empty tensors for site and tendon targets.
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(XML_WITH_SITES_AND_TENDONS),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # Only joint actuators, no tendon or site actuators.
        BuiltinMotorActuatorCfg(
          target_names_expr=("joint1",),
          effort_limit=10.0,
          transmission_type=TransmissionType.JOINT,
        ),
      )
    ),
  )

  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(num_envs=4, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)

  # Verify the entity has sites and tendons.
  assert len(entity.site_names) == 3
  assert len(entity.tendon_names) == 2

  # Verify tendon and site targets are empty (not allocated).
  assert entity.data.site_effort_target.shape == (4, 0)
  assert entity.data.tendon_len_target.shape == (4, 0)
  assert entity.data.tendon_vel_target.shape == (4, 0)
  assert entity.data.tendon_effort_target.shape == (4, 0)

  # Joint targets should still be allocated (2 joints).
  assert entity.data.joint_pos_target.shape == (4, 2)
