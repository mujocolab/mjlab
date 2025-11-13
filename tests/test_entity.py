"""Tests for entity module."""

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
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

FLOATING_BASE_ARTICULATED_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
        <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
        <site name="site1" pos="0 0 0"/>
      </body>
      <body name="link2" pos="0 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="0 1.57"/>
        <geom name="link2_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <jointpos name="joint1_pos" joint="joint1"/>
  </sensor>
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
          joint_names_expr=("joint1", "joint2"),
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
          joint_names_expr=("joint1", "joint2"),
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
  sim_cfg = SimulationCfg()
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
        "num_bodies": 1,
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
        "num_bodies": 3,
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
  """Test fixed-base entity's initial pos/rot are applied to the body."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(FIXED_BASE_XML),
    init_state=EntityCfg.InitialStateCfg((1.0, 2.0, 3.0), (0.7071, 0.7071, 0.0, 0.0)),
  )
  entity = Entity(cfg)
  model = entity.compile()

  body = model.body("object")
  np.testing.assert_allclose(body.pos, [1.0, 2.0, 3.0], rtol=1e-6)
  np.testing.assert_allclose(body.quat, [0.7071, 0.7071, 0.0, 0.0], atol=1e-4)


def test_keyframe_ctrl_maps_joint_pos_to_actuators():
  """Test keyframe ctrl values match init_state joint positions."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(FLOATING_BASE_ARTICULATED_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          joint_names_expr=(
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
          joint_names_expr=("joint1",),  # Only one actuator.
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
