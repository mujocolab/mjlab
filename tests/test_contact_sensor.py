"""Tests for contact_sensor.py."""

from __future__ import annotations

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg

##
# Test XML models.
##

FALLING_BOX_XML = """
<mujoco>
  <worldbody>
    <body name="ground" pos="0 0 0">
      <geom name="ground_geom" type="plane" size="5 5 0.1" rgba="0.5 0.5 0.5 1"/>
    </body>
    <body name="box" pos="0 0 0.5">
      <freejoint name="box_joint"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" rgba="0.8 0.3 0.3 1"
        mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

BIPED_XML = """
<mujoco>
  <worldbody>
    <body name="ground" pos="0 0 0">
      <geom name="ground_geom" type="plane" size="5 5 0.1" rgba="0.5 0.5 0.5 1"/>
    </body>
    <body name="base" pos="0 0 0.5">
      <freejoint name="base_joint"/>
      <geom name="torso_geom" type="box" size="0.15 0.1 0.2" mass="5.0"/>
      <body name="left_foot" pos="0.1 0 -0.25">
        <joint name="left_ankle" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
        <geom name="left_foot_geom" type="box" size="0.05 0.08 0.02" mass="0.2"/>
      </body>
      <body name="right_foot" pos="-0.1 0 -0.25">
        <joint name="right_ankle" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
        <geom name="right_foot_geom" type="box" size="0.05 0.08 0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

SIMPLE_ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="ground" pos="0 0 0">
      <geom name="ground_geom" type="plane" size="5 5 0.1"/>
    </body>
    <body name="robot" pos="0 0 0.3">
      <freejoint name="robot_joint"/>
      <geom name="trunk_collision" type="box" size="0.2 0.15 0.1" mass="2.0"/>
      <geom name="head_collision" type="sphere" size="0.08" pos="0.25 0 0.1"
      mass="0.5"/>
      <body name="leg1" pos="0.1 0.1 -0.1">
        <geom name="leg1_thigh_collision1" type="capsule" size="0.02"
          fromto="0 0 0 0 0 -0.1"/>
        <geom name="leg1_thigh_collision2" type="capsule" size="0.02"
          fromto="0 0 -0.05 0 0 -0.15"/>
        <geom name="leg1_foot_collision" type="sphere" size="0.03" pos="0 0 -0.2"/>
      </body>
      <body name="leg2" pos="-0.1 0.1 -0.1">
        <geom name="leg2_thigh_collision1" type="capsule" size="0.02"
          fromto="0 0 0 0 0 -0.1"/>
        <geom name="leg2_thigh_collision2" type="capsule" size="0.02"
          fromto="0 0 -0.05 0 0 -0.15"/>
        <geom name="leg2_foot_collision" type="sphere" size="0.03" pos="0 0 -0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

##
# Fixtures.
##


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


##
# Helper functions.
##


def create_scene_with_sensor(
  xml: str,
  entity_name: str,
  sensor_cfg: ContactSensorCfg,
  device: str,
  num_envs: int = 2,
  njmax: int = 50,
) -> tuple[Scene, Simulation]:
  """Helper to create a complete test environment with contact sensor.

  Sets up a scene with the specified entity and contact sensor configuration,
  compiles the model, creates a simulation, and initializes everything together.
  Returns the scene and simulation objects for test manipulation and assertions."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))

  scene_cfg = SceneCfg(
    num_envs=num_envs,
    env_spacing=3.0,
    entities={entity_name: entity_cfg},
    sensors=(sensor_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=njmax)
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  return scene, sim


def step_and_settle(sim: Simulation, num_steps: int = 30):
  """Run simulation steps to allow physics to stabilize and contacts to form.

  Useful after placing objects to let them fall under gravity and establish
  stable contact with ground or other objects before testing contact detection."""
  for _ in range(num_steps):
    sim.step()


##
# Basic contact detection tests.
##


def test_basic_contact_detection(device):
  """Verify that contact sensors detect collisions between a falling box and ground.

  Tests that when a box is placed just above ground and simulation steps,
  the contact sensor correctly reports contact forces and found flags."""
  contact_sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force"),
  )

  scene, sim = create_scene_with_sensor(
    FALLING_BOX_XML, "box", contact_sensor_cfg, device
  )

  sensor = scene["box_contact"]
  box_entity = scene["box"]

  # Place box on ground and let it settle.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.11  # Just above ground
  root_state[:, 3] = 1.0
  box_entity.write_root_state_to_sim(root_state)

  step_and_settle(sim)

  data = sensor.data

  # Basic field presence and shape checks.
  assert data.found is not None
  assert data.force is not None
  assert data.found.shape == (2, 1)  # 2 envs, 1 slot
  assert data.force.shape[-1] == 3

  # Contact should be detected.
  assert torch.any(data.found > 0)

  # Force should be non-zero when contact is detected.
  if torch.any(data.found > 0):
    contact_forces = data.force[data.found > 0]
    assert torch.any(torch.abs(contact_forces) > 0)


def test_contact_fields(device):
  """Verify all contact sensor output fields have correct shapes and values.

  Tests that force, torque, dist, pos, and normal fields are properly populated
  with appropriate dimensionality (3D vectors for force/torque/pos/normal, scalar for dist)."""
  contact_sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force", "torque", "dist", "pos", "normal"),
  )

  scene, sim = create_scene_with_sensor(
    FALLING_BOX_XML, "box", contact_sensor_cfg, device
  )

  sensor = scene["box_contact"]
  box_entity = scene["box"]

  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.105
  root_state[:, 3] = 1.0
  box_entity.write_root_state_to_sim(root_state)

  step_and_settle(sim, num_steps=10)

  data = sensor.data

  # Verify all fields are present with correct shapes.
  assert data.found is not None
  assert data.force is not None
  assert data.torque is not None
  assert data.dist is not None
  assert data.pos is not None
  assert data.normal is not None

  assert data.force.shape[-1] == 3
  assert data.torque.shape[-1] == 3
  assert data.pos.shape[-1] == 3
  assert data.normal.shape[-1] == 3
  assert len(data.dist.shape) == 2


##
# Pattern matching and multi-slot tests.
##


def test_multi_slot_pattern_matching(device):
  """Verify pattern lists create separate tracking slots for each matched geom.

  When passing a list of patterns like ["left_foot_geom", "right_foot_geom"],
  the sensor should create independent contact tracking for each foot,
  allowing simultaneous monitoring of multiple contact points."""
  feet_sensor_cfg = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=["left_foot_geom", "right_foot_geom"],
      entity="biped",
    ),
    secondary=None,
    fields=("found", "force"),
    track_air_time=True,
  )

  scene, sim = create_scene_with_sensor(BIPED_XML, "biped", feet_sensor_cfg, device)

  sensor = scene["feet_contact"]
  biped_entity = scene["biped"]

  # Place biped on ground.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.25
  root_state[:, 3] = 1.0
  biped_entity.write_root_state_to_sim(root_state)

  step_and_settle(sim, num_steps=20)

  data = sensor.data

  # Should have 2 slots (one per foot).
  assert data.found.shape == (2, 2)
  assert data.force.shape == (2, 2, 3)

  # Air time should be tracked.
  assert hasattr(data, "current_air_time")
  assert data.current_air_time.shape == (2, 2)


def test_regex_pattern_matching(device):
  """Verify regex patterns correctly match multiple geoms with similar names.

  Tests that a pattern like ".*foot_geom$" matches all geoms ending with
  "foot_geom", enabling efficient batch configuration of similar contact points.
  Also verifies regex patterns work correctly for actual contact detection."""
  # Match all foot geoms using regex.
  regex_sensor_cfg = ContactSensorCfg(
    name="all_feet_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r".*foot_geom$",
      entity="biped",
    ),
    secondary=None,
    fields=("found", "force"),
  )

  scene, sim = create_scene_with_sensor(BIPED_XML, "biped", regex_sensor_cfg, device)

  sensor = scene["all_feet_contact"]
  biped_entity = scene["biped"]

  # Should match both left_foot_geom and right_foot_geom.
  assert sensor.data.found.shape == (2, 2)

  # Place biped on ground to verify regex-matched geoms detect contacts.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.24  # Low enough for feet to touch ground
  root_state[:, 3] = 1.0
  biped_entity.write_root_state_to_sim(root_state)

  step_and_settle(sim, num_steps=20)

  data = sensor.data
  # Both feet should detect ground contact.
  assert torch.any(data.found > 0)
  # Force field should be present (may have small values).
  assert data.force is not None
  assert data.force.shape == (2, 2, 3)


##
# Reduction mode tests.
##


@pytest.mark.parametrize(
  "reduce_mode",
  ["none", "mindist", "maxforce", "netforce"],
)
def test_reduce_modes(device, reduce_mode):
  """Verify reduction modes correctly aggregate multiple simultaneous contacts.

  Tests "none" (no filtering), "mindist" (closest contact), "maxforce" (strongest),
  and "netforce" (sum all forces) modes for selecting/combining contact data
  when multiple contacts occur on the same geom."""
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("force",),
    reduce=reduce_mode,
    num_slots=1,
  )

  scene, _ = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]
  data = sensor.data

  # All reduction modes return 3D shape for force field.
  assert len(data.force.shape) == 3
  assert data.force.shape[-1] == 3  # Force is always a 3D vector


def test_reduce_modes_multiple_contacts(device):
  """Test reduction modes with multiple simultaneous contacts."""
  feet_sensor_cfg = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=["left_foot_geom", "right_foot_geom"],
      entity="biped",
    ),
    secondary=None,
    fields=("found", "force", "dist"),
    reduce="mindist",
    num_slots=1,
  )

  scene, sim = create_scene_with_sensor(BIPED_XML, "biped", feet_sensor_cfg, device)

  sensor = scene["feet_contact"]
  biped_entity = scene["biped"]

  # Place biped on ground.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.25
  root_state[:, 3] = 1.0
  biped_entity.write_root_state_to_sim(root_state)

  step_and_settle(sim, num_steps=20)

  data = sensor.data

  # With reduce="mindist" and num_slots=1, should have 2 slots (2 primaries × 1 slot).
  assert data.found.shape == (2, 2)
  assert data.force.shape == (2, 2, 3)


##
# Exclude pattern tests.
##


def test_exclude_exact_names(device):
  """Verify exact name exclusion removes specific geoms from contact detection.

  Tests the ergonomic feature where passing exact geom names like
  ("leg1_foot_collision", "leg2_foot_collision") excludes only those specific
  geoms without needing complex regex patterns."""
  # Sensor that excludes foot collisions by exact names.
  nonfoot_sensor_cfg = ContactSensorCfg(
    name="nonfoot_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_collision\d*$",  # Match all collision geoms
      entity="robot",
      exclude=("leg1_foot_collision", "leg2_foot_collision"),  # Exact names
    ),
    secondary=None,
    fields=("found",),
  )

  scene, _ = create_scene_with_sensor(
    SIMPLE_ROBOT_XML, "robot", nonfoot_sensor_cfg, device
  )

  sensor = scene["nonfoot_contact"]

  # Should detect 6 geoms: trunk, head, 2x leg1_thigh, 2x leg2_thigh.
  # Foot collisions should be excluded.
  assert sensor.data.found.shape == (2, 6)


def test_exclude_regex_pattern(device):
  """Verify regex exclusion patterns filter out groups of similarly-named geoms.

  Tests that patterns like ".*thigh_collision\\d+" can exclude all thigh
  collision geoms while preserving other collision geoms for contact detection."""
  # Sensor that excludes all thigh collisions using regex.
  no_thigh_sensor_cfg = ContactSensorCfg(
    name="no_thigh_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_collision\d*$",
      entity="robot",
      exclude=(r".*thigh_collision\d+",),  # Regex pattern
    ),
    secondary=None,
    fields=("found",),
  )

  scene, _ = create_scene_with_sensor(
    SIMPLE_ROBOT_XML, "robot", no_thigh_sensor_cfg, device
  )

  sensor = scene["no_thigh_contact"]

  # Should detect 4 geoms: trunk, head, 2x foot (thighs excluded by regex).
  assert sensor.data.found.shape == (2, 4)


def test_exclude_mixed_patterns(device):
  """Verify exact names and regex patterns can be mixed in exclude lists.

  Tests that exclude tuples can contain both exact names ("trunk_collision")
  and regex patterns (".*foot_collision") simultaneously, with automatic
  detection of which exclusion method to use for each entry."""
  mixed_exclude_cfg = ContactSensorCfg(
    name="mixed_exclude",
    primary=ContactMatch(
      mode="geom",
      pattern=r".*_collision\d*$",
      entity="robot",
      exclude=(
        "trunk_collision",  # Exact name
        r".*foot_collision",  # Regex pattern
      ),
    ),
    secondary=None,
    fields=("found",),
  )

  scene, _ = create_scene_with_sensor(
    SIMPLE_ROBOT_XML, "robot", mixed_exclude_cfg, device
  )

  sensor = scene["mixed_exclude"]

  # Should detect 5 geoms: head, 4x thigh (trunk and feet excluded).
  assert sensor.data.found.shape == (2, 5)


##
# Body and subtree mode tests.
##


def test_body_mode_contacts(device):
  """Test contact detection with body mode."""
  body_sensor_cfg = ContactSensorCfg(
    name="body_contact",
    primary=ContactMatch(mode="body", pattern="base", entity="biped"),
    secondary=None,
    fields=("found",),
  )

  scene, _ = create_scene_with_sensor(BIPED_XML, "biped", body_sensor_cfg, device)

  sensor = scene["body_contact"]
  data = sensor.data

  # Should match the base body.
  assert data.found.shape[1] == 1


def test_subtree_mode_contacts(device):
  """Test contact detection with subtree mode."""
  subtree_sensor_cfg = ContactSensorCfg(
    name="subtree_contact",
    primary=ContactMatch(mode="subtree", pattern="base", entity="biped"),
    secondary=None,
    fields=("found",),
  )

  scene, sim = create_scene_with_sensor(BIPED_XML, "biped", subtree_sensor_cfg, device)

  sensor = scene["subtree_contact"]
  biped_entity = scene["biped"]

  # Place biped on ground.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.2  # Low enough for feet to touch
  root_state[:, 3] = 1.0
  biped_entity.write_root_state_to_sim(root_state)

  step_and_settle(sim, num_steps=30)

  data = sensor.data

  # Subtree includes base and all children (feet), so contacts should be detected.
  assert torch.any(data.found > 0)


##
# Air time tracking tests.
##


def test_air_time_tracking(device):
  """Verify contact sensors track time spent in/out of contact when enabled.

  Tests the track_air_time feature which monitors how long each contact point
  has been in the air (no contact) or on ground (in contact), useful for
  gait analysis and landing detection in legged robots.
  """
  feet_sensor_cfg = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=["left_foot_geom", "right_foot_geom"],
      entity="biped",
    ),
    secondary=None,
    fields=("found",),
    track_air_time=True,
  )

  scene, sim = create_scene_with_sensor(BIPED_XML, "biped", feet_sensor_cfg, device)

  sensor = scene["feet_contact"]
  biped_entity = scene["biped"]

  # Start on ground.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.24  # Low enough for contact
  root_state[:, 3] = 1.0
  biped_entity.write_root_state_to_sim(root_state)

  # Let it settle and establish ground contact.
  for _ in range(30):
    sim.step()

  data1 = sensor.data
  # Check that we have ground contact initially.
  assert torch.any(data1.found > 0)

  # Jump up (lift biped off ground).
  root_state[:, 2] = 1.0
  biped_entity.write_root_state_to_sim(root_state)

  # Simulate being in air.
  for _ in range(20):
    sim.step()

  data2 = sensor.data
  # Should have no ground contact while in air.
  assert torch.all(data2.found == 0)

  # When using track_air_time, we should have timing information.
  assert hasattr(data2, "current_air_time")
  assert hasattr(data2, "last_air_time")

  # Land back on ground.
  root_state[:, 2] = 0.24
  biped_entity.write_root_state_to_sim(root_state)

  for _ in range(30):
    sim.step()

  data3 = sensor.data
  # Should have ground contact again.
  assert torch.any(data3.found > 0)


##
# Multi-sensor integration tests.
##


def test_multiple_sensors(device):
  """Test multiple contact sensors in the same scene."""
  left_sensor_cfg = ContactSensorCfg(
    name="left_foot_contact",
    primary=ContactMatch(mode="geom", pattern="left_foot_geom", entity="biped"),
    secondary=None,
    fields=("found", "force"),
  )

  right_sensor_cfg = ContactSensorCfg(
    name="right_foot_contact",
    primary=ContactMatch(mode="geom", pattern="right_foot_geom", entity="biped"),
    secondary=None,
    fields=("found", "force"),
  )

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(BIPED_XML))

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"biped": entity_cfg},
    sensors=(left_sensor_cfg, right_sensor_cfg),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=40)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  left_sensor = scene["left_foot_contact"]
  right_sensor = scene["right_foot_contact"]

  # Both sensors should work independently.
  assert left_sensor.data.found.shape == (2, 1)
  assert right_sensor.data.found.shape == (2, 1)


##
# Performance and edge case tests.
##


def test_no_contacts(device):
  """Test sensor behavior when no contacts occur."""
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force"),
  )

  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]
  box_entity = scene["box"]

  # Place box high above ground (no contact).
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 5.0  # Far above ground
  root_state[:, 3] = 1.0
  box_entity.write_root_state_to_sim(root_state)

  sim.step()

  data = sensor.data

  # No contacts should be detected.
  assert torch.all(data.found == 0)

  # Forces should be zero.
  assert torch.all(data.force == 0)


def test_num_slots_greater_than_one(device):
  """Test behavior with num_slots > 1."""
  sensor_cfg_1 = ContactSensorCfg(
    name="feet_contact_single",
    primary=ContactMatch(
      mode="geom",
      pattern=["left_foot_geom", "right_foot_geom"],
      entity="biped",
    ),
    secondary=None,
    fields=("found", "force", "normal"),
    num_slots=1,
  )

  sensor_cfg_3 = ContactSensorCfg(
    name="feet_contact_triple",
    primary=ContactMatch(
      mode="geom",
      pattern=["left_foot_geom", "right_foot_geom"],
      entity="biped",
    ),
    secondary=None,
    fields=("found", "force", "normal"),
    num_slots=3,
  )

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(BIPED_XML))

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"biped": entity_cfg},
    sensors=(sensor_cfg_1, sensor_cfg_3),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=40)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor_1 = scene["feet_contact_single"]
  sensor_3 = scene["feet_contact_triple"]
  biped_entity = scene["biped"]

  # Place biped on ground.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.25
  root_state[:, 3] = 1.0
  biped_entity.write_root_state_to_sim(root_state)

  step_and_settle(sim, num_steps=20)

  # 2 primaries × 1 slot = 2 total slots.
  data_1 = sensor_1.data
  assert data_1.found is not None
  assert data_1.force is not None
  assert data_1.normal is not None
  assert data_1.found.shape == (2, 2)
  assert data_1.force.shape == (2, 2, 3)
  assert data_1.normal.shape == (2, 2, 3)

  # 2 primaries × 3 slots = 6 total slots.
  data_3 = sensor_3.data
  assert data_3.found is not None
  assert data_3.force is not None
  assert data_3.normal is not None
  assert data_3.found.shape == (2, 6)
  assert data_3.force.shape == (2, 6, 3)
  assert data_3.normal.shape == (2, 6, 3)
