"""Tests for contact_sensor.py."""

from __future__ import annotations

import mujoco
import pytest
import torch
from conftest import get_test_device, load_fixture_xml

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

BIPED_XML = load_fixture_xml("biped")

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
  njmax: int = 75,
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
      pattern=("left_foot_geom", "right_foot_geom"),
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

  # Run simulation and update scene to invalidate cache.
  for _ in range(20):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

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
      pattern=("left_foot_geom", "right_foot_geom"),
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
      pattern=("left_foot_geom", "right_foot_geom"),
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
    scene.update(dt=sim.cfg.mujoco.timestep)

  data1 = sensor.data
  # Check that we have ground contact initially.
  assert torch.any(data1.found > 0)

  # Jump up (lift biped off ground).
  root_state[:, 2] = 1.0
  biped_entity.write_root_state_to_sim(root_state)

  # Simulate being in air.
  for _ in range(20):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

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
    scene.update(dt=sim.cfg.mujoco.timestep)

  data3 = sensor.data
  # Should have ground contact again.
  assert torch.any(data3.found > 0)


def test_air_time_exact_at_large_sim_clock(device):
  """Air-time accumulation is independent of the sim-clock magnitude.

  Regression for issue #1101: `_update_air_time_tracking` used to accumulate
  differences of the float32 sim clock (`data.time`), so the tracked values
  inherited the clock's quantization error (ULP ~= time * 1.2e-7). That error
  grows without bound as `data.time` grows (it is never reset on env reset) and
  eventually exceeds the abs_tol in compute_first_contact, silently missing
  first-substep touchdowns.

  The fix accumulates the exact float64 substep dt instead, so `data.time` is
  never read. We advance the clock to a large value and confirm the first
  contact substep still reads exactly dt and first-contact fires. Against the
  old code this asserts hard: the tracked time picks up the clock magnitude.
  """
  cfg = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=("left_foot_geom", "right_foot_geom"),
      entity="biped",
    ),
    secondary=None,
    fields=("found",),
    track_air_time=True,
  )

  scene, sim = create_scene_with_sensor(BIPED_XML, "biped", cfg, device)
  sensor = scene["feet_contact"]
  dt = sim.cfg.mujoco.timestep

  ground_state = torch.zeros((2, 13), device=sim.device)
  ground_state[:, 2] = 0.25
  ground_state[:, 3] = 1.0
  air_state = ground_state.clone()
  air_state[:, 2] = 1.0

  # Settle the biped onto the ground so both feet are in stable contact, then
  # lift it clear so the feet are airborne (current_air_time > 0, found == 0).
  scene["biped"].write_root_state_to_sim(ground_state)
  for _ in range(20):
    sim.step()
    scene.update(dt=dt)
  scene["biped"].write_root_state_to_sim(air_state)
  for _ in range(15):
    sim.step()
    scene.update(dt=dt)
  assert torch.all(sensor.data.found == 0), "expected feet airborne before landing"

  # Advance the sim clock far past the regime where float32 quantization exceeds
  # the default abs_tol. The old code differenced this clock, so the first
  # contact substep would inherit its magnitude; the fix ignores `data.time`.
  sim.data.time[:] = 20_000.0

  # Land: a single update in contact should read exactly one dt of contact time.
  scene["biped"].write_root_state_to_sim(ground_state)
  sim.step()
  scene.update(dt=dt)

  data = sensor.data
  assert data.current_contact_time is not None
  feet_landed = data.found.view(2, len(sensor.primary_names), -1).any(dim=-1)
  assert torch.any(feet_landed), "expected at least one foot to land"

  # The first contact substep reads exactly dt, independent of the huge clock.
  assert torch.all(torch.abs(data.current_contact_time[feet_landed] - dt) < 1e-6), (
    data.current_contact_time
  )

  # And first-contact detection at dt=step_dt fires for the feet that landed.
  first_contact = sensor.compute_first_contact(dt=dt)
  assert torch.all(first_contact[feet_landed])


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
      pattern=("left_foot_geom", "right_foot_geom"),
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
      pattern=("left_foot_geom", "right_foot_geom"),
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


def test_multi_slot_air_time_and_primary_names(device):
  """Multi-slot air-time stays per-primary, primaries are exposed by name.

  Regression for issue #914: previously `num_slots > 1` with `track_air_time`
  crashed in `_update_air_time_tracking` because air-time state was [B, P]
  while `found` was [B, P * num_slots]. The fix reduces `found` across slots.
  This test pins both the regression and the new `primary_names` API.
  """
  cfg = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=("left_foot_geom", "right_foot_geom"),
      entity="biped",
    ),
    secondary=None,
    fields=("found",),
    num_slots=3,
    track_air_time=True,
  )

  scene, sim = create_scene_with_sensor(BIPED_XML, "biped", cfg, device)
  sensor = scene["feet_contact"]

  # `primary_names` reflects pattern order and indexes the per-primary axis.
  assert sensor.primary_names == ["left_foot_geom", "right_foot_geom"]

  # Settle the biped on the ground so feet are in stable contact.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.25
  root_state[:, 3] = 1.0
  scene["biped"].write_root_state_to_sim(root_state)
  for _ in range(20):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

  data = sensor.data
  assert data.found is not None
  assert data.current_contact_time is not None

  # Per-contact axis is P * num_slots; per-primary axis is P.
  assert data.found.shape == (2, 2 * 3)
  assert data.current_contact_time.shape == (2, len(sensor.primary_names))

  # Air-time update actually ran and accumulated time for primaries in
  # contact (not just "didn't crash"). Each foot reports a contact in some
  # slot, so its per-primary contact time should grow above zero.
  any_contact_per_primary = (data.found > 0).view(2, 2, 3).any(dim=-1)
  assert torch.all(any_contact_per_primary), "expected both feet in contact"
  assert torch.all(data.current_contact_time > 0)


##
# History tests.
##


def test_history_shape(device):
  """Verify history tensors have correct shape [B, N, H, 3]."""
  history_len = 5
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force", "torque", "dist"),
    history_length=history_len,
  )

  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]
  box_entity = scene["box"]

  # Place box on ground.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.11
  root_state[:, 3] = 1.0
  box_entity.write_root_state_to_sim(root_state)

  # Step a few times to populate history.
  for _ in range(10):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

  data = sensor.data

  # Verify history shapes: [B, N, H, ...].
  assert data.force_history is not None
  assert data.torque_history is not None
  assert data.dist_history is not None
  assert data.force_history.shape == (2, 1, history_len, 3)
  assert data.torque_history.shape == (2, 1, history_len, 3)
  assert data.dist_history.shape == (2, 1, history_len)


def test_history_ordering(device):
  """Verify index 0 is most recent data in history buffer."""
  history_len = 3
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("force",),
    history_length=history_len,
  )

  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]
  box_entity = scene["box"]

  # Place box on ground to get contact.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.11
  root_state[:, 3] = 1.0
  box_entity.write_root_state_to_sim(root_state)

  # Step and capture history at each step.
  forces_over_time = []
  for _ in range(5):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)
    # Clone to avoid tensor aliasing.
    forces_over_time.append(sensor.data.force.clone())

  data = sensor.data

  # Index 0 should be most recent (last force we captured).
  assert data.force_history is not None
  torch.testing.assert_close(data.force_history[:, :, 0, :], forces_over_time[-1])

  # Index 1 should be second most recent.
  torch.testing.assert_close(data.force_history[:, :, 1, :], forces_over_time[-2])

  # Index 2 should be third most recent.
  torch.testing.assert_close(data.force_history[:, :, 2, :], forces_over_time[-3])


def test_history_reset(device):
  """Verify reset clears history for specified environments."""
  history_len = 5
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("force",),
    history_length=history_len,
  )

  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]
  box_entity = scene["box"]

  # Drop box from height to ensure impact forces.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.5  # Drop from 0.5m
  root_state[:, 3] = 1.0
  box_entity.write_root_state_to_sim(root_state)

  # Let box fall and impact ground.
  for _ in range(50):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

  data_before = sensor.data
  assert data_before.force_history is not None

  # Manually set history to known non-zero values to test reset behavior.
  sensor._history_state["force"][:] = 1.0

  # Reset only env 0.
  sensor.reset(torch.tensor([0], device=device))

  data_after = sensor.data

  # Env 0 history should be zeroed.
  assert torch.all(data_after.force_history[0] == 0)

  # Env 1 history should still have our test value.
  assert torch.all(data_after.force_history[1] == 1.0)


def test_history_disabled_by_default(device):
  """Verify history is None when history_length=0 (default)."""
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("force",),
    # history_length defaults to 0
  )

  scene, _ = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]
  data = sensor.data

  # History should be None when disabled.
  assert data.force_history is None
  assert data.torque_history is None
  assert data.dist_history is None


def test_history_captures_physically_correct_forces(device):
  """Verify history captures forces that match expected physics (F = mg).

  This test validates that the history buffer stores actual physics values,
  not just that the buffer mechanics work correctly. A 1kg box at rest on
  ground should experience a net contact force of approximately 9.81 N.
  """
  history_len = 10
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("force",),
    history_length=history_len,
    reduce="netforce",  # Sum all contact forces (already in global frame).
  )

  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]
  box_entity = scene["box"]

  # Place box just above ground and let it settle.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.11  # Just above ground (box half-height is 0.1).
  root_state[:, 3] = 1.0  # Unit quaternion.
  box_entity.write_root_state_to_sim(root_state)

  # Let the box settle to steady state.
  for _ in range(100):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

  data = sensor.data

  # Box mass is 1.0 kg, gravity is ~9.81 m/s².
  # Expected normal force magnitude ≈ 9.81 N in z direction.
  expected_force_magnitude = 9.81
  tolerance = 1.0  # Allow 1 N tolerance for numerical settling.

  # Check that the most recent force in history matches expected physics.
  assert data.force_history is not None
  force_z = data.force_history[
    :, :, 0, 2
  ]  # [B, N, H, 3] -> z-component of most recent.

  # Force magnitude should match mg (sign depends on contact frame convention).
  assert torch.allclose(
    force_z.abs(), torch.full_like(force_z, expected_force_magnitude), atol=tolerance
  ), f"Expected |force_z| ~{expected_force_magnitude} N, got {force_z}"

  # Verify forces are consistent across recent history (steady state).
  # In steady state, all history entries should have similar force magnitudes.
  force_magnitudes = torch.norm(data.force_history, dim=-1)  # [B, N, H]
  mean_force = force_magnitudes.mean(dim=2, keepdim=True)
  max_deviation = (force_magnitudes - mean_force).abs().max()
  assert max_deviation < 1.0, f"Forces should be steady, max deviation: {max_deviation}"


def test_history_captures_impact_forces(device):
  """Verify history captures transient impact forces during a drop.

  This is the primary use case for the history feature: catching peak forces
  that occur during impact but might be missed if only sampling at policy rate.
  When a box drops and impacts the ground, the peak force should exceed the
  steady-state force (mg) due to the impulse from deceleration.
  """
  history_len = 20  # Capture enough substeps to see the impact transient.
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("force",),
    history_length=history_len,
    reduce="netforce",  # Sum all contact forces.
  )

  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]
  box_entity = scene["box"]

  # Drop box from a height to create impact.
  drop_height = 0.5  # 0.5m above ground (box half-height is 0.1).
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = drop_height
  root_state[:, 3] = 1.0  # Unit quaternion.
  box_entity.write_root_state_to_sim(root_state)

  # Step until we detect contact and capture the impact.
  max_force_seen = torch.zeros(2, device=sim.device)
  contact_detected = False

  for _ in range(200):  # Enough steps for box to fall and settle.
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

    data = sensor.data
    if data.force_history is not None:
      # Track the maximum force magnitude seen in history.
      force_magnitudes = torch.norm(data.force_history, dim=-1)  # [B, N, H]
      current_max = force_magnitudes.max(dim=-1).values.squeeze(-1)  # [B]
      max_force_seen = torch.maximum(max_force_seen, current_max)

      # Check if we have contact.
      if torch.any(force_magnitudes > 0):
        contact_detected = True

  assert contact_detected, "Box should have made contact with ground"

  # Steady state force is mg ≈ 9.81 N for 1 kg box.
  steady_state_force = 9.81

  # Peak impact force should exceed steady state due to impulse.
  # For a drop from 0.5m, v = sqrt(2gh) ≈ 3.1 m/s at impact.
  # The peak force depends on contact stiffness, but should be > mg.
  assert torch.all(max_force_seen > steady_state_force), (
    f"Peak impact force ({max_force_seen}) should exceed steady state ({steady_state_force})"
  )

  # Verify the peak force was significantly above steady state, demonstrating
  # that the history captured the transient impact spike.
  assert torch.all(max_force_seen > steady_state_force * 1.5), (
    f"Peak force {max_force_seen} should be significantly above mg={steady_state_force}"
  )


def test_global_frame_maxforce_rotation(device):
  """A box at rest on a plane has its contact normals all vertical."""
  cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    fields=("found", "force", "normal", "tangent"),
    reduce="maxforce",
    global_frame=True,
  )
  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", cfg, device)

  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.11
  root_state[:, 3] = 1.0
  scene["box"].write_root_state_to_sim(root_state)
  for _ in range(150):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

  sensor_force = scene["box_contact"].data.force[:, 0, :]

  # On a flat plane the contact normal is vertical, so a correctly rotated
  # global-frame force should have its magnitude entirely on the z axis.
  assert torch.all(sensor_force[:, 0].abs() < 0.05), (
    f"sensor_force x-component should be ~0, got {sensor_force[:, 0].tolist()}"
  )
  assert torch.all(sensor_force[:, 1].abs() < 0.05), (
    f"sensor_force y-component should be ~0, got {sensor_force[:, 1].tolist()}"
  )
  assert torch.all(sensor_force[:, 2].abs() > 1.0), (
    f"sensor_force z-component should be non-trivial, got {sensor_force[:, 2].tolist()}"
  )


##
# History extension tests (found/pos/normal/tangent).
##


def _place_box_on_ground(sim: Simulation, scene: Scene, z: float = 0.11):
  """Place the falling-box entity at rest just above the ground plane."""
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = z
  root_state[:, 3] = 1.0
  scene["box"].write_root_state_to_sim(root_state)


def test_found_history_shape_and_disabled(device):
  """found_history has shape [B, N, H] when enabled and is None otherwise."""
  history_len = 4
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found",),
    history_length=history_len,
  )

  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  _place_box_on_ground(sim, scene)
  for _ in range(100):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

  data = scene["box_contact"].data
  assert data.found_history is not None
  assert data.found_history.shape == (2, 1, history_len)

  # While the box rests on the ground every buffered substep saw contact.
  assert torch.all(data.found_history > 0)


def test_found_history_disabled_by_default(device):
  """found_history stays None when history_length=0."""
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found",),
  )

  scene, _ = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  assert scene["box_contact"].data.found_history is None


def test_history_rolls_found(device):
  """index 0 holds the newest substep: lifting the box writes 0 to slot 0 only."""
  history_len = 3
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found",),
    history_length=history_len,
  )

  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]

  # Box resting on ground: every substep in contact.
  _place_box_on_ground(sim, scene)
  for _ in range(100):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

  # Teleport the box far above the ground: the next substep has no contact.
  _place_box_on_ground(sim, scene, z=1.0)
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)

  found_history = sensor.data.found_history
  assert found_history is not None
  assert torch.all(found_history[:, :, 0] == 0), (
    f"index 0 should hold the airborne substep, got {found_history}"
  )
  assert torch.all(found_history[:, :, 1] > 0), (
    f"index 1 should hold the resting substep, got {found_history}"
  )


def test_vector_field_history_shapes(device):
  """pos/normal/tangent history buffers are allocated and populated."""
  history_len = 3
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force", "dist", "pos", "normal", "tangent"),
    history_length=history_len,
  )

  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  _place_box_on_ground(sim, scene)
  for _ in range(100):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

  data = scene["box_contact"].data
  for name in ("pos_history", "normal_history", "tangent_history"):
    values = getattr(data, name)
    assert values is not None
    assert values.shape == (2, 1, history_len, 3)
  # With history_length=1 the newest entry equals the current read exactly.
  torch.testing.assert_close(data.pos_history[:, :, 0, :], data.pos)
  torch.testing.assert_close(data.normal_history[:, :, 0, :], data.normal)
  torch.testing.assert_close(data.tangent_history[:, :, 0, :], data.tangent)
  torch.testing.assert_close(data.found_history[:, :, 0], data.found)
  torch.testing.assert_close(data.dist_history[:, :, 0], data.dist)


def test_found_history_reset(device):
  """reset(env_ids) zeroes found_history only for the given envs."""
  history_len = 3
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found",),
    history_length=history_len,
  )

  scene, _ = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]
  sensor._history_state["found"][:] = 1.0

  sensor.reset(torch.tensor([0], device=device))

  data = sensor.data
  assert data.found_history is not None
  assert torch.all(data.found_history[0] == 0)
  assert torch.all(data.found_history[1] == 1.0)


##
# Substep contact latch tests (catch_substep_contacts).
##


# Stiff, highly elastic contact so a dropped ball impacts and separates
# within a single control step (mirrors scripts/demos/contact_sensor_decimation.py).
BOUNCY_BALL_XML = """
<mujoco>
  <worldbody>
    <body name="ground" pos="0 0 0">
      <geom name="ground_geom" type="plane" size="5 5 0.1"/>
    </body>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.05" mass="0.1"
            solref="-1000 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_latch_disabled_outputs_are_none(device):
  """Latch outputs stay None when catch_substep_contacts=False (default)."""
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force"),
  )

  scene, _ = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  data = scene["box_contact"].data
  assert data.found_any is None
  assert data.force_peak is None
  assert data.dist_at_peak is None
  assert data.pos_at_peak is None


def test_latch_requires_found_field(device):
  """catch_substep_contacts without the found field raises."""
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("force",),
    catch_substep_contacts=True,
  )

  with pytest.raises(ValueError, match="requires 'found'"):
    ContactSensorCfg.build(sensor_cfg)


def test_latch_found_any_or_semantics(device):
  """found_any stays True after contact resolves within the control step."""
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force"),
    catch_substep_contacts=True,
  )

  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]

  # Settle the box on the ground, then latch three resting substeps followed
  # by one airborne substep.
  _place_box_on_ground(sim, scene)
  for _ in range(100):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

  sensor.begin_control_step()
  for _ in range(3):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)
  _place_box_on_ground(sim, scene, z=1.0)
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)

  data = sensor.data
  assert data.found is not None
  assert torch.all(data.found == 0), "box should be airborne on the last substep"
  assert data.found_any is not None
  assert torch.all(data.found_any), (
    "found_any should latch the earlier resting substeps"
  )


def test_latch_force_peak_matches_substep_maximum(device):
  """force_peak/dist_at_peak equal the largest-magnitude substep values."""
  history_len = 8
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force", "dist"),
    history_length=history_len,
    catch_substep_contacts=True,
  )

  scene, sim = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]

  # Drop the box and wait for first ground contact, then capture one
  # control-step window of the settling impact where forces vary per substep.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.5
  root_state[:, 3] = 1.0
  scene["box"].write_root_state_to_sim(root_state)

  landed = False
  for _ in range(300):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)
    if torch.all(sensor.data.found > 0):
      landed = True
      break
  assert landed, "box should have reached the ground"

  sensor.begin_control_step()
  forces = []
  dists = []
  for _ in range(history_len):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)
    forces.append(sensor.data.force.clone())
    dists.append(sensor.data.dist.clone())

  force_sequence = torch.stack(forces, dim=2)  # [B, N, H, 3]
  mag = force_sequence.norm(dim=-1)  # [B, N, H]
  peak_idx = mag.argmax(dim=2)  # [B, N]

  data = sensor.data
  assert data.found_any is not None
  assert torch.any(data.found_any), "impact should have latched a contact"
  assert data.force_peak is not None
  for b in range(2):
    for n in range(1):
      torch.testing.assert_close(
        data.force_peak[b, n], force_sequence[b, n, peak_idx[b, n]]
      )
      torch.testing.assert_close(data.dist_at_peak[b, n], dists[peak_idx[b, n]][b, n])


def test_latch_begin_control_step_clears(device):
  """begin_control_step resets found_any and peak trackers."""
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force", "pos"),
    catch_substep_contacts=True,
  )

  scene, _ = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]
  sensor._latch_state["found_any"][:] = True
  sensor._latch_state["peak_mag"][:] = 5.0
  sensor._latch_state["force_peak"][:] = 9.0
  sensor._latch_state["pos_at_peak"][:] = 9.0

  sensor.begin_control_step()

  data = sensor.data
  assert data.found_any is not None
  assert torch.all(~data.found_any)
  assert data.force_peak is not None
  assert torch.all(data.force_peak == 0)
  assert data.pos_at_peak is not None
  assert torch.all(data.pos_at_peak == 0)


def test_latch_reset_scoped(device):
  """reset(env_ids) clears the latch only for the listed envs."""
  sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force"),
    catch_substep_contacts=True,
  )

  scene, _ = create_scene_with_sensor(FALLING_BOX_XML, "box", sensor_cfg, device)

  sensor = scene["box_contact"]
  sensor._latch_state["found_any"][:] = True
  sensor._latch_state["peak_mag"][:] = 5.0
  sensor._latch_state["force_peak"][:] = 9.0

  sensor.reset(torch.tensor([0], device=device))

  data = sensor.data
  assert data.found_any is not None
  assert not data.found_any[0].any()
  assert data.found_any[1].all()
  assert data.force_peak is not None
  assert torch.all(data.force_peak[0] == 0)
  assert torch.all(data.force_peak[1] == 9.0)


def test_latch_catches_impact_missed_by_final_substep(device):
  """Integration: a bouncing ball's brief impact latches even though the
  final substep of the control step reports no contact."""
  decimation = 20
  sensor_cfg = ContactSensorCfg(
    name="ball_contact",
    primary=ContactMatch(mode="geom", pattern="ball_geom", entity="ball"),
    secondary=None,
    fields=("found", "force"),
    catch_substep_contacts=True,
  )

  scene, sim = create_scene_with_sensor(BOUNCY_BALL_XML, "ball", sensor_cfg, device)

  sensor = scene["ball_contact"]

  # Drop the ball from 0.5 m and scan control-step windows around the bounces.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.5
  root_state[:, 3] = 1.0
  scene["ball"].write_root_state_to_sim(root_state)

  latched_only = 0
  for _ in range(40):
    sensor.begin_control_step()
    for _ in range(decimation):
      sim.step()
      scene.update(dt=sim.cfg.mujoco.timestep)
    data = sensor.data
    instant = data.found[:, 0] > 0
    latched = data.found_any[:, 0]
    assert torch.all(latched | ~instant), (
      "found_any must be a superset of the instantaneous read"
    )
    latched_only += int(torch.sum(latched & ~instant).item())

  assert latched_only > 0, (
    "expected at least one control step where the impact was latched but the "
    "final-substep snapshot reported no contact"
  )


##
# Fast-ball crossing regression test.
##


# Ball and paddle live in one entity so both geoms resolve in the same scope.
# The paddle is a jointless (static) thin box; the ball is launched fast
# enough to enter and leave the paddle's contact zone within a single
# control step of decimation substeps. Stiff near-elastic contact makes the
# ball rebound off the paddle rather than tunnel through it.
FAST_BALL_PADDLE_XML = """
<mujoco>
  <worldbody>
    <body name="paddle" pos="0 0 0.15">
      <geom name="paddle_geom" type="box" size="0.01 0.15 0.15" mass="100"/>
    </body>
    <body name="ball" pos="-0.25 0 0.15">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.033" mass="0.058"
            solref="-100000 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_fast_ball_crossing_latched_but_missed_by_snapshot(device):
  """Regression: a fast ball hitting a thin paddle mid-control-step.

  At 10 m/s and a 2 ms physics timestep the ball crosses the paddle's
  contact zone in ~4 of the 20 substeps of one control step. The
  final-substep ``found`` snapshot is 0 (the ball is already past the
  paddle), which is exactly the tennis-style collision that motivated
  ``catch_substep_contacts`` and the history extension.
  """
  decimation = 20
  sensor_cfg = ContactSensorCfg(
    name="ball_contact",
    primary=ContactMatch(mode="geom", pattern="ball_geom", entity="ball"),
    secondary=ContactMatch(mode="geom", pattern="paddle_geom", entity="ball"),
    fields=("found", "force", "dist", "pos"),
    history_length=decimation,
    catch_substep_contacts=True,
  )

  scene, sim = create_scene_with_sensor(
    FAST_BALL_PADDLE_XML, "ball", sensor_cfg, device
  )

  sensor = scene["ball_contact"]

  # Launch the ball at +10 m/s from 0.25 m before the paddle plane.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 0] = -0.25
  root_state[:, 2] = 0.15
  root_state[:, 3] = 1.0
  root_state[:, 7] = 10.0
  scene["ball"].write_root_state_to_sim(root_state)

  sensor.begin_control_step()
  for _ in range(decimation):
    sim.step()
    scene.update(dt=sim.cfg.mujoco.timestep)

  data = sensor.data

  # The final-substep snapshot missed the collision entirely: the ball is
  # already past the paddle.
  assert torch.all(data.found == 0), (
    "ball should have left the contact zone before the final substep; "
    "tune the launch speed so the crossing fits inside one control step"
  )

  # The latch caught it.
  assert data.found_any is not None
  assert torch.all(data.found_any), "latch must record the mid-step collision"
  assert data.force_peak is not None
  assert torch.all(data.force_peak.norm(dim=-1) > 0), "impact must carry force"
  assert data.pos_at_peak is not None
  assert torch.all(data.pos_at_peak[..., 0].abs() < 0.1), (
    f"impact should sit near the paddle plane, got {data.pos_at_peak[..., 0]}"
  )

  # The extended history caught it too.
  assert data.found_history is not None
  assert torch.all((data.found_history > 0).any(dim=-1)), (
    "found_history must contain at least one contact substep"
  )
