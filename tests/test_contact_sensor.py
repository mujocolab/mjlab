"""Tests for ContactSensor functionality."""

from __future__ import annotations

import mujoco
import pytest
import torch
import warp as wp

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg

wp.config.quiet = True


def get_test_device() -> str:
  """Get device for testing, preferring CUDA if available."""
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def falling_box_xml():
  """XML for a simple box that can fall and make contact with ground."""
  return """
    <mujoco>
      <worldbody>
        <body name="ground" pos="0 0 0">
          <geom name="ground_geom" type="plane" size="5 5 0.1" rgba="0.5 0.5 0.5 1"/>
        </body>
        <body name="box" pos="0 0 0.5">
          <freejoint name="box_joint"/>
          <geom name="box_geom" type="box" size="0.1 0.1 0.1" rgba="0.8 0.3 0.3 1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
  """


@pytest.fixture
def biped_xml():
  """XML for a simple biped with two feet that can make contact with ground."""
  return """
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


def test_basic_contact_detection(falling_box_xml, device):
  """Test basic contact sensor setup and contact detection."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(falling_box_xml))

  contact_sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force"),
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"box": entity_cfg},
    sensors=(contact_sensor_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["box_contact"]
  box_entity = scene["box"]

  # Place box on ground and let it settle.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.11  # Just above ground
  root_state[:, 3] = 1.0
  box_entity.write_root_state_to_sim(root_state)

  # Step simulation to establish contact.
  for _ in range(30):
    sim.step()

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


def test_contact_fields(falling_box_xml, device):
  """Test that all contact field types work correctly."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(falling_box_xml))

  contact_sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force", "torque", "dist", "pos", "normal"),
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"box": entity_cfg},
    sensors=(contact_sensor_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["box_contact"]
  box_entity = scene["box"]

  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.105
  root_state[:, 3] = 1.0
  box_entity.write_root_state_to_sim(root_state)

  for _ in range(10):
    sim.step()

  data = sensor.data

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


def test_multi_slot_pattern_matching(biped_xml, device):
  """Test that list patterns create multiple slots correctly."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(biped_xml))

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

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"biped": entity_cfg},
    sensors=(feet_sensor_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=40)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["feet_contact"]
  biped_entity = scene["biped"]

  # Verify sensor created 2 slots (one per foot).
  assert len(sensor._slots) == 2

  # Place biped on ground.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.25
  root_state[:, 3] = 1.0
  biped_entity.write_root_state_to_sim(root_state)

  for _ in range(20):
    sim.step()

  data = sensor.data

  # Should have 2 slots in data.
  assert data.found is not None
  assert data.found.shape[-1] == 2
  assert torch.any(data.found > 0)

  # Air time state should also have 2 slots.
  assert sensor._air_time_state is not None
  assert sensor._air_time_state.current_air_time.shape == (2, 2)  # 2 envs, 2 slots


def test_air_time_tracking_with_transitions(falling_box_xml, device):
  """Test air time tracking, accumulation, and air-contact transitions."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(falling_box_xml))

  contact_sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force"),
    track_air_time=True,
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"box": entity_cfg},
    sensors=(contact_sensor_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["box_contact"]
  box_entity = scene["box"]

  # Verify air time state was initialized.
  assert sensor._air_time_state is not None
  assert sensor._air_time_state.current_air_time.shape == (2, 1)

  # Test 1: Air time accumulation.
  # Box starts at z=0.5, in the air.
  sensor.reset()
  dt = sim.mj_model.opt.timestep
  num_steps = 10

  for _ in range(num_steps):
    sim.step()
    sensor.update(dt)

  # Air time should accumulate.
  expected_air_time = num_steps * dt
  assert torch.all(sensor._air_time_state.current_air_time > 0.0)
  assert torch.allclose(
    sensor._air_time_state.current_air_time,
    torch.full_like(sensor._air_time_state.current_air_time, expected_air_time),
    atol=1e-5,
  )

  # Test 2: Air-to-contact transition saves last_air_time.
  # Continue until box lands.
  max_steps = 200
  landed = False
  for _ in range(max_steps):
    sim.step()
    sensor.update(dt)
    if torch.any(sensor.data.found > 0):
      landed = True
      break

  assert landed, "Box should have landed"
  assert torch.all(sensor._air_time_state.last_air_time > 0.0)
  assert torch.all(sensor._air_time_state.current_air_time == 0.0)
  assert torch.all(sensor._air_time_state.current_contact_time > 0.0)

  # Test 3: Contact time accumulation.
  contact_time_before = sensor._air_time_state.current_contact_time.clone()
  for _ in range(10):
    sim.step()
    sensor.update(dt)

  assert torch.all(sensor._air_time_state.current_contact_time > contact_time_before)

  # Test 4: Contact-to-air transition saves last_contact_time.
  contact_time_before_liftoff = sensor.data.current_contact_time.clone()

  # Apply upward force to lift box.
  box_entity.write_external_wrench_to_sim(
    forces=torch.tensor([[0.0, 0.0, 50.0]] * 2, device=sim.device),
    torques=torch.zeros((2, 3), device=sim.device),
  )

  # Step until liftoff.
  lifted_off = False
  for _ in range(100):
    sim.step()
    sensor.update(dt)
    if torch.all(sensor.data.found == 0):
      lifted_off = True
      break

  assert lifted_off, "Box should have lifted off"
  assert torch.all(sensor._air_time_state.last_contact_time > 0.0)
  # last_contact_time should be at least as much as we had before applying force
  # (it accumulates until actual liftoff)
  assert torch.all(
    sensor._air_time_state.last_contact_time >= contact_time_before_liftoff
  )
  assert torch.all(sensor._air_time_state.current_contact_time == 0.0)
  assert torch.all(sensor._air_time_state.current_air_time > 0.0)

  # Test 5: Air time fields exposed in sensor.data.
  data = sensor.data
  assert data.current_air_time is not None
  assert data.last_air_time is not None
  assert data.current_contact_time is not None
  assert data.last_contact_time is not None


def test_first_contact_detection(falling_box_xml, device):
  """Test compute_first_contact at landing and during sustained contact."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(falling_box_xml))

  contact_sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force"),
    track_air_time=True,
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"box": entity_cfg},
    sensors=(contact_sensor_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["box_contact"]
  box_entity = scene["box"]

  # Test 1: First contact at landing.
  sensor.reset()
  dt = sim.mj_model.opt.timestep

  first_contact_detected = False
  for _ in range(200):
    sim.step()
    sensor.update(dt)

    first_contact = sensor.compute_first_contact(dt)
    if torch.any(first_contact):
      first_contact_detected = True
      # Verify we're actually in contact.
      assert torch.all(sensor.data.found[first_contact] > 0)
      # Current contact time should be very small.
      assert torch.all(
        sensor._air_time_state.current_contact_time[first_contact] <= dt * 1.1
      )
      break

  assert first_contact_detected, "Should detect first contact at landing"

  # Test 2: First contact NOT triggered during sustained contact.
  # Place box on ground and let it settle.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.1
  root_state[:, 3] = 1.0
  box_entity.write_root_state_to_sim(root_state)

  sim.step()
  sensor.reset()

  for _ in range(20):
    sim.step()
    sensor.update(dt)

  # After sustained contact, first_contact should be False.
  first_contact = sensor.compute_first_contact(dt)
  assert torch.all(~first_contact), (
    "Should not detect first contact during sustained contact"
  )


def test_global_frame_transformation(falling_box_xml, device):
  """Test that force/torque are correctly transformed to global frame."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(falling_box_xml))

  # Create two sensors: one with global_frame=False, one with True.
  contact_cfg = ContactSensorCfg(
    name="box_contact_contact_frame",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force", "torque", "normal", "tangent"),
    global_frame=False,
  )

  contact_global_cfg = ContactSensorCfg(
    name="box_contact_global_frame",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force", "torque", "normal", "tangent"),
    global_frame=True,
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"box": entity_cfg},
    sensors=(contact_cfg, contact_global_cfg),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor_contact = scene["box_contact_contact_frame"]
  sensor_global = scene["box_contact_global_frame"]
  box_entity = scene["box"]

  # Place box on ground with some rotation to ensure non-trivial contact frame.
  root_state = torch.zeros((2, 13), device=sim.device)
  root_state[:, 2] = 0.15  # z position
  root_state[:, 3] = 0.9239  # qw (30 deg rotation)
  root_state[:, 4] = 0.3827  # qx
  box_entity.write_root_state_to_sim(root_state)

  for _ in range(30):
    sim.step()

  data_contact = sensor_contact.data
  data_global = sensor_global.data

  # Verify contact is detected.
  assert torch.any(data_contact.found > 0)
  assert torch.any(data_global.found > 0)

  # Build rotation matrix manually and verify transformation.
  normal = data_contact.normal  # [B, N, 3]
  tangent = data_contact.tangent  # [B, N, 3]
  tangent2 = torch.cross(normal, tangent, dim=-1)
  R = torch.stack([tangent, tangent2, normal], dim=-1)  # [B, N, 3, 3]

  # Transform force from contact frame to global frame.
  force_global_expected = torch.einsum("...ij,...j->...i", R, data_contact.force)
  torque_global_expected = torch.einsum("...ij,...j->...i", R, data_contact.torque)

  # Check that global_frame sensor matches manual transformation.
  # Only check where contact exists (normal is non-zero).
  has_contact = torch.norm(normal, dim=-1) > 1e-8
  assert torch.allclose(
    data_global.force[has_contact], force_global_expected[has_contact], atol=1e-5
  )
  assert torch.allclose(
    data_global.torque[has_contact], torque_global_expected[has_contact], atol=1e-5
  )

  # Verify that normal and tangent are unchanged (already in global frame).
  assert torch.allclose(data_contact.normal, data_global.normal)
  assert torch.allclose(data_contact.tangent, data_global.tangent)
