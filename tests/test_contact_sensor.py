"""Tests for contact sensors."""

from __future__ import annotations

import mujoco
import pytest
import torch

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ROBOT_CFG
from mjlab.entity import Entity
from mjlab.sensor.contact import (
  ContactSensor,
  ContactSensorCfg,
  SelfCollisionSensor,
  SelfCollisionSensorCfg,
)
from mjlab.sim.sim import Simulation, SimulationCfg


def get_device() -> str:
  return "cuda" if torch.cuda.is_available() else "cpu"


def build_scene_spec() -> tuple[mujoco.MjSpec, dict[str, Entity]]:
  robot = Entity(G1_ROBOT_CFG)
  scene_spec = mujoco.MjSpec()
  frame = scene_spec.worldbody.add_frame()
  scene_spec.attach(robot.spec, prefix="robot/", frame=frame)

  terrain_body = scene_spec.worldbody.add_body(name="terrain")
  terrain_body.add_geom(
    name="terrain_plane",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[10, 10, 0.1],
  )

  return scene_spec, {"robot": robot}


def initialize_contact_sensor(
  cfg: ContactSensorCfg,
  *,
  num_envs: int = 2,
) -> tuple[ContactSensor, Simulation, mujoco.MjModel]:
  spec, entities = build_scene_spec()
  sensor = ContactSensor(cfg)
  sensor.edit_spec(spec, entities)
  model = spec.compile()

  device = get_device()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  sensor.initialize(model, sim.model, sim.data, device)
  return sensor, sim, model


def test_contact_sensor_slot_registration():
  """Resolved slots should map to robot foot geoms in deterministic order."""
  cfg = ContactSensorCfg(
    name="feet_contact",
    entity_name="robot",
    geom1=r"(left|right)_foot1_collision",
    secondary_entity="",
    body2="terrain",
    data=("found", "force"),
    reduce="maxforce",
  )

  sensor, _, _ = initialize_contact_sensor(cfg)
  assert sensor.slot_names == ["left_foot1_collision", "right_foot1_collision"]
  assert sensor.slot_map == {
    "left_foot1_collision": 0,
    "right_foot1_collision": 1,
  }
  data = sensor.data.force
  assert data.shape == (2, 2, 3)


def test_contact_sensor_world_matching():
  """World-side matching should resolve to the terrain body without prefixes."""
  cfg = ContactSensorCfg(
    name="feet_vs_terrain",
    entity_name="robot",
    geom1=r"(left|right)_foot1_collision",
    secondary_entity="",
    body2="terrain",
    data=("found",),
    reduce="maxforce",
  )

  sensor, _, model = initialize_contact_sensor(cfg)
  terrain_id = model.body("terrain").id
  slot0 = model.sensor("feet_vs_terrain_slot0")

  assert slot0.reftype[0] == mujoco.mjtObj.mjOBJ_BODY
  assert slot0.refid[0] == terrain_id
  assert sensor.slot_names[0] == "left_foot1_collision"


def test_contact_sensor_air_time_updates():
  """Seed sensordata to exercise air-time transitions without stepping the sim."""
  cfg = ContactSensorCfg(
    name="air_time",
    entity_name="robot",
    geom1=r"(left|right)_foot1_collision",
    secondary_entity="",
    body2="terrain",
    data=("found", "force"),
    track_air_time=True,
    reduce="maxforce",
    force_threshold=0.1,
  )

  sensor, sim, model = initialize_contact_sensor(cfg)
  dt = 0.01

  raw = sim.data.sensordata
  raw.zero_()
  sensor.update(dt)

  device = sensor.current_air_time.device
  assert torch.allclose(sensor.current_air_time, torch.full((2, 2), dt, device=device))
  assert torch.equal(sensor.last_air_time, torch.zeros(2, 2, device=device))

  # Land on slot 0 of env 0, keep others in the air.
  slot0 = model.sensor("air_time_slot0")
  start, dim = slot0.adr[0], slot0.dim[0]
  raw.zero_()
  raw[0, start : start + dim] = raw.new_tensor([1.0, 5.0, 0.0, 0.0])
  sensor.update(dt)

  expected_last_air = torch.tensor(
    [[2 * dt, 0.0], [0.0, 0.0]], device=device, dtype=torch.float32
  )
  expected_current_air = torch.tensor(
    [[0.0, 2 * dt], [2 * dt, 2 * dt]], device=device, dtype=torch.float32
  )
  expected_contact_time = torch.tensor(
    [[dt, 0.0], [0.0, 0.0]], device=device, dtype=torch.float32
  )

  assert torch.allclose(sensor.last_air_time, expected_last_air)
  assert torch.allclose(sensor.current_air_time, expected_current_air)
  current_contact_time = sensor.data.current_contact_time
  assert current_contact_time is not None
  assert torch.allclose(current_contact_time, expected_contact_time)


def test_contact_sensor_reset_clears_state():
  """Resetting selected environments zeroes their air-time buffers."""
  cfg = ContactSensorCfg(
    name="reset_test",
    entity_name="robot",
    geom1=r"(left|right)_foot1_collision",
    secondary_entity="",
    body2="terrain",
    data=("found", "force"),
    track_air_time=True,
    reduce="maxforce",
  )

  sensor, sim, model = initialize_contact_sensor(cfg)
  dt = 0.02
  raw = sim.data.sensordata
  slot0 = model.sensor("reset_test_slot0")
  start, dim = slot0.adr[0], slot0.dim[0]
  device = sensor.current_air_time.device

  raw.zero_()
  sensor.update(dt)

  raw.zero_()
  raw[:, start : start + dim] = raw.new_tensor(
    [
      [1.0, 5.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0],
    ]
  )
  sensor.update(dt)
  sensor.reset(torch.tensor([0], device=device))

  assert torch.equal(sensor.current_air_time[0], torch.zeros(2, device=device))
  assert torch.equal(sensor.last_air_time[0], torch.zeros(2, device=device))
  # Environment 1 remains untouched.
  assert torch.all(sensor.current_air_time[1] > 0)


def test_contact_sensor_pattern_conflicts():
  """Conflicting or missing patterns should fail fast with a ValueError."""
  spec, entities = build_scene_spec()
  cfg = ContactSensorCfg(
    name="conflict",
    entity_name="robot",
    geom1=r"left_foot1_collision",
    body1=r"left_ankle_roll_link",
  )
  sensor = ContactSensor(cfg)

  with pytest.raises(ValueError, match="exactly one"):
    sensor.edit_spec(spec, entities)


def test_contact_sensor_multiple_secondary_matches():
  """Secondary pattern must resolve to a single target."""
  spec, entities = build_scene_spec()
  cfg = ContactSensorCfg(
    name="multi_secondary",
    entity_name="robot",
    geom1="left_foot1_collision",
    geom2=r"(left|right)_foot1_collision",
  )
  sensor = ContactSensor(cfg)

  with pytest.raises(ValueError, match="single match"):
    sensor.edit_spec(spec, entities)


def test_self_collision_sensor_initialization():
  """Self-collision sensor should wire up a single slot for the robot root."""
  spec, entities = build_scene_spec()
  cfg = SelfCollisionSensorCfg(name="self_collision", entity_name="robot")
  sensor = SelfCollisionSensor(cfg)
  sensor.edit_spec(spec, entities)

  model = spec.compile()
  device = get_device()
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  sensor.initialize(model, sim.model, sim.data, device)

  assert sensor.count.shape == (2,)

  missing_cfg = SelfCollisionSensorCfg(name="missing", entity_name="ghost")
  missing_sensor = SelfCollisionSensor(missing_cfg)
  with pytest.raises(ValueError, match="not found"):
    missing_sensor.edit_spec(spec, entities)
