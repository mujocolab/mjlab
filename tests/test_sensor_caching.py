"""Tests for sensor caching behavior."""

from __future__ import annotations

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


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


ROBOT_WITH_FLOOR_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
    <body name="base" pos="0 0 2">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="5.0"/>
      <site name="base_site" pos="0 0 -0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def create_contact_sensor_scene(device: str, num_envs: int = 2):
  """Create a scene with a contact sensor."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(FALLING_BOX_XML))

  contact_sensor_cfg = ContactSensorCfg(
    name="box_contact",
    primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
    secondary=None,
    fields=("found", "force"),
  )

  scene_cfg = SceneCfg(
    num_envs=num_envs,
    env_spacing=3.0,
    entities={"box": entity_cfg},
    sensors=(contact_sensor_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=75)
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  return scene, sim


def create_raycast_sensor_scene(device: str, num_envs: int = 2):
  """Create a scene with a raycast sensor."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_WITH_FLOOR_XML)
  )

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(
      size=(0.5, 0.5), resolution=0.25, direction=(0.0, 0.0, -1.0)
    ),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=num_envs,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  return scene, sim


def test_cache_invalidated_by_update(device):
  """Verify that update() invalidates the cache."""
  scene, sim = create_contact_sensor_scene(device)
  sensor = scene["box_contact"]

  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)

  # Access data to populate cache.
  data1 = sensor.data

  # Call update to invalidate.
  sensor.update(dt=sim.cfg.mujoco.timestep)

  # Access data again - should be a new object.
  data2 = sensor.data

  # Should be different objects (cache was invalidated and recomputed).
  assert data1 is not data2


def test_cache_invalidated_by_reset(device):
  """Verify that reset() invalidates the cache."""
  scene, sim = create_contact_sensor_scene(device)
  sensor = scene["box_contact"]

  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)

  # Access data to populate cache.
  data1 = sensor.data

  # Call reset to invalidate.
  sensor.reset(None)

  # Access data again - should be a new object.
  data2 = sensor.data

  # Should be different objects (cache was invalidated and recomputed).
  assert data1 is not data2


def test_compute_data_called_once_per_cache_period(device):
  """Verify _compute_data is only called once when cache is valid."""
  scene, sim = create_contact_sensor_scene(device)
  sensor = scene["box_contact"]

  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)

  # Patch _compute_data to count calls.
  original_compute = sensor._compute_data
  call_count = [0]

  def counting_compute():
    call_count[0] += 1
    return original_compute()

  sensor._compute_data = counting_compute

  # Access data 5 times without invalidating cache.
  for _ in range(5):
    _ = sensor.data

  # _compute_data should have been called only once.
  assert call_count[0] == 1


def test_data_reflects_physics_after_step_and_update(device):
  """Verify sensor data reflects physics state after step + update cycle."""
  scene, sim = create_raycast_sensor_scene(device)
  sensor = scene["terrain_scan"]
  robot_entity = scene["robot"]

  # Initial position at z=2.
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  data1 = sensor.data

  # All rays should hit floor ~2m away.
  assert torch.allclose(
    data1.distances, torch.full_like(data1.distances, 2.0), atol=0.1
  )

  # Move robot higher.
  root_state = torch.zeros((2, 13), device=device)
  root_state[:, 2] = 4.0  # Move to z=4.
  root_state[:, 3] = 1.0  # Unit quaternion.
  robot_entity.write_root_state_to_sim(root_state)

  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)

  data2 = sensor.data

  # Now rays should hit floor ~4m away.
  assert torch.allclose(
    data2.distances, torch.full_like(data2.distances, 4.0), atol=0.1
  )

  # Data objects should be different (cache was invalidated).
  assert data1 is not data2


def test_stale_cache_without_update(device):
  """Verify that without update(), cache returns stale data."""
  scene, sim = create_raycast_sensor_scene(device)
  sensor = scene["terrain_scan"]
  robot_entity = scene["robot"]

  # Initial step + update.
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)

  # Access data to populate cache.
  data1 = sensor.data
  initial_distances = data1.distances.clone()

  # Move robot higher.
  root_state = torch.zeros((2, 13), device=device)
  root_state[:, 2] = 4.0  # Move to z=4.
  root_state[:, 3] = 1.0  # Unit quaternion.
  robot_entity.write_root_state_to_sim(root_state)

  # Step but DON'T call update - cache should NOT be invalidated.
  sim.step()

  # Access data - should still be cached (stale).
  data2 = sensor.data

  # Data should be the same object (cache was not invalidated).
  assert data1 is data2

  # Values should be the same (stale).
  assert torch.allclose(data2.distances, initial_distances)
