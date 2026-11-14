"""Tests for sensor caching behavior."""

from __future__ import annotations

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import (
  BuiltinSensorCfg,
  CameraSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
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
  if scene.sensor_context is not None:
    sim.set_sensor_context(scene.sensor_context)

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
  sim.sense()
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
  sim.sense()

  data2 = sensor.data

  # Now rays should hit floor ~4m away.
  assert torch.allclose(
    data2.distances, torch.full_like(data2.distances, 4.0), atol=0.1
  )

  # Data objects should be different (cache was invalidated).
  assert data1 is not data2


def test_sense_refreshes_cached_raycast_data(device):
  """sense() drops data cached before it, so all fields come from the new rays."""
  scene, sim = create_raycast_sensor_scene(device)
  sensor = scene["terrain_scan"]

  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  sim.sense()
  _ = sensor.data  # Populate the cache.

  root_state = torch.zeros((2, 13), device=device)
  root_state[:, 2] = 4.0
  root_state[:, 3] = 1.0
  scene["robot"].write_root_state_to_sim(root_state)
  sim.forward()
  sim.sense()

  data = sensor.data
  torch.testing.assert_close(data.distances, torch.full_like(data.distances, 4.0))
  torch.testing.assert_close(
    data.frame_pos_w[..., 2], torch.full((2, 1), 4.0, device=device)
  )


def _box_force(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["box_contact"].data.force.flatten(start_dim=1)


def test_observations_see_post_forward_contact_data(device):
  """A reward read before step()'s forward() must not leak into observations."""
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=2,
      env_spacing=3.0,
      entities={
        "box": EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(FALLING_BOX_XML))
      },
      sensors=(
        ContactSensorCfg(
          name="box_contact",
          primary=ContactMatch(mode="geom", pattern="box_geom", entity="box"),
          secondary=None,
          fields=("found", "force"),
        ),
      ),
    ),
    observations={
      "obs": ObservationGroupCfg(terms={"force": ObservationTermCfg(func=_box_force)})
    },
    rewards={"force": RewardTermCfg(func=lambda env: _box_force(env).sum(1), weight=1)},
    sim=SimulationCfg(njmax=75),
    decimation=4,
    episode_length_s=10.0,  # No resets, which would hide stale data.
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()

  for _ in range(60):  # The box lands after about 36 steps.
    obs = env.step(torch.zeros(2, 0, device=device))[0]["obs"]
    env.scene.invalidate_sensor_caches()
    torch.testing.assert_close(obs, _box_force(env))
  env.close()


def test_independent_camera_and_raycast_update_periods(device):
  """Raycast and camera sensors can run at independent update periods."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_WITH_FLOOR_XML)
  )
  cam_cfg = CameraSensorCfg(
    name="cam",
    pos=(0.0, 0.0, 5.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    width=16,
    height=12,
    data_types=("depth",),
    update_period=3,
  )
  ray_cfg = RayCastSensorCfg(
    name="scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.2),
    max_distance=10.0,
    update_period=2,
  )
  scene = Scene(
    SceneCfg(num_envs=1, entities={"robot": entity_cfg}, sensors=(cam_cfg, ray_cfg)),
    device,
  )
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(njmax=20), model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  sim.set_sensor_context(scene.sensor_context)

  def _set_height(z: float) -> None:
    root = torch.zeros((1, 13), device=device)
    root[:, 2] = z
    root[:, 3] = 1.0
    scene["robot"].write_root_state_to_sim(root)
    sim.forward()

  # Sense 0 (both due): z=2.0.
  _set_height(2.0)
  sim.sense()
  cam_d0 = scene["cam"].data.depth.clone()
  torch.testing.assert_close(
    scene["scan"].data.distances,
    torch.full_like(scene["scan"].data.distances, 2.0),
  )

  # Sense 1 (neither due): z=3.0, both hold z=2.0 readings.
  _set_height(3.0)
  sim.sense()
  torch.testing.assert_close(scene["cam"].data.depth, cam_d0)
  torch.testing.assert_close(
    scene["scan"].data.distances,
    torch.full_like(scene["scan"].data.distances, 2.0),
  )

  # Sense 2 (raycast due at period=2, camera still holds period=3): z=3.5.
  _set_height(3.5)
  sim.sense()
  torch.testing.assert_close(scene["cam"].data.depth, cam_d0)
  torch.testing.assert_close(
    scene["scan"].data.distances,
    torch.full_like(scene["scan"].data.distances, 3.5),
  )

  # Sense 3 (camera due at period=3, raycast holds 3.5): z=4.0.
  _set_height(4.0)
  sim.sense()
  assert not torch.allclose(scene["cam"].data.depth, cam_d0)
  torch.testing.assert_close(
    scene["scan"].data.distances,
    torch.full_like(scene["scan"].data.distances, 3.5),
  )


def test_env_periodic_sensors_hold_and_refresh_on_auto_reset(device):
  """Periodic context and builtin sensors hold across steps and refresh on reset."""
  reset_heights = [2.0]

  def _place_robot(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    n = env.num_envs if env_ids is None else len(env_ids)
    root = torch.zeros((n, 13), device=env.device)
    root[:, 2] = reset_heights[0]
    root[:, 3] = 1.0
    env.scene["robot"].write_root_state_to_sim(root, env_ids=ids)

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=2,
      env_spacing=5.0,
      entities={
        "robot": EntityCfg(
          spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_WITH_FLOOR_XML)
        )
      },
      sensors=(
        RayCastSensorCfg(
          name="scan",
          frame=ObjRef(type="body", name="base", entity="robot"),
          pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.2),
          max_distance=10.0,
          update_period=4,
        ),
        BuiltinSensorCfg(
          name="base_pos",
          sensor_type="framepos",
          obj=ObjRef(type="body", name="base", entity="robot"),
          update_period=4,
        ),
      ),
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "ray": ObservationTermCfg(
            func=lambda env: env.scene["scan"].data.distances,
          ),
          "z": ObservationTermCfg(
            func=lambda env: env.scene["robot/base_pos"].data[:, 2:3],
          ),
        },
        concatenate_terms=True,
      )
    },
    events={"reset_pose": EventTermCfg(func=_place_robot, mode="reset")},
    terminations={"time_out": TerminationTermCfg(func=time_out, time_out=True)},
    sim=SimulationCfg(njmax=20),
    decimation=4,
    # Two env steps per episode (2 * 4 * 0.002 = 0.016s), shorter than update_period=4.
    episode_length_s=0.016,
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  obs, _ = env.reset()
  actor_obs0 = obs["actor"]
  assert isinstance(actor_obs0, torch.Tensor)
  torch.testing.assert_close(actor_obs0, torch.full((2, 2), 2.0, device=device))

  # Step 1: robot falls under gravity, no reset yet, update_period=4 holds 2.0.
  action = torch.zeros((2, 0), device=device)
  obs1, _, term1, trunc1, _ = env.step(action)
  assert not (term1 | trunc1).any()
  actor_obs1 = obs1["actor"]
  assert isinstance(actor_obs1, torch.Tensor)
  torch.testing.assert_close(actor_obs1, torch.full((2, 2), 2.0, device=device))

  # Step 2: episode times out and auto-resets to z=3.5; both sensors must refresh
  # immediately even though only 2 steps elapsed (< update_period=4).
  reset_heights[0] = 3.5
  obs2, _, _, trunc2, _ = env.step(action)
  assert trunc2.all()
  actor_obs2 = obs2["actor"]
  assert isinstance(actor_obs2, torch.Tensor)
  torch.testing.assert_close(actor_obs2, torch.full((2, 2), 3.5, device=device))
  env.close()
