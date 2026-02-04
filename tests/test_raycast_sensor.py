"""Tests for raycast_sensor.py."""

from __future__ import annotations

import math

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastData,
  RayCastSensorCfg,
)
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture(scope="module")
def robot_with_floor_xml():
  """XML for a floating body above a ground plane."""
  return """
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


@pytest.fixture(scope="module")
def scene_with_obstacles_xml():
  """XML for a body above various obstacles."""
  return """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <geom name="box1" type="box" size="0.5 0.5 0.5" pos="1 0 0.5"/>
        <geom name="box2" type="box" size="0.3 0.3 0.8" pos="-1 0 0.8"/>
        <body name="scanner" pos="0 0 3">
          <freejoint name="free_joint"/>
          <geom name="scanner_geom" type="sphere" size="0.1" mass="1.0"/>
          <site name="scan_site" pos="0 0 0"/>
        </body>
      </worldbody>
    </mujoco>
  """


def test_basic_raycast_hit_detection(robot_with_floor_xml, device):
  """Verify rays detect the ground plane and return correct distances."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_floor_xml)
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
    num_envs=2,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["terrain_scan"]
  sim.step()
  data = sensor.data

  assert isinstance(data, RayCastData)
  assert data.distances.shape[0] == 2  # num_envs
  assert data.distances.shape[1] == sensor.num_rays
  assert data.normals_w.shape == (2, sensor.num_rays, 3)

  # All rays should hit the floor (distance > 0).
  assert torch.all(data.distances >= 0)

  # Distance should be approximately 2m (body at z=2, floor at z=0).
  assert torch.allclose(data.distances, torch.full_like(data.distances, 2.0), atol=0.1)


def test_raycast_normals_point_up(robot_with_floor_xml, device):
  """Verify surface normals point upward when hitting a horizontal floor."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_floor_xml)
  )

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(
      size=(0.3, 0.3), resolution=0.15, direction=(0.0, 0.0, -1.0)
    ),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["terrain_scan"]
  sim.step()
  data = sensor.data

  # Normals should point up (+Z) for a horizontal floor.
  assert torch.allclose(
    data.normals_w[:, :, 2], torch.ones_like(data.normals_w[:, :, 2])
  )
  assert torch.allclose(
    data.normals_w[:, :, 0], torch.zeros_like(data.normals_w[:, :, 0])
  )
  assert torch.allclose(
    data.normals_w[:, :, 1], torch.zeros_like(data.normals_w[:, :, 1])
  )


def test_raycast_miss_returns_negative_one(device):
  """Verify rays that miss return distance of -1."""
  # Scene with no floor - rays will miss.
  no_floor_xml = """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 2">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="5.0"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(no_floor_xml))

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(
      size=(0.3, 0.3), resolution=0.15, direction=(0.0, 0.0, -1.0)
    ),
    max_distance=10.0,
    exclude_parent_body=True,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["terrain_scan"]
  sim.step()
  data = sensor.data

  # All rays should miss (distance = -1).
  assert torch.all(data.distances == -1)


def test_raycast_exclude_parent_body(robot_with_floor_xml, device):
  """Verify parent body is excluded from ray intersection when configured."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_floor_xml)
  )

  # With exclude_parent_body=True, rays should pass through the robot body.
  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.1, 0.1), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
    exclude_parent_body=True,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["terrain_scan"]
  sim.step()
  data = sensor.data

  # Rays should hit the floor, not the parent body geom.
  # Floor is at z=0, body is at z=2, so distance should be ~2m.
  assert torch.allclose(data.distances, torch.full_like(data.distances, 2.0), atol=0.1)


def test_raycast_include_geom_groups(device):
  """Verify include_geom_groups filters which geoms are hit."""
  # Scene with floor in group 0 and a box in group 1.
  groups_xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0" group="0"/>
        <geom name="platform" type="box" size="1 1 0.1" pos="0 0 1" group="1"/>
        <body name="base" pos="0 0 3">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(groups_xml))

  # Only include group 0 (floor) - should skip the platform in group 1.
  raycast_cfg = RayCastSensorCfg(
    name="group_filter_test",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
    include_geom_groups=(0,),  # Only hit floor, not platform
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["group_filter_test"]
  sim.step()
  data = sensor.data

  # Should hit floor at z=0, not platform at z=1.1. Distance from z=3 to z=0 is 3m.
  assert torch.allclose(data.distances, torch.full_like(data.distances, 3.0), atol=0.1)

  # Now test with group 1 included - should hit platform instead.
  raycast_cfg_group1 = RayCastSensorCfg(
    name="group1_test",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
    include_geom_groups=(1,),  # Only hit platform
  )

  scene_cfg2 = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg_group1,),
  )

  scene2 = Scene(scene_cfg2, device)
  model2 = scene2.compile()
  sim2 = Simulation(num_envs=1, cfg=sim_cfg, model=model2, device=device)
  scene2.initialize(sim2.mj_model, sim2.model, sim2.data)

  sensor2 = scene2["group1_test"]
  sim2.step()
  data2 = sensor2.data

  # Should hit platform at z=1.1. Distance from z=3 to z=1.1 is 1.9m.
  assert torch.allclose(
    data2.distances, torch.full_like(data2.distances, 1.9), atol=0.1
  )


def test_raycast_frame_attachment_geom(device):
  """Verify rays can be attached to a geom frame."""
  geom_xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 2">
          <freejoint name="free_joint"/>
          <geom name="sensor_mount" type="box" size="0.1 0.1 0.05" pos="0 0 -0.05"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(geom_xml))

  raycast_cfg = RayCastSensorCfg(
    name="geom_scan",
    frame=ObjRef(type="geom", name="sensor_mount", entity="robot"),
    pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["geom_scan"]
  sim.step()
  data = sensor.data

  assert isinstance(data, RayCastData)
  # Geom is at z=1.95 (body at z=2, geom offset -0.05), floor at z=0.
  assert torch.allclose(data.distances, torch.full_like(data.distances, 1.95), atol=0.1)


def test_raycast_frame_attachment_site(robot_with_floor_xml, device):
  """Verify rays can be attached to a site frame."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_floor_xml)
  )

  raycast_cfg = RayCastSensorCfg(
    name="site_scan",
    frame=ObjRef(type="site", name="base_site", entity="robot"),
    pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["site_scan"]
  sim.step()
  data = sensor.data

  assert isinstance(data, RayCastData)
  # Site is at z=1.9 (body at z=2, site offset -0.1), floor at z=0.
  assert torch.allclose(data.distances, torch.full_like(data.distances, 1.9), atol=0.1)


def test_raycast_grid_pattern_num_rays(device):
  """Verify grid pattern generates correct number of rays."""
  simple_xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1"/>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.1"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_xml))

  # Grid: size=(1.0, 0.5), resolution=0.5.
  # X: from -0.5 to 0.5 step 0.5 -> 3 points.
  # Y: from -0.25 to 0.25 step 0.5 -> 2 points.
  # Total: 3 * 2 = 6 rays.
  raycast_cfg = RayCastSensorCfg(
    name="grid_test",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(1.0, 0.5), resolution=0.5),
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["grid_test"]
  assert sensor.num_rays == 6


def test_raycast_different_direction(device):
  """Verify rays work with non-default direction."""
  wall_xml = """
    <mujoco>
      <worldbody>
        <geom name="wall" type="box" size="0.1 5 5" pos="2 0 2"/>
        <body name="base" pos="0 0 2">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.1"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(wall_xml))

  # Rays pointing in +X direction should hit wall at x=1.9.
  raycast_cfg = RayCastSensorCfg(
    name="forward_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1, direction=(1.0, 0.0, 0.0)),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["forward_scan"]
  sim.step()
  data = sensor.data

  # Wall is at x=1.9 (wall center at x=2, size 0.1), body at x=0.
  # Distance should be ~1.9m.
  assert torch.allclose(data.distances, torch.full_like(data.distances, 1.9), atol=0.1)

  # Normal should point in -X direction (toward the body).
  assert torch.allclose(
    data.normals_w[:, :, 0], -torch.ones_like(data.normals_w[:, :, 0]), atol=0.01
  )


def test_raycast_error_on_invalid_frame_type(device):
  """Verify ValueError is raised for invalid frame type."""
  with pytest.raises(ValueError, match="must be 'body', 'site', or 'geom'"):
    simple_xml = """
      <mujoco>
        <worldbody>
          <body name="base"><geom type="sphere" size="0.1"/></body>
        </worldbody>
      </mujoco>
    """
    entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_xml))

    raycast_cfg = RayCastSensorCfg(
      name="invalid",
      frame=ObjRef(type="joint", name="some_joint", entity="robot"),  # Invalid type
      pattern=GridPatternCfg(size=(0.1, 0.1), resolution=0.1),
    )

    scene_cfg = SceneCfg(
      num_envs=1,
      entities={"robot": entity_cfg},
      sensors=(raycast_cfg,),
    )

    scene = Scene(scene_cfg, device)
    model = scene.compile()
    sim_cfg = SimulationCfg(njmax=20)
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)


def test_raycast_hit_pos_w_correctness(robot_with_floor_xml, device):
  """Verify hit_pos_w returns correct world-space hit positions."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_floor_xml)
  )

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.4, 0.4), resolution=0.2, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["terrain_scan"]
  sim.step()
  data = sensor.data

  # All hit positions should be on the floor (z=0).
  assert torch.allclose(
    data.hit_pos_w[:, :, 2], torch.zeros_like(data.hit_pos_w[:, :, 2]), atol=0.01
  )

  # Hit positions X and Y should match the ray grid pattern offset from body origin.
  # Body is at (0, 0, 2), grid is 0.4x0.4 with 0.2 resolution = 3x3 = 9 rays.
  # X positions should be in range [-0.2, 0.2], Y positions in range [-0.2, 0.2].
  assert torch.all(data.hit_pos_w[:, :, 0] >= -0.3)
  assert torch.all(data.hit_pos_w[:, :, 0] <= 0.3)
  assert torch.all(data.hit_pos_w[:, :, 1] >= -0.3)
  assert torch.all(data.hit_pos_w[:, :, 1] <= 0.3)


def test_raycast_max_distance_clamping(device):
  """Verify hits beyond max_distance are reported as misses."""
  far_floor_xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 5">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(far_floor_xml))

  # max_distance=3.0, but floor is 5m away. Should report miss.
  raycast_cfg = RayCastSensorCfg(
    name="short_range",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=3.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["short_range"]
  sim.step()
  data = sensor.data

  # All rays should miss (floor is beyond max_distance).
  assert torch.all(data.distances == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_raycast_body_rotation_affects_rays(device):
  """Verify rays rotate with the body frame."""
  rotated_body_xml = """
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 2">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(rotated_body_xml))

  # Rays point in -Z in body frame. We'll tilt body 45 degrees around X.
  # So rays should point diagonally down, hitting floor at a longer distance.
  raycast_cfg = RayCastSensorCfg(
    name="rotated_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["rotated_scan"]

  # First, verify baseline: unrotated body, rays hit floor at ~2m.
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  data_unrotated = sensor.data
  assert torch.allclose(
    data_unrotated.distances, torch.full_like(data_unrotated.distances, 2.0), atol=0.1
  )

  # Now tilt body 45 degrees around X axis.
  # Ray direction -Z in body frame becomes diagonal in world frame.
  # Distance to floor should be 2 / cos(45) = 2 * sqrt(2) ≈ 2.83m.
  angle = math.pi / 4
  quat = [math.cos(angle / 2), math.sin(angle / 2), 0, 0]  # w, x, y, z
  sim.data.qpos[0, 3:7] = torch.tensor(quat, device=device)
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  data_rotated = sensor.data

  expected_distance = 2.0 / math.cos(angle)  # ~2.83m
  assert torch.allclose(
    data_rotated.distances,
    torch.full_like(data_rotated.distances, expected_distance),
    atol=0.15,
  ), f"Expected ~{expected_distance:.2f}m, got {data_rotated.distances}"


# ============================================================================
# Pinhole Camera Pattern Tests
# ============================================================================


def test_pinhole_camera_pattern_num_rays(device):
  """Verify pinhole pattern generates width * height rays."""
  simple_xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1"/>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.1"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(simple_xml))

  raycast_cfg = RayCastSensorCfg(
    name="camera_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=PinholeCameraPatternCfg(width=16, height=12, fovy=74.0),
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["camera_scan"]
  assert sensor.num_rays == 16 * 12


def test_pinhole_camera_fov(robot_with_floor_xml, device):
  """Verify pinhole pattern ray angles match FOV."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_floor_xml)
  )

  # 90 degree vertical FOV.
  raycast_cfg = RayCastSensorCfg(
    name="camera_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=PinholeCameraPatternCfg(width=3, height=3, fovy=90.0),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["camera_scan"]
  assert sensor.num_rays == 9


def test_pinhole_from_intrinsic_matrix():
  """Verify from_intrinsic_matrix creates correct config."""
  # Intrinsic matrix with fx=500, fy=500, cx=320, cy=240.
  intrinsic = [500.0, 0, 320, 0, 500.0, 240, 0, 0, 1]
  width, height = 640, 480

  cfg = PinholeCameraPatternCfg.from_intrinsic_matrix(intrinsic, width, height)

  # Expected vertical FOV: 2 * atan(480 / (2 * 500)) = 2 * atan(0.48) ≈ 51.3 degrees.
  fy = intrinsic[4]
  expected_fov = 2 * math.atan(height / (2 * fy)) * 180 / math.pi
  assert abs(cfg.fovy - expected_fov) < 0.1
  assert cfg.width == width
  assert cfg.height == height


def test_pinhole_from_mujoco_camera(device):
  """Verify pinhole pattern can be created from MuJoCo camera."""
  # XML with a camera that has explicit resolution, sensorsize, and focal.
  camera_xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 2">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.1" mass="1.0"/>
          <camera name="depth_cam" pos="0 0 0" resolution="64 48"
                  sensorsize="0.00389 0.00292" focal="0.00193 0.00193"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(camera_xml))

  # Use from_mujoco_camera() to get params from MuJoCo camera.
  raycast_cfg = RayCastSensorCfg(
    name="camera_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=PinholeCameraPatternCfg.from_mujoco_camera("robot/depth_cam"),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["camera_scan"]
  # Should have 64 * 48 = 3072 rays.
  assert sensor.num_rays == 64 * 48

  # Verify rays work.
  sim.step()
  data = sensor.data
  assert torch.all(data.distances >= 0)  # Should hit floor


def test_pinhole_from_mujoco_camera_fovy_mode(device):
  """Verify pinhole pattern works with MuJoCo camera using fovy (not sensorsize)."""
  # XML with a camera using fovy mode (no sensorsize/focal).
  camera_xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 2">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.1" mass="1.0"/>
          <camera name="fovy_cam" pos="0 0 0" fovy="60" resolution="32 24"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(camera_xml))

  raycast_cfg = RayCastSensorCfg(
    name="camera_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=PinholeCameraPatternCfg.from_mujoco_camera("robot/fovy_cam"),
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["camera_scan"]
  # Should have 32 * 24 = 768 rays.
  assert sensor.num_rays == 32 * 24

  # Verify rays work.
  sim.step()
  data = sensor.data
  assert torch.all(data.distances >= 0)  # Should hit floor


# ============================================================================
# Ray Alignment Tests
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_ray_alignment_yaw(device):
  """Verify yaw alignment ignores pitch/roll."""
  rotated_body_xml = """
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 2">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(rotated_body_xml))

  # With yaw alignment, tilting the body should NOT affect ray direction.
  raycast_cfg = RayCastSensorCfg(
    name="yaw_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    ray_alignment="yaw",
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["yaw_scan"]

  # Baseline: unrotated.
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  data_unrotated = sensor.data
  baseline_dist = data_unrotated.distances.clone()

  # Tilt body 45 degrees around X axis.
  angle = math.pi / 4
  quat = [math.cos(angle / 2), math.sin(angle / 2), 0, 0]  # w, x, y, z
  sim.data.qpos[0, 3:7] = torch.tensor(quat, device=device)
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  data_tilted = sensor.data

  # With yaw alignment, distance should remain ~2m (not change due to tilt).
  assert torch.allclose(data_tilted.distances, baseline_dist, atol=0.1), (
    f"Expected ~2m, got {data_tilted.distances}"
  )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_ray_alignment_world(device):
  """Verify world alignment keeps rays fixed."""
  rotated_body_xml = """
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 2">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(rotated_body_xml))

  # With world alignment, rotating body should NOT affect ray direction.
  raycast_cfg = RayCastSensorCfg(
    name="world_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    ray_alignment="world",
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["world_scan"]

  # Baseline: unrotated.
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  data_unrotated = sensor.data
  baseline_dist = data_unrotated.distances.clone()

  # Rotate body 90 degrees around Z (yaw), then tilt 45 degrees around X.
  # With world alignment, distance should still be ~2m.
  yaw_angle = math.pi / 2
  pitch_angle = math.pi / 4
  # Compose quaternions: yaw then pitch.
  cy, sy = math.cos(yaw_angle / 2), math.sin(yaw_angle / 2)
  cp, sp = math.cos(pitch_angle / 2), math.sin(pitch_angle / 2)
  # q_yaw = [cy, 0, 0, sy], q_pitch = [cp, sp, 0, 0]
  # q = q_pitch * q_yaw
  qw = cp * cy
  qx = sp * cy
  qy = sp * sy
  qz = cp * sy
  sim.data.qpos[0, 3:7] = torch.tensor([qw, qx, qy, qz], device=device)
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  data_rotated = sensor.data

  # With world alignment, distance should remain ~2m.
  assert torch.allclose(data_rotated.distances, baseline_dist, atol=0.1), (
    f"Expected ~2m, got {data_rotated.distances}"
  )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_ray_alignment_yaw_singularity(device):
  """Test yaw alignment handles 90° pitch singularity correctly.

  With yaw alignment, rays should maintain their pattern regardless of body pitch.
  At 90° pitch, the body's X-axis is vertical, making yaw extraction ambiguous.
  The implementation uses Y-axis fallback to produce a valid yaw rotation.

  This test verifies that distances at 90° pitch match the baseline (0° pitch).
  """
  xml = """
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 2">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
  """

  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))

  # Use grid pattern with diagonal direction - has X component to expose singularity.
  # Direction [1, 0, -1] points forward and down at 45°.
  raycast_cfg = RayCastSensorCfg(
    name="yaw_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(1.0, 0.0, -1.0)),
    ray_alignment="yaw",
    max_distance=10.0,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
    sensors=(raycast_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["yaw_scan"]

  # Baseline: no rotation. Ray at 45° from height 2m hits floor at x=2, z=0.
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  baseline_hit_pos = sensor.data.hit_pos_w.clone()
  # Ray goes diagonally +X and -Z, starting from (0,0,2), should hit floor at (2, 0, 0).
  assert torch.allclose(
    baseline_hit_pos[0, 0, 0], torch.tensor(2.0, device=device), atol=0.1
  ), f"Baseline X hit should be ~2, got {baseline_hit_pos[0, 0, 0]}"
  assert torch.allclose(
    baseline_hit_pos[0, 0, 2], torch.tensor(0.0, device=device), atol=0.1
  ), f"Baseline Z hit should be ~0, got {baseline_hit_pos[0, 0, 2]}"

  # Pitch 90° around Y-axis. Body X-axis now points straight down (singularity).
  angle = math.pi / 2
  quat = [math.cos(angle / 2), 0, math.sin(angle / 2), 0]  # w, x, y, z
  sim.data.qpos[0, 3:7] = torch.tensor(quat, device=device)
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)

  singularity_hit_pos = sensor.data.hit_pos_w

  # With yaw alignment, hit position should match baseline regardless of pitch.
  # The ray should still go diagonally and hit at (2, 0, 0).
  assert torch.allclose(singularity_hit_pos, baseline_hit_pos, atol=0.1), (
    f"Yaw alignment failed at 90° pitch singularity.\n"
    f"Baseline hit_pos: {baseline_hit_pos}\n"
    f"Singularity hit_pos: {singularity_hit_pos}"
  )
