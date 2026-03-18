"""Tests for raycast_sensor.py."""

from __future__ import annotations

import math

import pytest
import torch
from conftest import get_test_device, make_scene_and_sim

from mjlab.envs.mdp.observations import height_scan
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastData,
  RayCastSensorCfg,
  RingPatternCfg,
)


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def test_basic_raycast_hit_detection(robot_with_floor_xml, device):
  """Verify rays detect the ground plane and return correct distances."""
  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(
      size=(0.5, 0.5), resolution=0.25, direction=(0.0, 0.0, -1.0)
    ),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(
    device, robot_with_floor_xml, sensors=(raycast_cfg,), num_envs=2
  )

  sensor = scene["terrain_scan"]
  sim.step()
  sim.sense()
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
  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(
      size=(0.3, 0.3), resolution=0.15, direction=(0.0, 0.0, -1.0)
    ),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, robot_with_floor_xml, sensors=(raycast_cfg,))

  sensor = scene["terrain_scan"]
  sim.step()
  sim.sense()
  data = sensor.data

  # Normals should point up (+Z) for a horizontal floor.
  expected = torch.zeros_like(data.normals_w)
  expected[:, :, 2] = 1.0
  assert torch.allclose(data.normals_w, expected)


def test_raycast_miss_returns_negative_one(device):
  """Verify rays that miss return distance of -1."""
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

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(
      size=(0.3, 0.3), resolution=0.15, direction=(0.0, 0.0, -1.0)
    ),
    max_distance=10.0,
    exclude_parent_body=True,
  )

  scene, sim = make_scene_and_sim(device, no_floor_xml, sensors=(raycast_cfg,))

  sensor = scene["terrain_scan"]
  sim.step()
  sim.sense()
  data = sensor.data

  # All rays should miss (distance = -1).
  assert torch.all(data.distances == -1)


def test_raycast_exclude_parent_body(robot_with_floor_xml, device):
  """Verify parent body is excluded from ray intersection when configured."""
  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.1, 0.1), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
    exclude_parent_body=True,
  )

  scene, sim = make_scene_and_sim(device, robot_with_floor_xml, sensors=(raycast_cfg,))

  sensor = scene["terrain_scan"]
  sim.step()
  sim.sense()
  data = sensor.data

  # Rays should hit the floor, not the parent body geom.
  # Floor is at z=0, body is at z=2, so distance should be ~2m.
  assert torch.allclose(data.distances, torch.full_like(data.distances, 2.0), atol=0.1)


def test_raycast_include_geom_groups(device):
  """Verify include_geom_groups filters which geoms are hit."""
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

  # Only include group 0 (floor) - should skip the platform in group 1.
  raycast_cfg = RayCastSensorCfg(
    name="group_filter_test",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
    include_geom_groups=(0,),
  )

  scene, sim = make_scene_and_sim(device, groups_xml, sensors=(raycast_cfg,))

  sensor = scene["group_filter_test"]
  sim.step()
  sim.sense()
  data = sensor.data

  # Should hit floor at z=0, not platform at z=1.1. Distance from z=3 to z=0 is 3m.
  assert torch.allclose(data.distances, torch.full_like(data.distances, 3.0), atol=0.1)

  # Now test with group 1 included - should hit platform instead.
  raycast_cfg_group1 = RayCastSensorCfg(
    name="group1_test",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
    include_geom_groups=(1,),
  )

  scene2, sim2 = make_scene_and_sim(device, groups_xml, sensors=(raycast_cfg_group1,))

  sensor2 = scene2["group1_test"]
  sim2.step()
  sim2.sense()
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

  raycast_cfg = RayCastSensorCfg(
    name="geom_scan",
    frame=ObjRef(type="geom", name="sensor_mount", entity="robot"),
    pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, geom_xml, sensors=(raycast_cfg,))

  sensor = scene["geom_scan"]
  sim.step()
  sim.sense()
  data = sensor.data

  assert isinstance(data, RayCastData)
  # Geom is at z=1.95 (body at z=2, geom offset -0.05), floor at z=0.
  assert torch.allclose(data.distances, torch.full_like(data.distances, 1.95), atol=0.1)


def test_raycast_frame_attachment_site(robot_with_floor_xml, device):
  """Verify rays can be attached to a site frame."""
  raycast_cfg = RayCastSensorCfg(
    name="site_scan",
    frame=ObjRef(type="site", name="base_site", entity="robot"),
    pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, robot_with_floor_xml, sensors=(raycast_cfg,))

  sensor = scene["site_scan"]
  sim.step()
  sim.sense()
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

  # Grid: size=(1.0, 0.5), resolution=0.5.
  # X: from -0.5 to 0.5 step 0.5 -> 3 points.
  # Y: from -0.25 to 0.25 step 0.5 -> 2 points.
  # Total: 3 * 2 = 6 rays.
  raycast_cfg = RayCastSensorCfg(
    name="grid_test",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(1.0, 0.5), resolution=0.5),
  )

  scene, sim = make_scene_and_sim(device, simple_xml, sensors=(raycast_cfg,))

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

  raycast_cfg = RayCastSensorCfg(
    name="forward_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1, direction=(1.0, 0.0, 0.0)),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, wall_xml, sensors=(raycast_cfg,))

  sensor = scene["forward_scan"]
  sim.step()
  sim.sense()
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
  simple_xml = """
    <mujoco>
      <worldbody>
        <body name="base"><geom type="sphere" size="0.1"/></body>
      </worldbody>
    </mujoco>
  """
  raycast_cfg = RayCastSensorCfg(
    name="invalid",
    frame=ObjRef(type="joint", name="some_joint", entity="robot"),
    pattern=GridPatternCfg(size=(0.1, 0.1), resolution=0.1),
  )
  with pytest.raises(ValueError, match="must be 'body', 'site', or 'geom'"):
    make_scene_and_sim(device, simple_xml, sensors=(raycast_cfg,))


def test_raycast_hit_pos_w_correctness(robot_with_floor_xml, device):
  """Verify hit_pos_w returns correct world-space hit positions."""
  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.4, 0.4), resolution=0.2, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, robot_with_floor_xml, sensors=(raycast_cfg,))

  sensor = scene["terrain_scan"]
  sim.step()
  sim.sense()
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

  raycast_cfg = RayCastSensorCfg(
    name="short_range",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=3.0,
  )

  scene, sim = make_scene_and_sim(device, far_floor_xml, sensors=(raycast_cfg,))

  sensor = scene["short_range"]
  sim.step()
  sim.sense()
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

  raycast_cfg = RayCastSensorCfg(
    name="rotated_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, rotated_body_xml, sensors=(raycast_cfg,))

  sensor = scene["rotated_scan"]

  # First, verify baseline: unrotated body, rays hit floor at ~2m.
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  sim.sense()
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
  sim.sense()
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

  raycast_cfg = RayCastSensorCfg(
    name="camera_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=PinholeCameraPatternCfg(width=16, height=12, fovy=74.0),
  )

  scene, sim = make_scene_and_sim(device, simple_xml, sensors=(raycast_cfg,))

  sensor = scene["camera_scan"]
  assert sensor.num_rays == 16 * 12


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

  # Use from_mujoco_camera() to get params from MuJoCo camera.
  raycast_cfg = RayCastSensorCfg(
    name="camera_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=PinholeCameraPatternCfg.from_mujoco_camera("robot/depth_cam"),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, camera_xml, sensors=(raycast_cfg,))

  sensor = scene["camera_scan"]
  # Should have 64 * 48 = 3072 rays.
  assert sensor.num_rays == 64 * 48

  # Verify rays work.
  sim.step()
  sim.sense()
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

  raycast_cfg = RayCastSensorCfg(
    name="camera_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=PinholeCameraPatternCfg.from_mujoco_camera("robot/fovy_cam"),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, camera_xml, sensors=(raycast_cfg,))

  sensor = scene["camera_scan"]
  # Should have 32 * 24 = 768 rays.
  assert sensor.num_rays == 32 * 24

  # Verify rays work.
  sim.step()
  sim.sense()
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

  # With yaw alignment, tilting the body should NOT affect ray direction.
  raycast_cfg = RayCastSensorCfg(
    name="yaw_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    ray_alignment="yaw",
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, rotated_body_xml, sensors=(raycast_cfg,))

  sensor = scene["yaw_scan"]

  # Baseline: unrotated.
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  sim.sense()
  data_unrotated = sensor.data
  baseline_dist = data_unrotated.distances.clone()

  # Tilt body 45 degrees around X axis.
  angle = math.pi / 4
  quat = [math.cos(angle / 2), math.sin(angle / 2), 0, 0]  # w, x, y, z
  sim.data.qpos[0, 3:7] = torch.tensor(quat, device=device)
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  sim.sense()
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

  # With world alignment, rotating body should NOT affect ray direction.
  raycast_cfg = RayCastSensorCfg(
    name="world_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(0.0, 0.0, -1.0)),
    ray_alignment="world",
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, rotated_body_xml, sensors=(raycast_cfg,))

  sensor = scene["world_scan"]

  # Baseline: unrotated.
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  sim.sense()
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
  sim.sense()
  data_rotated = sensor.data

  # With world alignment, distance should remain ~2m.
  assert torch.allclose(data_rotated.distances, baseline_dist, atol=0.1), (
    f"Expected ~2m, got {data_rotated.distances}"
  )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_ray_alignment_yaw_singularity(device):
  """Test yaw alignment handles 90 degree pitch singularity correctly.

  With yaw alignment, rays should maintain their pattern regardless of
  body pitch. At 90 degree pitch, the body's X-axis is vertical, making
  yaw extraction ambiguous. The implementation uses Y-axis fallback to
  produce a valid yaw rotation.

  This test verifies that distances at 90 degree pitch match the
  baseline (0 degree pitch).
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

  # Use grid pattern with diagonal direction - has X component to
  # expose singularity. Direction [1, 0, -1] points forward and down
  # at 45 degrees.
  raycast_cfg = RayCastSensorCfg(
    name="yaw_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1, direction=(1.0, 0.0, -1.0)),
    ray_alignment="yaw",
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, xml, sensors=(raycast_cfg,))

  sensor = scene["yaw_scan"]

  # Baseline: no rotation. Ray at 45 degrees from height 2m hits floor
  # at x=2, z=0.
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  sim.sense()
  baseline_hit_pos = sensor.data.hit_pos_w.clone()
  # Ray goes diagonally +X and -Z, starting from (0,0,2), should hit
  # floor at (2, 0, 0).
  assert torch.allclose(
    baseline_hit_pos[0, 0, 0],
    torch.tensor(2.0, device=device),
    atol=0.1,
  ), f"Baseline X hit should be ~2, got {baseline_hit_pos[0, 0, 0]}"
  assert torch.allclose(
    baseline_hit_pos[0, 0, 2],
    torch.tensor(0.0, device=device),
    atol=0.1,
  ), f"Baseline Z hit should be ~0, got {baseline_hit_pos[0, 0, 2]}"

  # Pitch 90 degrees around Y-axis. Body X-axis now points straight
  # down (singularity).
  angle = math.pi / 2
  quat = [math.cos(angle / 2), 0, math.sin(angle / 2), 0]  # w, x, y, z
  sim.data.qpos[0, 3:7] = torch.tensor(quat, device=device)
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  sim.sense()

  singularity_hit_pos = sensor.data.hit_pos_w

  # With yaw alignment, hit position should match baseline regardless
  # of pitch. The ray should still go diagonally and hit at (2, 0, 0).
  assert torch.allclose(singularity_hit_pos, baseline_hit_pos, atol=0.1), (
    f"Yaw alignment failed at 90 degree pitch singularity.\n"
    f"Baseline hit_pos: {baseline_hit_pos}\n"
    f"Singularity hit_pos: {singularity_hit_pos}"
  )


class _FakeEnv:
  """Minimal env-like object for testing observation functions."""

  def __init__(self, scene):
    self.scene = scene


def test_height_scan_hits(robot_with_floor_xml, device):
  """height_scan returns correct heights for hits."""

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(
      size=(0.3, 0.3),
      resolution=0.15,
      direction=(0.0, 0.0, -1.0),
    ),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(
    device, robot_with_floor_xml, sensors=(raycast_cfg,), num_envs=2
  )

  sim.step()
  sim.sense()

  env = _FakeEnv(scene)
  heights = height_scan(env, "terrain_scan")  # type: ignore[invalid-argument-type]

  sensor = scene["terrain_scan"]
  # Shape: [num_envs, num_rays].
  assert heights.shape == (2, sensor.num_rays)
  # Body at z=2, floor at z=0 → height ≈ 2.0.
  assert torch.allclose(heights, torch.full_like(heights, 2.0), atol=0.1)


def test_height_scan_misses(device):
  """height_scan reports max_distance for rays that miss (no ground)."""

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

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(
      size=(0.2, 0.2),
      resolution=0.1,
      direction=(0.0, 0.0, -1.0),
    ),
    max_distance=10.0,
    exclude_parent_body=True,
  )

  scene, sim = make_scene_and_sim(device, no_floor_xml, sensors=(raycast_cfg,))

  sim.step()
  sim.sense()

  env = _FakeEnv(scene)
  heights = height_scan(env, "terrain_scan")  # type: ignore[invalid-argument-type]

  # Misses default to sensor max_distance.
  assert torch.allclose(
    heights, torch.full_like(heights, raycast_cfg.max_distance), atol=1e-5
  )


# ============================================================================
# Multi-Frame Tests
# ============================================================================

MULTI_SITE_XML = """
  <mujoco>
    <worldbody>
      <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
      <body name="base" pos="0 0 3">
        <freejoint name="free_joint"/>
        <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="5.0"/>
        <site name="site_top" pos="0 0 0.5"/>
        <site name="site_mid" pos="0 0 0"/>
        <site name="site_bot" pos="0 0 -0.5"/>
      </body>
    </worldbody>
  </mujoco>
"""


def test_multi_frame_shapes(device):
  """Multi-frame sensor produces correct output shapes."""
  cfg = RayCastSensorCfg(
    name="multi",
    frame=(
      ObjRef(type="site", name="site_top", entity="robot"),
      ObjRef(type="site", name="site_mid", entity="robot"),
      ObjRef(type="site", name="site_bot", entity="robot"),
    ),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, MULTI_SITE_XML, (cfg,), num_envs=2)
  sim.step()
  sim.sense()

  sensor = scene["multi"]
  data = sensor.data

  assert sensor.num_frames == 3
  assert sensor.num_rays_per_frame == 1
  assert sensor.num_rays == 3

  assert data.distances.shape == (2, 3)
  assert data.frame_pos_w.shape == (2, 3, 3)
  assert data.frame_quat_w.shape == (2, 3, 4)
  # Backward compat: pos_w/quat_w equal first frame.
  assert data.pos_w.shape == (2, 3)
  assert data.quat_w.shape == (2, 4)
  assert torch.allclose(data.pos_w, data.frame_pos_w[:, 0])
  assert torch.allclose(data.quat_w, data.frame_quat_w[:, 0])


def test_multi_frame_heights(device):
  """Sites at different Z offsets produce proportional distances."""
  cfg = RayCastSensorCfg(
    name="multi",
    frame=(
      ObjRef(type="site", name="site_top", entity="robot"),
      ObjRef(type="site", name="site_mid", entity="robot"),
      ObjRef(type="site", name="site_bot", entity="robot"),
    ),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, MULTI_SITE_XML, (cfg,))
  sim.step()
  sim.sense()

  distances = scene["multi"].data.distances[0]
  # site_top at z=3.5, site_mid at z=3, site_bot at z=2.5.
  assert distances[0].item() == pytest.approx(3.5, abs=0.1)
  assert distances[1].item() == pytest.approx(3.0, abs=0.1)
  assert distances[2].item() == pytest.approx(2.5, abs=0.1)


def test_multi_frame_body_exclusion(device):
  """Each frame excludes only its own parent body, not others.

  body_a has a platform geom directly below site_b. Frame B's rays
  should skip body_b's own geom but HIT body_a's platform. Frame A's
  rays should skip body_a and hit the floor.
  """
  xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <body name="body_a" pos="0 0 1">
          <freejoint name="free_a"/>
          <geom name="geom_a" type="box" size="2 2 0.1" mass="5.0"/>
          <site name="site_a" pos="0 0 0"/>
        </body>
        <body name="body_b" pos="0 0 3">
          <freejoint name="free_b"/>
          <geom name="geom_b" type="box" size="0.5 0.5 0.5" mass="5.0"/>
          <site name="site_b" pos="0 0 0"/>
        </body>
      </worldbody>
    </mujoco>
  """

  cfg = RayCastSensorCfg(
    name="multi",
    frame=(
      ObjRef(type="site", name="site_a", entity="robot"),
      ObjRef(type="site", name="site_b", entity="robot"),
    ),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),
    max_distance=10.0,
    exclude_parent_body=True,
  )

  scene, sim = make_scene_and_sim(device, xml, (cfg,))
  sim.step()
  sim.sense()

  distances = scene["multi"].data.distances[0]
  # Frame A (site_a at z=1): excludes body_a, hits floor at z=0 -> dist ~1.0.
  assert distances[0].item() == pytest.approx(1.0, abs=0.15)
  # Frame B (site_b at z=3): excludes body_b, hits body_a's platform
  # at z=1.1 -> dist ~1.9.
  assert distances[1].item() == pytest.approx(1.9, abs=0.15)


def test_multi_frame_height_scan(device):
  """height_scan works correctly with multi-frame sensors."""
  cfg = RayCastSensorCfg(
    name="multi",
    frame=(
      ObjRef(type="site", name="site_top", entity="robot"),
      ObjRef(type="site", name="site_mid", entity="robot"),
      ObjRef(type="site", name="site_bot", entity="robot"),
    ),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, MULTI_SITE_XML, (cfg,), num_envs=2)
  sim.step()
  sim.sense()

  env = _FakeEnv(scene)
  heights = height_scan(env, "multi")  # type: ignore[invalid-argument-type]

  assert heights.shape == (2, 3)
  # Each frame's height = frame_z - hit_z.
  # site_top at z=3.5, site_mid at z=3, site_bot at z=2.5; floor at z=0.
  assert heights[0, 0].item() == pytest.approx(3.5, abs=0.1)
  assert heights[0, 1].item() == pytest.approx(3.0, abs=0.1)
  assert heights[0, 2].item() == pytest.approx(2.5, abs=0.1)


# ============================================================================
# Ring Pattern Tests
# ============================================================================


def test_ring_pattern_num_rays(device):
  """Ring with 8 samples + center = 9 rays."""
  cfg = RayCastSensorCfg(
    name="ring",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=RingPatternCfg.single_ring(radius=0.1, num_samples=8, include_center=True),
  )

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

  scene, sim = make_scene_and_sim(device, simple_xml, (cfg,))
  sensor = scene["ring"]
  assert sensor.num_rays == 9
  assert sensor.num_rays_per_frame == 9


def test_ring_pattern_concentric(device):
  """Concentric rings: 1 center + 4 + 6 + 8 = 19 rays."""
  cfg = RayCastSensorCfg(
    name="ring",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=RingPatternCfg(
      rings=(
        RingPatternCfg.Ring(radius=0.05, num_samples=4),
        RingPatternCfg.Ring(radius=0.10, num_samples=6),
        RingPatternCfg.Ring(radius=0.20, num_samples=8),
      ),
      include_center=True,
    ),
  )

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

  scene, sim = make_scene_and_sim(device, simple_xml, (cfg,))
  sensor = scene["ring"]
  assert sensor.num_rays == 19


def test_ring_pattern_hits_floor(device):
  """Ring pattern hits floor and ring offsets lie on a circle."""
  radius = 0.1
  cfg = RayCastSensorCfg(
    name="ring",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=RingPatternCfg.single_ring(
      radius=radius, num_samples=4, include_center=True
    ),
    max_distance=10.0,
  )

  simple_xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1"/>
        <body name="base" pos="0 0 2">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.05"/>
        </body>
      </worldbody>
    </mujoco>
  """

  scene, sim = make_scene_and_sim(device, simple_xml, (cfg,))
  sim.step()
  sim.sense()

  data = scene["ring"].data
  # All 5 rays (1 center + 4 ring) should hit the floor.
  assert torch.all(data.distances >= 0)
  assert torch.allclose(data.distances, torch.full_like(data.distances, 2.0), atol=0.1)

  # Verify ring geometry: non-center hits should lie on a circle of
  # the given radius around the body's XY position.
  hit_xy = data.hit_pos_w[0, :, :2]  # [5, 2]
  body_xy = data.pos_w[0, :2]  # [2]
  dists_from_center = (hit_xy - body_xy).norm(dim=1)
  # Center ray (index 0) should be at ~0 offset.
  assert dists_from_center[0].item() == pytest.approx(0.0, abs=0.02)
  # Ring rays (indices 1-4) should be at ~radius offset.
  for i in range(1, 5):
    assert dists_from_center[i].item() == pytest.approx(radius, abs=0.02)


def test_ring_pattern_no_center(device):
  """Ring with include_center=False omits the center ray."""
  cfg = RayCastSensorCfg(
    name="ring",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=RingPatternCfg.single_ring(radius=0.1, num_samples=4, include_center=False),
  )
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
  scene, sim = make_scene_and_sim(device, simple_xml, (cfg,))
  assert scene["ring"].num_rays == 4


def test_multi_frame_multiple_rays_per_frame(device):
  """Multi-frame with grid pattern: verifies reshape ordering."""
  cfg = RayCastSensorCfg(
    name="multi",
    frame=(
      ObjRef(type="site", name="site_top", entity="robot"),
      ObjRef(type="site", name="site_bot", entity="robot"),
    ),
    pattern=GridPatternCfg(size=(0.2, 0.2), resolution=0.1),
    max_distance=10.0,
  )

  scene, sim = make_scene_and_sim(device, MULTI_SITE_XML, (cfg,))
  sim.step()
  sim.sense()

  sensor = scene["multi"]
  N = sensor.num_rays_per_frame
  assert N == 9  # 3x3 grid
  assert sensor.num_rays == 18  # 2 frames * 9

  data = sensor.data
  # First frame (site_top, z=3.5) rays should all be ~3.5.
  top_dists = data.distances[0, :N]
  assert torch.allclose(top_dists, torch.full_like(top_dists, 3.5), atol=0.1)
  # Second frame (site_bot, z=2.5) rays should all be ~2.5.
  bot_dists = data.distances[0, N:]
  assert torch.allclose(bot_dists, torch.full_like(bot_dists, 2.5), atol=0.1)

  # height_scan should also produce correct per-frame heights.
  env = _FakeEnv(scene)
  heights = height_scan(env, "multi")  # type: ignore[invalid-argument-type]
  assert heights.shape == (1, 18)
  top_heights = heights[0, :N]
  bot_heights = heights[0, N:]
  assert torch.allclose(top_heights, torch.full_like(top_heights, 3.5), atol=0.1)
  assert torch.allclose(bot_heights, torch.full_like(bot_heights, 2.5), atol=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_site_origin_is_physical_with_world_alignment(device):
  """ray_alignment only controls direction, not origin position.

  When a calf body pitches, the foot site swings to its physical
  position (site_xpos). Rays still point -Z (world-aligned), giving
  the correct distance from the actual foot to the ground.
  """
  xml = """
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <geom name="floor" type="plane" size="50 50 0.1" pos="0 0 0"/>
        <body name="calf" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="calf_geom" type="sphere" size="0.05" mass="1.0"/>
          <site name="foot" pos="0 0 -0.5"/>
        </body>
      </worldbody>
    </mujoco>
  """

  cfg = RayCastSensorCfg(
    name="scan",
    frame=ObjRef(type="site", name="foot", entity="robot"),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),
    ray_alignment="world",
    max_distance=5.0,
  )

  scene, sim = make_scene_and_sim(device, xml, (cfg,))

  # Baseline: body upright, site at z=0.5.
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  sim.sense()
  data = scene["scan"].data
  assert data.pos_w[0, 2].item() == pytest.approx(0.5, abs=0.05)
  assert data.distances[0, 0].item() == pytest.approx(0.5, abs=0.05)

  # Pitch calf 90 degrees. Site swings to (0, -0.5, 1.0).
  # Ray points -Z (world), distance to floor = 1.0.
  angle = math.pi / 2
  quat = [math.cos(angle / 2), math.sin(angle / 2), 0, 0]
  sim.data.qpos[0, 3:7] = torch.tensor(quat, device=device)
  sim.step()
  scene.update(dt=sim.cfg.mujoco.timestep)
  sim.sense()
  data = scene["scan"].data

  assert data.pos_w[0, 2].item() == pytest.approx(1.0, abs=0.1)
  assert data.distances[0, 0].item() == pytest.approx(1.0, abs=0.1)
