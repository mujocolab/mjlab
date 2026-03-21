"""Verify TerrainHeightSensor.data.heights matches analytic expectations.

Two key properties:
1. On flat terrain, heights must equal site_pos_w Z (terrain at z=0).
2. On stepped terrain, heights must equal site_z - step_z.
"""

from __future__ import annotations

import pytest
import torch
from conftest import get_test_device, make_scene_and_sim

from mjlab.sensor import ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

# Single foot floating above a flat ground plane at z=0.
FLAT_TERRAIN_XML = """
  <mujoco>
    <worldbody>
      <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
      <body name="base" pos="0 0 1.0">
        <freejoint name="free_joint"/>
        <geom name="base_geom" type="sphere" size="0.05" mass="1.0" group="1"/>
        <site name="left_foot" pos="-0.1 0 -0.2"/>
        <site name="right_foot" pos="0.1 0 -0.4"/>
      </body>
    </worldbody>
  </mujoco>
"""

# Step at z=0.3 for x<0, flat ground at z=0 for x>=0.
# Body at x=0, z=1.0.
# left_foot at x=-0.5 (over step), right_foot at x=0.5 (over ground).
STEPPED_TERRAIN_XML = """
  <mujoco>
    <worldbody>
      <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
      <geom name="step" type="box" size="5 5 0.15" pos="-5 0 0.15"/>
      <body name="base" pos="0 0 1.0">
        <freejoint name="free_joint"/>
        <geom name="base_geom" type="sphere" size="0.05" mass="1.0" group="1"/>
        <site name="left_foot" pos="-0.5 0 0"/>
        <site name="right_foot" pos="0.5 0 0"/>
      </body>
    </worldbody>
  </mujoco>
"""


def _sensor_cfg(reduction: str = "mean") -> TerrainHeightSensorCfg:
  return TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(
      ObjRef(type="site", name="left_foot", entity="robot"),
      ObjRef(type="site", name="right_foot", entity="robot"),
    ),
    ray_alignment="yaw",
    pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
    max_distance=2.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    reduction=reduction,
  )


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def test_flat_terrain_heights_match_site_z(device):
  """On flat terrain (z=0), sensor heights must equal site_pos_w Z."""
  cfg = _sensor_cfg()
  scene, sim = make_scene_and_sim(device, FLAT_TERRAIN_XML, (cfg,))
  entity = scene["robot"]
  sensor: TerrainHeightSensor = scene["foot_height_scan"]

  sim.step()
  sim.forward()
  sim.sense()

  site_z = entity.data.site_pos_w[0, :, 2]  # [num_sites]
  sensor_heights = sensor.data.heights[0]  # [num_frames]

  assert site_z.shape == sensor_heights.shape, (
    f"Shape mismatch: site_z {site_z.shape} vs sensor {sensor_heights.shape}"
  )
  torch.testing.assert_close(
    sensor_heights,
    site_z,
    atol=1e-3,
    rtol=0,
    msg="Sensor heights diverge from site_pos_w Z on flat terrain",
  )


def test_flat_terrain_heights_match_site_z_after_motion(device):
  """After several steps of free-fall, sensor heights still match site Z."""
  cfg = _sensor_cfg()
  scene, sim = make_scene_and_sim(device, FLAT_TERRAIN_XML, (cfg,))
  entity = scene["robot"]
  sensor: TerrainHeightSensor = scene["foot_height_scan"]

  # Let the body fall for a few steps.
  for _ in range(10):
    sim.step()
  sim.forward()
  sim.sense()

  site_z = entity.data.site_pos_w[0, :, 2]
  sensor_heights = sensor.data.heights[0]

  torch.testing.assert_close(
    sensor_heights,
    site_z,
    atol=1e-3,
    rtol=0,
    msg="Sensor heights diverge from site Z after motion on flat terrain",
  )


def test_stepped_terrain_analytic_heights(device):
  """On stepped terrain, heights must equal site_z - terrain_z."""
  cfg = _sensor_cfg()
  scene, sim = make_scene_and_sim(device, STEPPED_TERRAIN_XML, (cfg,))
  sensor: TerrainHeightSensor = scene["foot_height_scan"]

  sim.step()
  sim.forward()
  sim.sense()

  heights = sensor.data.heights[0]

  # left_foot at (-0.5, 0, 1.0), over step top at z=0.3 -> height = 0.7.
  assert heights[0].item() == pytest.approx(0.7, abs=0.05)
  # right_foot at (0.5, 0, 1.0), over ground at z=0 -> height = 1.0.
  assert heights[1].item() == pytest.approx(1.0, abs=0.05)


def test_reduction_min_vs_mean_on_flat(device):
  """On flat terrain, min and mean reduction should give identical results."""
  cfg_min = _sensor_cfg(reduction="min")
  cfg_min.name = "foot_min"
  cfg_mean = _sensor_cfg(reduction="mean")
  cfg_mean.name = "foot_mean"

  scene, sim = make_scene_and_sim(device, FLAT_TERRAIN_XML, (cfg_min, cfg_mean))
  sim.step()
  sim.forward()
  sim.sense()

  sensor_min: TerrainHeightSensor = scene["foot_min"]
  sensor_mean: TerrainHeightSensor = scene["foot_mean"]

  torch.testing.assert_close(
    sensor_min.data.heights,
    sensor_mean.data.heights,
    atol=1e-3,
    rtol=0,
    msg="Min and mean reduction diverge on flat terrain",
  )


def test_foot_below_ground_plane(device):
  """When foot penetrates ground (site Z < 0), sensor should not report max_distance."""
  # Foot site at z=0.01 (just barely above ground) and z=-0.01 (slightly below).
  below_ground_xml = """
    <mujoco>
      <worldbody>
        <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 0.05">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="sphere" size="0.02" mass="1.0" group="1"/>
          <site name="left_foot" pos="0 0 -0.04"/>
          <site name="right_foot" pos="0 0 -0.06"/>
        </body>
      </worldbody>
    </mujoco>
  """
  # left_foot at z=0.01 (above ground), right_foot at z=-0.01 (below ground).
  cfg = _sensor_cfg()
  scene, sim = make_scene_and_sim(device, below_ground_xml, (cfg,))
  sim.step()
  sim.forward()
  sim.sense()

  sensor: TerrainHeightSensor = scene["foot_height_scan"]
  heights = sensor.data.heights[0]

  # Left foot (above ground) should report ~0.01.
  assert heights[0].item() < 0.1, (
    f"Left foot above ground should report small height, got {heights[0].item()}"
  )
  # Right foot (below ground) should NOT report max_distance.
  assert heights[1].item() < 0.5, (
    f"Right foot below ground reports {heights[1].item()}, "
    f"likely max_distance={cfg.max_distance} due to ray miss"
  )
