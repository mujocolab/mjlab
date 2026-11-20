"""Tests for tactile_sensor.py."""

import pytest
import torch
from conftest import get_test_device, make_scene_and_sim

from mjlab.sensor import ObjRef, TactileSensorCfg

PAD_XML = """
<mujoco>
  <worldbody>
    <body name="pad"><geom name="pad" type="box" size="0.02 0.02 0.002"/></body>
  </worldbody>
</mujoco>
"""

BALL_XML = """
<mujoco>
  <worldbody>
    <body name="ball">
      <freejoint/>
      <geom type="sphere" size="0.005" mass="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

SHAPES_XML = """
<mujoco>
  <asset>
    <mesh name="cube" scale="0.01 0.01 0.01"
      vertex="-1 -1 -1  1 -1 -1  -1 1 -1  1 1 -1  -1 -1 1  1 -1 1  -1 1 1  1 1 1"/>
  </asset>
  <worldbody>
    <body name="shapes">
      <geom name="sphere" type="sphere" size="0.01"/>
      <geom name="cube" type="mesh" mesh="cube" pos="0.1 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""

RADIUS, PRESS, PAD_TOP = 0.005, 0.001, 0.002


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def pressed(device):
  """A ball pressed 1 mm into a flat pad, sliding along +x at 0.1 m/s."""
  cfg = TactileSensorCfg(
    name="tactile",
    geom=ObjRef("geom", "pad", entity="pad"),
    size=(0.01, 0.01),
    resolution=(5, 5),
  )
  scene, sim = make_scene_and_sim(
    device, {"pad": PAD_XML, "ball": BALL_XML}, sensors=(cfg,)
  )
  center = (0.0, 0.0, PAD_TOP + RADIUS - PRESS)
  state = torch.tensor([[*center, 1, 0, 0, 0, 0.1, 0, 0, 0, 0, 0]], device=device)
  scene["ball"].write_root_state_to_sim(state)
  sim.forward()
  return scene["pad/tactile"], torch.tensor(center, device=device)


def test_depth_matches_sphere_distance(pressed):
  sensor, center = pressed
  taxels = sensor.taxel_pos_w()[0]
  expected = (RADIUS - (taxels - center).norm(dim=-1)).clamp(min=0).view(5, 5)
  depth = sensor.data.depth[0]
  assert depth[2, 2] == pytest.approx(PRESS, abs=1e-6)
  torch.testing.assert_close(depth, expected, atol=1e-6, rtol=0)


def test_slip_follows_sliding_direction(pressed):
  sensor, _ = pressed
  data = sensor.data
  touching = data.depth[0] > 0
  assert touching.sum() > 1
  torch.testing.assert_close(data.slip[0, ..., 0], 0.1 * touching.float())
  assert torch.all(data.slip[0, ..., 1] == 0)


def test_taxels_lie_on_the_collision_surface(device):
  sensors = tuple(
    TactileSensorCfg(
      name=name,
      geom=ObjRef("geom", name, entity="obj"),
      pos=pos,
      size=(0.008, 0.008),
      resolution=(4, 4),
    )
    for name, pos in (("sphere", (0.0, 0.0, 0.0)), ("cube", (0.1, 0.0, 0.0)))
  )
  scene, sim = make_scene_and_sim(device, {"obj": SHAPES_XML}, sensors=sensors)
  sim.forward()
  on_sphere = scene["obj/sphere"].taxel_pos_w()[0]
  on_cube = scene["obj/cube"].taxel_pos_w()[0]
  torch.testing.assert_close(
    on_sphere.norm(dim=-1), torch.full((16,), 0.01, device=device)
  )
  torch.testing.assert_close(on_cube[:, 2], torch.full((16,), 0.01, device=device))


def test_grid_missing_the_geom_raises(device):
  cfg = TactileSensorCfg(
    name="tactile", geom=ObjRef("geom", "pad", entity="pad"), size=(0.05, 0.05)
  )
  with pytest.raises(ValueError, match="miss geom"):
    make_scene_and_sim(device, {"pad": PAD_XML}, sensors=(cfg,))
