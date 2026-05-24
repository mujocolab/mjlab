"""Tests for sensor-based projected gravity (framezaxis up-vector sensor).

The shipped robots expose a ``framezaxis`` sensor that outputs the world Z-axis in the
IMU site frame; negating it gives projected gravity. These tests check the sensor (and
the ``projected_gravity_from_sensor`` observation that wraps it) against an independent
ground-truth computation, and verify that -- unlike the entity-data
``projected_gravity_b`` -- it tracks the IMU site orientation, which is what makes IMU
mounting domain randomization observable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.observations import projected_gravity_from_sensor
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Gravity points along world -Z; projected gravity is this expressed in a body frame.
_GRAVITY_DIR_W = (0.0, 0.0, -1.0)


def _quat_to_mat(q: tuple[float, float, float, float]) -> torch.Tensor:
  """Rotation matrix from a (w, x, y, z) quaternion. Independent of MuJoCo/mjlab."""
  w, x, y, z = q
  return torch.tensor(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ],
    dtype=torch.float64,
  )


def _expected_projected_gravity(q: tuple[float, float, float, float]) -> torch.Tensor:
  """Ground-truth projected gravity for a body with world orientation ``q``.

  proj = R(q)^T @ g_world, computed from an explicit rotation matrix so it does not
  share a code path with the sensor or with ``projected_gravity_b``.
  """
  g_w = torch.tensor(_GRAVITY_DIR_W, dtype=torch.float64)
  return _quat_to_mat(q).T @ g_w


class Env:
  """Minimal env stub for driving observation and dr functions in tests."""

  def __init__(self, scene, sim, device):
    self.scene = scene
    self.sim = sim
    self.num_envs = scene.num_envs
    self.device = device


def _make_env(scene, sim, device) -> ManagerBasedRlEnv:
  """Build the env stub, typed as the real env for the functions under test."""
  return cast("ManagerBasedRlEnv", Env(scene, sim, device))


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _robot_xml(site_euler: str = "0 0 0") -> str:
  """Free-floating box with an IMU site and the framezaxis up-vector sensor."""
  return f"""
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="5.0"/>
          <site name="imu" pos="0.05 0 0" euler="{site_euler}"/>
        </body>
      </worldbody>
      <sensor>
        <framezaxis name="imu_upvector" objtype="body" objname="world"
                    reftype="site" refname="imu"/>
      </sensor>
    </mujoco>
  """


def _build(xml: str, device: str, num_envs: int = 2):
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))
  scene = Scene(
    SceneCfg(num_envs=num_envs, env_spacing=3.0, entities={"robot": entity_cfg}),
    device,
  )
  model = scene.compile()
  sim = Simulation(
    num_envs=num_envs, cfg=SimulationCfg(njmax=20), model=model, device=device
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return scene, sim


def _set_root_quat(robot, q: tuple[float, float, float, float], device: str) -> None:
  root_state = robot.data.default_root_state.clone()
  root_state[:, 3:7] = torch.tensor(q, device=device, dtype=root_state.dtype)
  robot.write_root_state_to_sim(root_state)


def test_sensor_matches_ground_truth_when_site_aligned(device):
  """Sensor and entity both equal hand-computed projected gravity for a tilted base."""
  scene, sim = _build(_robot_xml(), device)
  robot = scene["robot"]

  # Compose a 0.6 rad roll with a 0.3 rad pitch into a single root quaternion.
  ax = (math.cos(0.3), math.sin(0.3), 0.0, 0.0)
  ay = (math.cos(0.15), 0.0, math.sin(0.15), 0.0)
  q = (
    ax[0] * ay[0] - ax[1] * ay[1] - ax[2] * ay[2] - ax[3] * ay[3],
    ax[0] * ay[1] + ax[1] * ay[0] + ax[2] * ay[3] - ax[3] * ay[2],
    ax[0] * ay[2] - ax[1] * ay[3] + ax[2] * ay[0] + ax[3] * ay[1],
    ax[0] * ay[3] + ax[1] * ay[2] - ax[2] * ay[1] + ax[3] * ay[0],
  )
  _set_root_quat(robot, q, device)
  sim.forward()

  expected = _expected_projected_gravity(q).to(device=device, dtype=torch.float32)
  # Guard against a vacuous pass: the tilt must actually move gravity off straight-down.
  straight_down = torch.tensor(_GRAVITY_DIR_W, device=device)
  assert (expected - straight_down).abs().max() > 0.3

  sensor_grav = -scene["robot/imu_upvector"].data
  entity_grav = robot.data.projected_gravity_b
  torch.testing.assert_close(sensor_grav[0], expected, atol=1e-5, rtol=0)
  torch.testing.assert_close(entity_grav[0], expected, atol=1e-5, rtol=0)


def test_observation_fn_tracks_site_orientation(device):
  """The observation fn reflects IMU site tilt; the entity-data version does not.

  With the base upright but the IMU site rolled 30 deg about x, projected gravity in the
  site frame is (0, -sin30, -cos30). The entity-data version stays straight-down because
  it uses the root body orientation and is blind to the site.
  """
  scene_rot, sim_rot = _build(_robot_xml(site_euler="30 0 0"), device)
  scene_flat, sim_flat = _build(_robot_xml(site_euler="0 0 0"), device)
  sim_rot.forward()
  sim_flat.forward()

  # Drive through the actual shipped observation function, not the raw sensor.
  env_rot = _make_env(scene_rot, sim_rot, device)
  env_flat = _make_env(scene_flat, sim_flat, device)
  grav_rot = projected_gravity_from_sensor(env_rot, "robot/imu_upvector")
  grav_flat = projected_gravity_from_sensor(env_flat, "robot/imu_upvector")

  expected_rot = torch.tensor(
    [0.0, -math.sin(math.radians(30)), -math.cos(math.radians(30))], device=device
  )
  straight_down = torch.tensor(_GRAVITY_DIR_W, device=device)
  torch.testing.assert_close(grav_rot[0], expected_rot, atol=1e-5, rtol=0)
  torch.testing.assert_close(grav_flat[0], straight_down, atol=1e-5, rtol=0)

  # The entity-data version is unchanged by the site rotation (so it cannot be used to
  # observe IMU mounting randomization), confirming why the sensor path is needed.
  entity_rot = scene_rot["robot"].data.projected_gravity_b
  torch.testing.assert_close(entity_rot[0], straight_down, atol=1e-5, rtol=0)


@pytest.mark.filterwarnings(
  "ignore:Use of index_put_ on expanded tensors is deprecated:UserWarning"
)
def test_site_quat_randomization_changes_sensor(device):
  """The full DR path: running ``dr.site_quat`` perturbs the gravity observation.

  This is what the G1 example configs rely on -- randomizing the IMU site orientation
  must show up in the sensor-based projected gravity, per-environment.
  """
  scene, sim = _build(_robot_xml(), device, num_envs=4)
  sim.expand_model_fields(("site_quat",))
  env = _make_env(scene, sim, device)

  sim.forward()
  straight_down = torch.tensor(_GRAVITY_DIR_W, device=device)
  before = projected_gravity_from_sensor(env, "robot/imu_upvector").clone()
  # Upright base + identity site quat => straight-down gravity in every env.
  torch.testing.assert_close(before, straight_down.expand_as(before), atol=1e-5, rtol=0)

  torch.manual_seed(0)
  dr.site_quat(
    env,
    env_ids=None,
    roll_range=(-0.3, 0.3),
    pitch_range=(-0.3, 0.3),
    yaw_range=(-0.3, 0.3),
    asset_cfg=SceneEntityCfg("robot", site_names=("imu",)),
  )
  sim.forward()
  after = projected_gravity_from_sensor(env, "robot/imu_upvector")

  # Randomization moved the reading off straight-down and made it env-dependent.
  assert (after - straight_down).abs().max() > 0.05
  assert not torch.allclose(after, before, atol=1e-3)
  assert torch.unique(after, dim=0).shape[0] >= 2
  # The perturbation is a rotation, so gravity stays a unit vector.
  norms = torch.linalg.norm(after, dim=-1)
  torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)
