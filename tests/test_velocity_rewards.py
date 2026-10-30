"""Tests for velocity task reward functions."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import torch
from conftest import get_test_device, make_scene_and_sim

from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactSensor,
  GridPatternCfg,
  ObjRef,
  RayCastData,
  RayCastSensor,
  RayCastSensorCfg,
  TerrainHeightSensor,
)
from mjlab.tasks.velocity.mdp.rewards import feet_swing_height, upright
from mjlab.utils.lab_api.math import quat_from_euler_xyz


def _identity_quat(B: int) -> torch.Tensor:
  """(w, x, y, z) = (1, 0, 0, 0)."""
  q = torch.zeros(B, 4)
  q[:, 0] = 1.0
  return q


def _quat_from_roll(roll_rad: float, B: int = 1) -> torch.Tensor:
  roll = torch.full((B,), roll_rad)
  zero = torch.zeros(B)
  return quat_from_euler_xyz(roll, zero, zero)


def _quat_from_pitch(pitch_rad: float, B: int = 1) -> torch.Tensor:
  pitch = torch.full((B,), pitch_rad)
  zero = torch.zeros(B)
  return quat_from_euler_xyz(zero, pitch, zero)


def _make_env_and_reward(
  terrain_sensor_names: tuple[str, ...] | None = None,
  body_quat_w: torch.Tensor | None = None,
  terrain_hit_z: float = 0.0,
  terrain_slope_x: float = 0.0,
):
  """Build mocked env + upright reward instance.

  Args:
    terrain_sensor_names: If set, enables terrain-aware mode.
    body_quat_w: [B, 4] root orientation. Defaults to identity.
    terrain_hit_z: Z value for flat terrain hits.
    terrain_slope_x: Slope in X (z = terrain_slope_x * x).
  """
  B = 1 if body_quat_w is None else body_quat_w.shape[0]
  if body_quat_w is None:
    body_quat_w = _identity_quat(B)

  # An empty body selection uses root_link_quat_w.
  asset = MagicMock()
  asset.data.root_link_quat_w = body_quat_w
  asset.data.root_link_pos_w = torch.zeros(B, 3)
  asset.data.gravity_vec_w = torch.tensor([0.0, 0.0, -1.0]).expand(B, 3)
  asset_cfg = SceneEntityCfg("robot", body_names=None, body_ids=[])

  # Mock terrain sensor if needed.
  sensors: dict = {"robot": asset}
  if terrain_sensor_names is not None:
    N = 100
    torch.manual_seed(0)
    hit_pos = torch.zeros(B, N, 3)
    hit_pos[:, :, 0] = torch.randn(B, N)
    hit_pos[:, :, 1] = torch.randn(B, N)
    hit_pos[:, :, 2] = terrain_hit_z + terrain_slope_x * hit_pos[:, :, 0]

    raycast_sensor = MagicMock(spec=RayCastSensor)
    raycast_data = RayCastData(
      distances=torch.ones(B, N),
      normals_w=torch.zeros(B, N, 3),
      hit_pos_w=hit_pos,
      pos_w=torch.zeros(B, 3),
      quat_w=torch.zeros(B, 4),
      frame_pos_w=torch.zeros(B, 1, 3),
      frame_quat_w=torch.zeros(B, 1, 4),
    )
    type(raycast_sensor).data = PropertyMock(return_value=raycast_data)
    for name in terrain_sensor_names:
      sensors[name] = raycast_sensor

  env = MagicMock()
  env.scene.__getitem__ = MagicMock(side_effect=lambda n: sensors[n])

  params: dict = {"std": 1.0, "asset_cfg": asset_cfg}
  if terrain_sensor_names is not None:
    params["terrain_sensor_names"] = terrain_sensor_names
  cfg = MagicMock(spec=RewardTermCfg)
  cfg.params = params

  reward_fn = upright(cfg, env)
  return env, reward_fn, params


def test_world_up_identity_gives_max_reward():
  """Perfectly upright robot on flat ground → reward ≈ 1."""
  env, reward, params = _make_env_and_reward()
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])
  assert r.shape == (1,)
  assert r.item() > 0.99


def test_world_up_tilted_gives_lower_reward():
  """30° roll → reward significantly below 1."""
  quat = _quat_from_roll(math.radians(30))
  env, reward, params = _make_env_and_reward(body_quat_w=quat)
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])
  assert r.item() < 0.8


def test_terrain_aware_aligned_with_slope():
  """Robot pitched to match a slope → terrain-aware reward ≈ 1."""
  slope = 0.5  # z = 0.5 * x
  tilt = math.atan(slope)  # Pitch to match slope in XZ plane.
  quat = _quat_from_pitch(-tilt)
  env, reward, params = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
    terrain_slope_x=slope,
  )
  r = reward(
    env,
    std=params["std"],
    asset_cfg=params["asset_cfg"],
    terrain_sensor_names=params["terrain_sensor_names"],
  )
  # Should be close to 1 since robot matches terrain.
  assert r.item() > 0.9


def test_terrain_aware_upright_on_slope_penalized():
  """Robot staying vertical on a slope → terrain-aware reward < 1."""
  slope = 0.5
  quat = _identity_quat(1)  # Robot is world-vertical, not matching slope.
  env, reward, params = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
    terrain_slope_x=slope,
  )
  r = reward(
    env,
    std=params["std"],
    asset_cfg=params["asset_cfg"],
    terrain_sensor_names=params["terrain_sensor_names"],
  )
  # Should be penalized since robot doesn't match terrain.
  assert r.item() < 0.95


def test_terrain_aware_flat_ground_matches_world_up():
  """On flat terrain, terrain-aware and world-up should give same reward."""
  quat = _quat_from_roll(math.radians(15))
  env_t, reward_t, params_t = _make_env_and_reward(
    terrain_sensor_names=("terrain_scan",),
    body_quat_w=quat,
  )
  env_w, reward_w, params_w = _make_env_and_reward(body_quat_w=quat)

  r_terrain = reward_t(
    env_t,
    std=params_t["std"],
    asset_cfg=params_t["asset_cfg"],
    terrain_sensor_names=params_t["terrain_sensor_names"],
  )
  r_world = reward_w(env_w, std=params_w["std"], asset_cfg=params_w["asset_cfg"])

  torch.testing.assert_close(r_terrain, r_world, atol=0.02, rtol=0.02)


def test_batch_consistency():
  """Multiple envs with different orientations get independent rewards."""
  B = 4
  quats = torch.zeros(B, 4)
  quats[:, 0] = 1.0  # All identity.
  # Tilt env 2 by 45°.
  quats[2] = _quat_from_roll(math.radians(45))[0]

  env, reward, params = _make_env_and_reward(body_quat_w=quats)
  r = reward(env, std=params["std"], asset_cfg=params["asset_cfg"])

  assert r.shape == (B,)
  # Env 0, 1, 3 should be ~1, env 2 should be lower.
  assert r[0].item() > 0.99
  assert r[1].item() > 0.99
  assert r[2].item() < 0.7
  assert r[3].item() > 0.99


@pytest.fixture(scope="module")
def upright_env():
  """Real articulated entity with different root and child orientations."""
  device = get_test_device()
  xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 0.1" group="0"/>
        <body name="base" pos="0 0 1">
          <freejoint/>
          <geom type="sphere" size="0.1" mass="1" group="1"/>
          <body name="child" pos="0 0 0.3">
            <joint name="hinge" type="hinge" axis="1 0 0"/>
            <geom type="sphere" size="0.1" mass="0.1" group="1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
  """
  sensor = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="base", entity="robot"),
    pattern=GridPatternCfg(size=(0.4, 0.4), resolution=0.2),
    ray_alignment="world",
    include_geom_groups=(0,),
  )
  scene, sim = make_scene_and_sim(device, xml, sensors=(sensor,), num_envs=2)
  asset = scene["robot"]
  root_state = torch.zeros(2, 13, device=device)
  root_state[:, 2] = 1.0
  root_state[:, 3:7] = torch.cat((_identity_quat(1), _quat_from_roll(math.pi / 6))).to(
    device
  )
  asset.write_root_state_to_sim(root_state)
  asset.write_joint_state_to_sim(
    position=torch.full((2, 1), math.pi / 3, device=device),
    velocity=torch.zeros(2, 1, device=device),
  )
  sim.forward()
  sim.sense()

  env = MagicMock()
  env.scene = scene
  env.num_envs = 2
  env.device = device
  return env


@pytest.mark.parametrize("terrain_aware", [False, True])
@pytest.mark.parametrize(
  "asset_cfg,child_selected",
  [
    pytest.param(None, False, id="omitted"),
    pytest.param(SceneEntityCfg("robot"), False, id="default"),
    pytest.param(SceneEntityCfg("robot", body_ids=[]), False, id="empty"),
    pytest.param(SceneEntityCfg("robot", body_names=("child",)), True, id="body-name"),
    pytest.param(SceneEntityCfg("robot", body_ids=[1]), True, id="body-id"),
    pytest.param(SceneEntityCfg("robot", body_ids=slice(1, 2)), True, id="body-slice"),
  ],
)
def test_upright_body_selection(upright_env, asset_cfg, child_selected, terrain_aware):
  """Default selection uses the root, while explicit selections use that body."""
  params: dict = {"std": 1.0}
  if asset_cfg is not None:
    params["asset_cfg"] = asset_cfg
  if terrain_aware:
    params["terrain_sensor_names"] = ("terrain_scan",)
  manager = RewardManager(
    {"upright": RewardTermCfg(func=upright, weight=1.0, params=params)},
    upright_env,
    scale_by_dt=False,
  )
  roll = torch.tensor([0.0, math.pi / 6], device=upright_env.device)
  if child_selected:
    roll += math.pi / 3

  if terrain_aware:
    visualizer = MagicMock()
    visualizer.get_env_indices.return_value = range(2)
    manager.debug_vis(visualizer)
    arrows = visualizer.add_arrow.call_args_list
    assert len(arrows) == 4
    for i in range(2):
      terrain_arrow, body_arrow = arrows[2 * i : 2 * i + 2]
      for arrow in (terrain_arrow, body_arrow):
        assert arrow.kwargs["start"].shape == (3,)
        assert arrow.kwargs["end"].shape == (3,)
      np.testing.assert_allclose(
        terrain_arrow.kwargs["end"] - terrain_arrow.kwargs["start"],
        [0.0, 0.0, 0.25],
        atol=1e-6,
      )
      np.testing.assert_allclose(
        body_arrow.kwargs["end"] - body_arrow.kwargs["start"],
        [0.0, -0.25 * math.sin(roll[i]), 0.25 * math.cos(roll[i])],
        atol=1e-6,
      )

  rewards = manager.compute(dt=0.02)
  assert rewards.shape == (2,)
  torch.testing.assert_close(rewards, torch.exp(-torch.sin(roll).square()))


def test_feet_swing_height_reset_clears_peak_heights():
  """Resetting an env mid-swing clears its peak_heights before the next landing."""
  B, num_feet = 2, 2
  contact_sensor = MagicMock(spec=ContactSensor)
  contact_sensor.data.found = torch.zeros(B, num_feet)
  contact_sensor.compute_first_contact.return_value = torch.zeros(
    B, num_feet, dtype=torch.bool
  )

  height_sensor = MagicMock(spec=TerrainHeightSensor)
  height_sensor.num_frames = num_feet
  height_sensor.data.heights = torch.full((B, num_feet), 0.5)

  env = MagicMock()
  env.num_envs = B
  env.device = "cpu"
  env.step_dt = 0.02
  env.max_episode_length_s = 10.0
  env.extras = {"log": {}}
  sensors = {"feet_contact": contact_sensor, "foot_height": height_sensor}
  env.scene.__getitem__ = MagicMock(side_effect=lambda k: sensors[k])
  env.command_manager.get_command.return_value = torch.tensor([[1.0, 0.0, 0.0]] * B)

  cfg = {
    "swing": RewardTermCfg(
      func=feet_swing_height,
      weight=1.0,
      params={
        "sensor_name": "feet_contact",
        "height_sensor_name": "foot_height",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    )
  }
  manager = RewardManager(cfg, env, scale_by_dt=False)

  # Step 1: both envs are in the air at 0.5 m, recording peak_heights = 0.5.
  manager.compute(dt=0.02)
  term = manager._class_term_cfgs[0].func
  assert torch.allclose(term.peak_heights, torch.full((B, num_feet), 0.5))

  # Env 0 resets mid-air; env 1 continues its episode.
  manager.reset(env_ids=torch.tensor([0]))
  assert torch.allclose(term.peak_heights[0], torch.zeros(num_feet))
  assert torch.allclose(term.peak_heights[1], torch.full((num_feet,), 0.5))

  # Step 2: env 0 swings to target_height (0.1 m) in the new episode.
  height_sensor.data.heights = torch.tensor([[0.1, 0.1], [0.1, 0.1]])
  manager.compute(dt=0.02)

  # Step 3: both envs land. Env 0 has zero error; env 1 is penalized for its 0.5 m peak.
  contact_sensor.data.found = torch.ones(B, num_feet)
  contact_sensor.compute_first_contact.return_value = torch.ones(
    B, num_feet, dtype=torch.bool
  )
  rewards = manager.compute(dt=0.02)
  expected = torch.tensor([0.0, 2.0 * (0.5 / 0.1 - 1.0) ** 2])
  torch.testing.assert_close(rewards, expected)
