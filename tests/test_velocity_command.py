"""Tests for the velocity command's initial-velocity injection.

The init_velocity_prob path runs inside the reset pipeline, after reset
events wrote the new pose to qpos but before sim.forward(), so it must not
read (or write back) derived kinematics.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch
from conftest import get_test_device, load_fixture_xml, make_scene_and_sim

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def test_init_velocity_preserves_fresh_reset_pose(device):
  scene, sim = make_scene_and_sim(
    device, load_fixture_xml("floating_base_articulated"), sensors=(), num_envs=2
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(scene=scene, sim=sim, num_envs=2, device=device),
  )
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1e9, 1e9),
    init_velocity_prob=1.0,
    rel_heading_envs=0.0,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.5, 0.5), lin_vel_y=(0.2, 0.2), ang_vel_z=(0.0, 0.0)
    ),
  )
  term = cfg.build(env)
  robot = scene["robot"]
  env_ids = torch.arange(2, device=device)

  # Derived kinematics now hold the spawn pose (the "previous episode" state).
  sim.forward()

  # Emulate a reset event: write a fresh pose to qpos, no forward yet.
  pose = torch.tensor(
    [
      [1.0, 2.0, 1.5, 1.0, 0.0, 0.0, 0.0],
      [3.0, -1.0, 1.5, 1.0, 0.0, 0.0, 0.0],
    ],
    device=device,
  )
  robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)

  term.reset(env_ids=env_ids)
  sim.forward()

  # The fresh reset pose survives. Before the fix, the init-velocity path
  # wrote the stale pre-reset pose back into the sim.
  assert torch.allclose(robot.data.root_link_pos_w, pose[:, :3], atol=1e-6)

  # Planar velocity matches the sampled command (identity orientation, so
  # body frame equals world frame).
  assert torch.allclose(
    robot.data.root_link_lin_vel_b[:, :2],
    term.vel_command_b[:, :2],
    atol=1e-5,
  )


def test_mid_episode_resample_does_not_write_velocity(device):
  """Init velocity applies on reset only; a timer-expiry resample runs after
  step()'s forward and must not write sim state."""
  scene, sim = make_scene_and_sim(
    device, load_fixture_xml("floating_base_articulated"), sensors=(), num_envs=2
  )
  env = cast(
    "ManagerBasedRlEnv",
    SimpleNamespace(scene=scene, sim=sim, num_envs=2, device=device, step_dt=0.02),
  )
  cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(0.001, 0.001),  # Expires on the first compute.
    init_velocity_prob=1.0,
    rel_heading_envs=0.0,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.7, 0.7), lin_vel_y=(0.3, 0.3), ang_vel_z=(0.0, 0.0)
    ),
  )
  term = cfg.build(env)
  robot = scene["robot"]
  env_ids = torch.arange(2, device=device)

  term.reset(env_ids=env_ids)
  sim.forward()

  # Mid-episode the robot has decelerated to rest.
  robot.write_root_link_velocity_to_sim(
    torch.zeros(2, 6, device=device), env_ids=env_ids
  )
  sim.forward()

  # Step's command compute: the 1 ms timer expires and resamples.
  counter = term.command_counter.clone()
  term.compute(dt=1.0)
  assert (term.command_counter > counter).all()

  # Only the command changed; sim velocity is untouched.
  qvel_lin = sim.data.qvel[:, robot.indexing.free_joint_v_adr[:3]]
  assert torch.allclose(qvel_lin, torch.zeros_like(qvel_lin), atol=1e-6)
  assert torch.allclose(
    robot.data.root_link_lin_vel_b,
    torch.zeros_like(robot.data.root_link_lin_vel_b),
    atol=1e-6,
  )
