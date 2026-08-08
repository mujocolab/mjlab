"""Tests for lifting command resample kinematics freshness.

A timer-expiry resample runs inside command compute, after the step's single
sim.forward(). The commands must refresh kinematics themselves so observations
and the reward-path cache see post-teleport state.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch
from conftest import get_test_device, make_scene_and_sim

from mjlab.tasks.manipulation.mdp.commands import (
  LiftingCommandCfg,
  MultiCubeLiftingCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

CUBE_XML = """
<mujoco>
  <worldbody>
    <body name="cube">
      <freejoint/>
      <geom name="cube_geom" type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _make_env(device, entity_names):
  scene, sim = make_scene_and_sim(
    device, {name: CUBE_XML for name in entity_names}, sensors=(), num_envs=2
  )
  env = SimpleNamespace(scene=scene, sim=sim, num_envs=scene.num_envs, device=device)
  return cast("ManagerBasedRlEnv", env)


def _assert_kinematics_fresh(scene, sim, entity_name):
  """xpos-derived root pos must match qpos after a compute-path resample."""
  ent = scene[entity_name]
  q_adr = ent.indexing.free_joint_q_adr
  qpos_pos = sim.data.qpos[:, q_adr[:3]]
  assert torch.allclose(ent.data.root_link_pos_w, qpos_pos, atol=1e-6)


def test_lifting_resample_refreshes_kinematics(device):
  env = _make_env(device, ("cube0",))
  cfg = LiftingCommandCfg(entity_name="cube0", resampling_time_range=(0.001, 0.001))
  term = cfg.build(env)

  term.reset(env_ids=torch.arange(env.num_envs, device=device))
  env.sim.forward()

  counter = term.command_counter.clone()
  term.compute(dt=1.0)  # Timer (1 ms) expires: teleports the object.
  assert (term.command_counter > counter).all()

  _assert_kinematics_fresh(env.scene, env.sim, "cube0")


def test_multi_cube_resample_refreshes_kinematics_and_cache(device):
  names = ("cube0", "cube1")
  env = _make_env(device, names)
  cfg = MultiCubeLiftingCommandCfg(
    entity_names=names, resampling_time_range=(0.001, 0.001)
  )
  term = cfg.build(env)

  term.reset(env_ids=torch.arange(env.num_envs, device=device))
  env.sim.forward()

  counter = term.command_counter.clone()
  term.compute(dt=1.0)  # Timer (1 ms) expires: teleports all cubes.
  assert (term.command_counter > counter).all()

  for name in names:
    _assert_kinematics_fresh(env.scene, env.sim, name)

  # Reward-path cache serves the post-teleport position of the target cube.
  all_pos = torch.stack([env.scene[n].data.root_link_pos_w for n in names])
  arange = torch.arange(env.num_envs, device=device)
  expected = all_pos[term.target_selection, arange]
  assert torch.allclose(term.target_object_pos(), expected, atol=1e-6)
