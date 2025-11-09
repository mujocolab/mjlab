"""Tests for EntityData."""

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import Entity, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg

FLOATING_BASE_XML = """
<mujoco>
  <worldbody>
    <body name="object" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="object_geom" type="box" size="0.1 0.1 0.1" rgba="0.3 0.3 0.8 1" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


def create_floating_base_entity():
  """Create a floating-base entity."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(FLOATING_BASE_XML))
  return Entity(cfg)


def initialize_entity_with_sim(entity, device, num_envs=1):
  """Initialize an entity with a simulation."""
  model = entity.compile()
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


def test_root_velocity_world_frame_roundtrip(device):
  """Test reading and writing root velocity is a no-op (world frame).

  Verifies that the API is consistent: if you write a velocity, read it
  back, and write it again, you get the same result. This ensures no
  unintended transformations happen in the read/write cycle.
  """
  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  pose = torch.tensor([0.0, 0.0, 1.0, 0.6, 0.2, 0.3, 0.7141], device=device).unsqueeze(
    0
  )
  entity.write_root_link_pose_to_sim(pose)

  vel_w = torch.tensor([1.0, 0.5, 0.0, 0.0, 0.3, 0.1], device=device).unsqueeze(0)
  entity.write_root_link_velocity_to_sim(vel_w)
  sim.forward()

  vel_w_read = entity.data.root_link_vel_w.clone()
  assert torch.allclose(vel_w_read, vel_w, atol=1e-4)

  entity.write_root_link_velocity_to_sim(vel_w_read)
  sim.forward()
  vel_w_after = entity.data.root_link_vel_w

  assert torch.allclose(vel_w_after, vel_w_read, atol=1e-4)


def test_root_velocity_frame_conversion(device):
  """Test angular velocity converts from world to body frame internally.

  The API accepts angular velocity in world frame, but MuJoCo's qvel
  stores it in body frame. This test verifies the conversion happens
  correctly by checking qvel directly.
  """
  from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    quat_apply_inverse,
  )

  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  quat_w = torch.tensor([0.6, 0.2, 0.3, 0.7141], device=device).unsqueeze(0)
  pose = torch.cat([torch.zeros(1, 3, device=device), quat_w], dim=-1)
  entity.write_root_link_pose_to_sim(pose)

  lin_vel_w = torch.tensor([1.0, 0.5, 0.2], device=device).unsqueeze(0)
  ang_vel_w = torch.tensor([0.1, 0.2, 0.3], device=device).unsqueeze(0)
  vel_w = torch.cat([lin_vel_w, ang_vel_w], dim=-1)
  entity.write_root_link_velocity_to_sim(vel_w)

  v_slice = entity.data.indexing.free_joint_v_adr
  qvel = sim.data.qvel[:, v_slice]

  assert torch.allclose(qvel[:, :3], lin_vel_w, atol=1e-5)

  expected_ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)
  assert torch.allclose(qvel[:, 3:], expected_ang_vel_b, atol=1e-5)


def test_write_velocity_uses_qpos_not_xquat(device):
  """Test write_root_velocity uses qpos (not stale xquat).

  Writing pose then velocity without forward() must work. This would fail
  if write_root_velocity used xquat (stale) instead of qpos (current).
  """
  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  initial_pose = torch.tensor(
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], device=device
  ).unsqueeze(0)
  entity.write_root_link_pose_to_sim(initial_pose)
  sim.forward()  # xquat now has identity orientation.

  # Write different orientation without forward() - xquat stale, qpos current.
  new_pose = torch.tensor(
    [0.0, 0.0, 1.0, 0.707, 0.0, 0.707, 0.0], device=device
  ).unsqueeze(0)
  vel_w = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device).unsqueeze(0)

  entity.write_root_link_pose_to_sim(new_pose)
  entity.write_root_link_velocity_to_sim(vel_w)

  sim.forward()
  vel_w_read = entity.data.root_link_vel_w

  assert torch.allclose(vel_w_read, vel_w, atol=1e-4)


def test_read_requires_forward_to_be_current(device):
  """Test read properties are stale until forward() is called.

  Demonstrates why event order matters and why forward() is needed
  between writes and reads.
  """
  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  sim.forward()
  initial_pose = entity.data.root_link_pose_w.clone()

  new_pose = torch.tensor(
    [1.0, 2.0, 3.0, 0.707, 0.0, 0.707, 0.0], device=device
  ).unsqueeze(0)
  entity.write_root_link_pose_to_sim(new_pose)

  stale_pose = entity.data.root_link_pose_w
  assert torch.allclose(stale_pose, initial_pose, atol=1e-5)

  sim.forward()
  current_pose = entity.data.root_link_pose_w
  assert torch.allclose(current_pose, new_pose, atol=1e-4)
  assert not torch.allclose(current_pose, initial_pose, atol=1e-4)
