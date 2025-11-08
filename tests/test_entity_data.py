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
  """Test that reading and writing root velocity is a no-op (both in world frame)."""
  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  # fmt: off
  root_state = torch.tensor([
    0.0, 0.0, 1.0,
    0.6, 0.2, 0.3, 0.7141,
    1.0, 0.5, 0.0,
    0.0, 0.3, 0.1
  ], device=device).unsqueeze(0)
  # fmt: on
  entity.write_root_state_to_sim(root_state)
  sim.forward()

  vel_w_before = entity.data.root_link_vel_w.clone()
  entity.write_root_link_velocity_to_sim(vel_w_before)
  sim.forward()
  vel_w_after = entity.data.root_link_vel_w

  assert torch.allclose(vel_w_after, vel_w_before, atol=1e-4), (
    "Reading and writing root velocity should be a no-op"
  )


def test_root_velocity_frame_conversion(device):
  """Test that angular velocity is correctly converted from world to body frame."""
  from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse

  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  quat_w = torch.tensor([0.6, 0.2, 0.3, 0.7141], device=device).unsqueeze(0)
  lin_vel_w = torch.tensor([1.0, 0.5, 0.2], device=device).unsqueeze(0)
  ang_vel_w = torch.tensor([0.1, 0.2, 0.3], device=device).unsqueeze(0)

  vel_w = torch.cat([lin_vel_w, ang_vel_w], dim=-1)
  entity.write_root_link_pose_to_sim(
    torch.cat([torch.zeros(1, 3, device=device), quat_w], dim=-1)
  )
  entity.write_root_link_velocity_to_sim(vel_w)

  v_slice = entity.data.indexing.free_joint_v_adr
  qvel = sim.data.qvel[:, v_slice]

  assert torch.allclose(qvel[:, :3], lin_vel_w, atol=1e-5)

  expected_ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)
  assert torch.allclose(qvel[:, 3:], expected_ang_vel_b, atol=1e-5), (
    "Angular velocity should be converted from world to body frame in qvel"
  )
