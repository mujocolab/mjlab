"""Tests for ball joint support in the Entity system."""

import mujoco
import pytest
import torch
from conftest import get_test_device, initialize_entity

from mjlab.entity import Entity, EntityCfg

# XML with ball joints for testing.
BALL_JOINT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="sphere" size="0.1" mass="1"/>
      <body name="link1" pos="0.2 0 0">
        <joint name="ball1" type="ball"/>
        <geom name="link1_geom" type="capsule" size="0.05" fromto="0 0 0 0.2 0 0" mass="0.5"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="hinge1" type="hinge" axis="0 1 0"/>
          <geom name="link2_geom" type="capsule" size="0.05" fromto="0 0 0 0.2 0 0" mass="0.5"/>
          <body name="link3" pos="0.2 0 0">
            <joint name="ball2" type="ball"/>
            <geom name="link3_geom" type="sphere" size="0.05" mass="0.3"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# XML with only hinge joints for comparison.
HINGE_ONLY_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="sphere" size="0.1" mass="1"/>
      <body name="link1" pos="0.2 0 0">
        <joint name="hinge1" type="hinge" axis="1 0 0"/>
        <geom name="link1_geom" type="capsule" size="0.05" fromto="0 0 0 0.2 0 0" mass="0.5"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="hinge2" type="hinge" axis="0 1 0"/>
          <geom name="link2_geom" type="capsule" size="0.05" fromto="0 0 0 0.2 0 0" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


def create_ball_joint_entity():
  """Create an entity with ball joints."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(BALL_JOINT_XML))
  return Entity(cfg)


def create_hinge_only_entity():
  """Create an entity with only hinge joints."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(HINGE_ONLY_XML))
  return Entity(cfg)


def test_nq_nv_with_ball_joints(device):
  """Test that nq and nv are correctly computed for ball joints.

  Ball joints have 4 qpos (quaternion) and 3 qvel (angular velocity).
  Hinge joints have 1 qpos and 1 qvel.
  """
  entity = create_ball_joint_entity()
  entity, _ = initialize_entity(entity, device)

  # Entity has: ball1 (4 qpos, 3 dof), hinge1 (1 qpos, 1 dof), ball2 (4 qpos, 3 dof)
  # Total: 4 + 1 + 4 = 9 qpos, 3 + 1 + 3 = 7 dof
  assert entity.num_joints == 3
  assert entity.nq == 9, f"Expected nq=9, got {entity.nq}"
  assert entity.nv == 7, f"Expected nv=7, got {entity.nv}"

  # Check indexing fields match.
  assert entity.indexing.nq == 9
  assert entity.indexing.nv == 7


def test_nq_nv_hinge_only(device):
  """Test that nq and nv equal num_joints for hinge-only entities."""
  entity = create_hinge_only_entity()
  entity, _ = initialize_entity(entity, device)

  # Entity has: hinge1 (1 qpos, 1 dof), hinge2 (1 qpos, 1 dof)
  assert entity.num_joints == 2
  assert entity.nq == 2
  assert entity.nv == 2


def test_joint_offset_tensors(device):
  """Test that q_offsets and v_offsets are correctly computed."""
  entity = create_ball_joint_entity()
  entity, _ = initialize_entity(entity, device)

  indexing = entity.indexing

  # Expected cumulative offsets: ball1(4,3), hinge1(1,1), ball2(4,3)
  expected_q_offsets = torch.tensor([0, 4, 5], dtype=torch.int, device=device)
  expected_v_offsets = torch.tensor([0, 3, 4], dtype=torch.int, device=device)

  assert torch.equal(indexing.q_offsets, expected_q_offsets)
  assert torch.equal(indexing.v_offsets, expected_v_offsets)

  # Test expand_to_q_indices for single joint (ball1).
  q_indices = indexing.expand_to_q_indices(torch.tensor([0], device=device))
  assert isinstance(q_indices, torch.Tensor)
  assert torch.equal(q_indices, torch.tensor([0, 1, 2, 3], device=device))

  # Test expand_to_v_indices for single joint (ball1).
  v_indices = indexing.expand_to_v_indices(torch.tensor([0], device=device))
  assert isinstance(v_indices, torch.Tensor)
  assert torch.equal(v_indices, torch.tensor([0, 1, 2], device=device))

  # Test expand for hinge joint (joint 1).
  q_indices = indexing.expand_to_q_indices(torch.tensor([1], device=device))
  assert isinstance(q_indices, torch.Tensor)
  assert torch.equal(q_indices, torch.tensor([4], device=device))

  # Test expand for multiple joints (ball1, ball2).
  q_indices = indexing.expand_to_q_indices(torch.tensor([0, 2], device=device))
  assert isinstance(q_indices, torch.Tensor)
  assert torch.equal(q_indices, torch.tensor([0, 1, 2, 3, 5, 6, 7, 8], device=device))


def test_joint_qpos_widths(device):
  """Test that joint_qpos_widths and joint_dof_widths are correct."""
  entity = create_ball_joint_entity()
  entity, _ = initialize_entity(entity, device)

  indexing = entity.indexing

  # ball1, hinge1, ball2
  expected_qpos_widths = torch.tensor([4, 1, 4], dtype=torch.int, device=device)
  expected_dof_widths = torch.tensor([3, 1, 3], dtype=torch.int, device=device)

  assert torch.equal(indexing.joint_qpos_widths, expected_qpos_widths)
  assert torch.equal(indexing.joint_dof_widths, expected_dof_widths)


def test_ball_joint_initial_state(device):
  """Test that ball joints default to identity quaternion (1, 0, 0, 0)."""
  entity = create_ball_joint_entity()
  entity, _ = initialize_entity(entity, device)

  # Check default joint positions have correct shape.
  default_joint_pos = entity.data.default_joint_pos
  assert default_joint_pos.shape == (1, 9), (
    f"Expected shape (1, 9), got {default_joint_pos.shape}"
  )

  # Ball joint 1 (indices 0:4) should be identity quaternion.
  ball1_quat = default_joint_pos[0, 0:4]
  expected_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
  assert torch.allclose(ball1_quat, expected_identity, atol=1e-6)

  # Hinge joint (index 4) should be 0.
  hinge_pos = default_joint_pos[0, 4]
  assert abs(hinge_pos.item()) < 1e-6

  # Ball joint 2 (indices 5:9) should be identity quaternion.
  ball2_quat = default_joint_pos[0, 5:9]
  assert torch.allclose(ball2_quat, expected_identity, atol=1e-6)


def test_default_joint_vel_shape(device):
  """Test that default_joint_vel has correct shape (nv, not num_joints)."""
  entity = create_ball_joint_entity()
  entity, _ = initialize_entity(entity, device)

  default_joint_vel = entity.data.default_joint_vel
  assert default_joint_vel.shape == (1, 7), (
    f"Expected shape (1, 7), got {default_joint_vel.shape}"
  )


def test_joint_pos_target_shape(device):
  """Test that joint_pos_target has correct shape (nq) for non-actuated entity."""
  entity = create_ball_joint_entity()
  entity, _ = initialize_entity(entity, device)

  # Non-actuated entities have empty target tensors.
  joint_pos_target = entity.data.joint_pos_target
  assert joint_pos_target.shape == (1, 0), (
    f"Expected shape (1, 0) for non-actuated entity, got {joint_pos_target.shape}"
  )


def test_joint_vel_target_shape(device):
  """Test that joint_vel_target has correct shape (nv) for non-actuated entity."""
  entity = create_ball_joint_entity()
  entity, _ = initialize_entity(entity, device)

  # Non-actuated entities have empty target tensors.
  joint_vel_target = entity.data.joint_vel_target
  assert joint_vel_target.shape == (1, 0), (
    f"Expected shape (1, 0) for non-actuated entity, got {joint_vel_target.shape}"
  )


def test_write_joint_position_with_ball_joints(device):
  """Test writing joint positions with ball joints."""
  entity = create_ball_joint_entity()
  entity, sim = initialize_entity(entity, device)

  # Write all joint positions.
  # ball1 (4) + hinge1 (1) + ball2 (4) = 9 values
  new_pos = torch.tensor(
    [
      [
        0.7071,
        0.7071,
        0.0,
        0.0,  # ball1: 90 degree rotation around x
        0.5,  # hinge1
        1.0,
        0.0,
        0.0,
        0.0,  # ball2: identity
      ]
    ],
    device=device,
  )

  entity.data.write_joint_position(new_pos)
  sim.forward()

  # Read back and verify.
  joint_pos = entity.data.joint_pos
  assert torch.allclose(joint_pos, new_pos, atol=1e-4)


def test_write_joint_velocity_with_ball_joints(device):
  """Test writing joint velocities with ball joints."""
  entity = create_ball_joint_entity()
  entity, sim = initialize_entity(entity, device)

  # Write all joint velocities.
  # ball1 (3) + hinge1 (1) + ball2 (3) = 7 values
  new_vel = torch.tensor(
    [
      [
        0.1,
        0.2,
        0.3,  # ball1 angular velocity
        0.5,  # hinge1 velocity
        0.0,
        0.0,
        0.0,  # ball2 angular velocity
      ]
    ],
    device=device,
  )

  entity.data.write_joint_velocity(new_vel)
  sim.forward()

  # Read back and verify.
  joint_vel = entity.data.joint_vel
  assert torch.allclose(joint_vel, new_vel, atol=1e-4)


##
# MDP function tests with ball joints.
##


def test_mdp_joint_pos_rel_with_ball_joints(device):
  """Test joint_pos_rel observation with ball joints."""
  from unittest.mock import Mock

  from mjlab.envs.mdp import observations
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  entity = create_ball_joint_entity()
  entity, _ = initialize_entity(entity, device)

  env = Mock()
  env.scene = {"robot": entity}

  asset_cfg = SceneEntityCfg("robot", joint_ids=slice(None))

  # Get relative joint positions.
  result = observations.joint_pos_rel(env, biased=False, asset_cfg=asset_cfg)

  # Should have shape (1, nv=7) in DOF space, not (1, nq=9) in qpos space.
  # Ball joints contribute 3 DOF (axis-angle) instead of 4 qpos (quaternion).
  assert result.shape == (1, 7), f"Expected (1, 7), got {result.shape}"

  # At default position, all relative positions should be zero.
  assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


def test_mdp_joint_vel_rel_with_ball_joints(device):
  """Test joint_vel_rel observation with ball joints."""
  from unittest.mock import Mock

  from mjlab.envs.mdp import observations
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  entity = create_ball_joint_entity()
  entity, _ = initialize_entity(entity, device)

  env = Mock()
  env.scene = {"robot": entity}

  asset_cfg = SceneEntityCfg("robot", joint_ids=slice(None))

  # Get relative joint velocities.
  result = observations.joint_vel_rel(env, asset_cfg=asset_cfg)

  # Should have shape (1, nv=7), not (1, num_joints=3).
  assert result.shape == (1, 7), f"Expected (1, 7), got {result.shape}"


def test_mdp_joint_vel_l2_with_ball_joints(device):
  """Test joint_vel_l2 reward with ball joints."""
  from unittest.mock import Mock

  from mjlab.envs.mdp import rewards
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  entity = create_ball_joint_entity()
  entity, _ = initialize_entity(entity, device)

  env = Mock()
  env.scene = {"robot": entity}

  asset_cfg = SceneEntityCfg("robot", joint_ids=slice(None))

  # Compute joint velocity L2 penalty.
  result = rewards.joint_vel_l2(env, asset_cfg=asset_cfg)

  # Should return scalar per env.
  assert result.shape == (1,), f"Expected (1,), got {result.shape}"


def test_mdp_joint_acc_l2_with_ball_joints(device):
  """Test joint_acc_l2 reward with ball joints."""
  from unittest.mock import Mock

  from mjlab.envs.mdp import rewards
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  entity = create_ball_joint_entity()
  entity, _ = initialize_entity(entity, device)

  env = Mock()
  env.scene = {"robot": entity}

  asset_cfg = SceneEntityCfg("robot", joint_ids=slice(None))

  # Compute joint acceleration L2 penalty.
  result = rewards.joint_acc_l2(env, asset_cfg=asset_cfg)

  # Should return scalar per env.
  assert result.shape == (1,), f"Expected (1,), got {result.shape}"


def test_mdp_joint_pos_limits_with_ball_joints(device):
  """Test joint_pos_limits reward with ball joints."""
  from unittest.mock import Mock

  from mjlab.envs.mdp import rewards
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  entity = create_ball_joint_entity()
  entity, _ = initialize_entity(entity, device)

  env = Mock()
  env.device = device
  env.scene = {"robot": entity}

  asset_cfg = SceneEntityCfg("robot", joint_ids=slice(None))

  # Compute joint position limits penalty.
  result = rewards.joint_pos_limits(env, asset_cfg=asset_cfg)

  # Should return scalar per env.
  assert result.shape == (1,), f"Expected (1,), got {result.shape}"


def test_quaternion_difference_correctness(device):
  """Test that joint_pos_rel computes correct quaternion difference.

  For a 90-degree rotation around z-axis, the axis-angle representation
  should be [0, 0, pi/2] (rotation axis z, angle 90 degrees).
  """
  import math
  from unittest.mock import Mock

  from mjlab.envs.mdp import observations
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  entity = create_ball_joint_entity()
  entity, sim = initialize_entity(entity, device)

  # Apply a 90-degree rotation around z-axis to ball joint 1.
  # Quaternion for 90 deg around z: (cos(45°), 0, 0, sin(45°)) = (0.7071, 0, 0, 0.7071)
  angle = math.pi / 2
  half_angle = angle / 2
  new_pos = entity.data.joint_pos.clone()
  new_pos[0, 0:4] = torch.tensor(
    [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)], device=device
  )
  entity.data.write_joint_position(new_pos)
  sim.forward()

  env = Mock()
  env.scene = {"robot": entity}

  asset_cfg = SceneEntityCfg("robot", joint_ids=slice(None))
  result = observations.joint_pos_rel(env, biased=False, asset_cfg=asset_cfg)

  # Ball joint 1 (indices 0:3 in DOF space) should have axis-angle [0, 0, pi/2].
  ball1_diff = result[0, 0:3]
  expected_axis_angle = torch.tensor([0.0, 0.0, math.pi / 2], device=device)
  assert torch.allclose(ball1_diff, expected_axis_angle, atol=1e-4), (
    f"Expected axis-angle {expected_axis_angle}, got {ball1_diff}"
  )

  # Hinge joint (index 3 in DOF space) should be unchanged (0).
  assert abs(result[0, 3].item()) < 1e-6

  # Ball joint 2 (indices 4:7 in DOF space) should be unchanged (identity -> 0).
  ball2_diff = result[0, 4:7]
  assert torch.allclose(ball2_diff, torch.zeros(3, device=device), atol=1e-6)
