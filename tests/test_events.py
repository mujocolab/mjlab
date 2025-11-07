"""Tests for MDP events functionality."""

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.envs.mdp import events
from mjlab.managers.event_manager import EventManager
from mjlab.managers.manager_term_config import EventTermCfg, term
from mjlab.managers.scene_entity_config import SceneEntityCfg


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


def test_reset_joints_by_offset(device):
  """Test that reset_joints_by_offset applies offsets and respects joint limits."""
  env = Mock()
  env.num_envs = 2
  env.device = device

  mock_entity = Mock()
  mock_entity.data.default_joint_pos = torch.zeros((2, 3), device=device)
  mock_entity.data.default_joint_vel = torch.zeros((2, 3), device=device)
  mock_entity.data.soft_joint_pos_limits = torch.tensor(
    [
      [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
      [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
    ],
    device=device,
  )
  mock_entity.write_joint_state_to_sim = Mock()

  env.scene = {"robot": mock_entity}

  # Test normal offset application.
  events.reset_joints_by_offset(
    env,
    torch.tensor([0], device=device),
    position_range=(0.3, 0.3),
    velocity_range=(0.2, 0.2),
    asset_cfg=SceneEntityCfg("robot", joint_ids=slice(None)),
  )

  call_args = mock_entity.write_joint_state_to_sim.call_args
  joint_pos, joint_vel = call_args[0][0], call_args[0][1]
  assert torch.allclose(joint_pos, torch.ones_like(joint_pos) * 0.3)
  assert torch.allclose(joint_vel, torch.ones_like(joint_vel) * 0.2)

  # Test clamping when offset exceeds limits.
  events.reset_joints_by_offset(
    env,
    torch.tensor([1], device=device),
    position_range=(1.0, 1.0),
    velocity_range=(0.0, 0.0),
    asset_cfg=SceneEntityCfg("robot", joint_ids=slice(None)),
  )

  call_args = mock_entity.write_joint_state_to_sim.call_args
  joint_pos = call_args[0][0]
  assert torch.allclose(joint_pos, torch.ones_like(joint_pos) * 0.5)


def test_class_based_event_with_domain_randomization(device):
  """Test that class-based events work and domain_randomization flag tracks fields."""

  # Create a simple class-based event term.
  class CustomRandomizer:
    def __init__(self, cfg, env):
      self.cfg = cfg
      self.env = env

    def __call__(self, env, env_ids, field, ranges):
      pass  # No-op for testing

  # Create a mock environment with minimal requirements.
  env = Mock()
  env.num_envs = 4
  env.device = device
  env.scene = {}
  env.sim = Mock()

  # Create event manager config with both DR and non-DR terms.
  @dataclass
  class EventCfg:
    # Class-based DR term should be tracked.
    custom_dr: EventTermCfg = term(
      EventTermCfg,
      mode="startup",
      func=CustomRandomizer,
      domain_randomization=True,
      params={"field": "geom_friction", "ranges": (0.3, 1.2)},
    )
    # Regular function-based DR term should be tracked.
    standard_dr: EventTermCfg = term(
      EventTermCfg,
      mode="reset",
      func=events.randomize_field,
      domain_randomization=True,
      params={"field": "body_mass", "ranges": (0.8, 1.2)},
    )
    # Non-DR term should not be tracked.
    regular_event: EventTermCfg = term(
      EventTermCfg,
      mode="reset",
      func=events.reset_joints_by_offset,
      params={"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0)},
    )

  cfg = EventCfg()
  manager = EventManager(cfg, env)

  # Verify that DR fields are tracked.
  assert "geom_friction" in manager.domain_randomization_fields
  assert "body_mass" in manager.domain_randomization_fields

  # Verify that non-DR event is not tracked.
  assert len(manager.domain_randomization_fields) == 2
