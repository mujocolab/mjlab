"""Tests for MDP events functionality."""

from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab import actuator
from mjlab.envs.mdp import events
from mjlab.managers.event_manager import EventManager
from mjlab.managers.manager_term_config import EventTermCfg
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
  cfg = {
    # Class-based DR term should be tracked.
    "custom_dr": EventTermCfg(
      mode="startup",
      func=CustomRandomizer,
      domain_randomization=True,
      params={"field": "geom_friction", "ranges": (0.3, 1.2)},
    ),
    # Regular function-based DR term should be tracked.
    "standard_dr": EventTermCfg(
      mode="reset",
      func=events.randomize_field,
      domain_randomization=True,
      params={"field": "body_mass", "ranges": (0.8, 1.2)},
    ),
    # Non-DR term should not be tracked.
    "regular_event": EventTermCfg(
      mode="reset",
      func=events.reset_joints_by_offset,
      params={"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0)},
    ),
  }

  manager = EventManager(cfg, env)

  # Verify that DR fields are tracked.
  assert "geom_friction" in manager.domain_randomization_fields
  assert "body_mass" in manager.domain_randomization_fields

  # Verify that non-DR event is not tracked.
  assert len(manager.domain_randomization_fields) == 2


def test_randomize_pd_gains(device):
  """Test PD gain randomization."""
  env = Mock()
  env.num_envs = 2
  env.device = device

  mock_entity = Mock()

  builtin_actuator = Mock(spec=actuator.BuiltinPositionActuator)
  builtin_actuator.ctrl_ids = torch.tensor([0, 1], device=device)

  xml_actuator = Mock(spec=actuator.XmlPositionActuator)
  xml_actuator.ctrl_ids = torch.tensor([2, 3], device=device)

  ideal_actuator = Mock(spec=actuator.IdealPdActuator)
  ideal_actuator.ctrl_ids = torch.tensor([4, 5], device=device)
  ideal_actuator.stiffness = torch.tensor(
    [[100.0, 100.0], [100.0, 100.0]], device=device
  )
  ideal_actuator.damping = torch.tensor([[10.0, 10.0], [10.0, 10.0]], device=device)

  ideal_actuator.set_gains = actuator.IdealPdActuator.set_gains.__get__(ideal_actuator)

  mock_entity.actuators = [builtin_actuator, xml_actuator, ideal_actuator]
  env.scene = {"robot": mock_entity}

  env.sim = Mock()
  env.sim.model = Mock()
  env.sim.model.actuator_gainprm = torch.ones((2, 6, 10), device=device) * 50.0
  env.sim.model.actuator_biasprm = torch.zeros((2, 6, 10), device=device)
  env.sim.model.actuator_biasprm[:, :, 1] = -50.0  # -Kp
  env.sim.model.actuator_biasprm[:, :, 2] = -5.0  # -Kd

  # Test scale operation.
  events.randomize_pd_gains(
    env,
    torch.tensor([0], device=device),
    kp_range=(1.5, 1.5),  # 1.5x scale
    kd_range=(2.0, 2.0),  # 2.0x scale
    asset_cfg=SceneEntityCfg("robot"),
    distribution="uniform",
    operation="scale",
  )

  assert torch.allclose(
    env.sim.model.actuator_gainprm[0, [0, 1], 0],
    torch.tensor([75.0, 75.0], device=device),
  )
  assert torch.allclose(
    env.sim.model.actuator_biasprm[0, [0, 1], 1],
    torch.tensor([-75.0, -75.0], device=device),
  )
  assert torch.allclose(
    env.sim.model.actuator_biasprm[0, [0, 1], 2],
    torch.tensor([-10.0, -10.0], device=device),
  )

  assert torch.allclose(
    env.sim.model.actuator_gainprm[0, [2, 3], 0],
    torch.tensor([75.0, 75.0], device=device),
  )
  assert torch.allclose(
    env.sim.model.actuator_biasprm[0, [2, 3], 1],
    torch.tensor([-75.0, -75.0], device=device),
  )
  assert torch.allclose(
    env.sim.model.actuator_biasprm[0, [2, 3], 2],
    torch.tensor([-10.0, -10.0], device=device),
  )

  assert torch.allclose(
    ideal_actuator.stiffness[0],
    torch.tensor([150.0, 150.0], device=device),
  )
  assert torch.allclose(
    ideal_actuator.damping[0],
    torch.tensor([20.0, 20.0], device=device),
  )

  # Test abs operation.
  events.randomize_pd_gains(
    env,
    torch.tensor([1], device=device),
    kp_range=(200.0, 200.0),
    kd_range=(25.0, 25.0),
    asset_cfg=SceneEntityCfg("robot"),
    distribution="uniform",
    operation="abs",
  )

  assert torch.allclose(
    env.sim.model.actuator_gainprm[1, [0, 1], 0],
    torch.tensor([200.0, 200.0], device=device),
  )
  assert torch.allclose(
    env.sim.model.actuator_biasprm[1, [0, 1], 1],
    torch.tensor([-200.0, -200.0], device=device),
  )
  assert torch.allclose(
    env.sim.model.actuator_biasprm[1, [0, 1], 2],
    torch.tensor([-25.0, -25.0], device=device),
  )

  assert torch.allclose(
    env.sim.model.actuator_gainprm[1, [2, 3], 0],
    torch.tensor([200.0, 200.0], device=device),
  )
  assert torch.allclose(
    env.sim.model.actuator_biasprm[1, [2, 3], 1],
    torch.tensor([-200.0, -200.0], device=device),
  )
  assert torch.allclose(
    env.sim.model.actuator_biasprm[1, [2, 3], 2],
    torch.tensor([-25.0, -25.0], device=device),
  )

  assert torch.allclose(
    ideal_actuator.stiffness[1],
    torch.tensor([200.0, 200.0], device=device),
  )
  assert torch.allclose(
    ideal_actuator.damping[1],
    torch.tensor([25.0, 25.0], device=device),
  )


def test_randomize_effort_limits(device):
  """Test effort limit randomization."""
  env = Mock()
  env.num_envs = 2
  env.device = device

  mock_entity = Mock()

  builtin_actuator = Mock(spec=actuator.BuiltinPositionActuator)
  builtin_actuator.ctrl_ids = torch.tensor([0, 1], device=device)

  xml_actuator = Mock(spec=actuator.XmlPositionActuator)
  xml_actuator.ctrl_ids = torch.tensor([2, 3], device=device)

  ideal_actuator = Mock(spec=actuator.IdealPdActuator)
  ideal_actuator.ctrl_ids = torch.tensor([4, 5], device=device)
  ideal_actuator.force_limit = torch.tensor([[50.0, 50.0], [50.0, 50.0]], device=device)

  ideal_actuator.set_effort_limit = actuator.IdealPdActuator.set_effort_limit.__get__(
    ideal_actuator
  )

  mock_entity.actuators = [builtin_actuator, xml_actuator, ideal_actuator]
  env.scene = {"robot": mock_entity}

  env.sim = Mock()
  env.sim.model = Mock()
  env.sim.model.actuator_forcerange = torch.zeros((2, 6, 2), device=device)
  env.sim.model.actuator_forcerange[:, :, 0] = -100.0  # Lower limit
  env.sim.model.actuator_forcerange[:, :, 1] = 100.0  # Upper limit

  # Test scale operation.
  events.randomize_effort_limits(
    env,
    torch.tensor([0], device=device),
    effort_limit_range=(2.0, 2.0),  # 2x scale
    asset_cfg=SceneEntityCfg("robot"),
    distribution="uniform",
    operation="scale",
  )

  assert torch.allclose(
    env.sim.model.actuator_forcerange[0, [0, 1], 0],
    torch.tensor([-200.0, -200.0], device=device),
  )
  assert torch.allclose(
    env.sim.model.actuator_forcerange[0, [0, 1], 1],
    torch.tensor([200.0, 200.0], device=device),
  )

  assert torch.allclose(
    env.sim.model.actuator_forcerange[0, [2, 3], 0],
    torch.tensor([-200.0, -200.0], device=device),
  )
  assert torch.allclose(
    env.sim.model.actuator_forcerange[0, [2, 3], 1],
    torch.tensor([200.0, 200.0], device=device),
  )

  assert torch.allclose(
    ideal_actuator.force_limit[0],
    torch.tensor([100.0, 100.0], device=device),
  )

  # Test abs operation.
  events.randomize_effort_limits(
    env,
    torch.tensor([1], device=device),
    effort_limit_range=(150.0, 150.0),
    asset_cfg=SceneEntityCfg("robot"),
    distribution="uniform",
    operation="abs",
  )

  assert torch.allclose(
    env.sim.model.actuator_forcerange[1, [0, 1], 0],
    torch.tensor([-150.0, -150.0], device=device),
  )
  assert torch.allclose(
    env.sim.model.actuator_forcerange[1, [0, 1], 1],
    torch.tensor([150.0, 150.0], device=device),
  )

  assert torch.allclose(
    env.sim.model.actuator_forcerange[1, [2, 3], 0],
    torch.tensor([-150.0, -150.0], device=device),
  )
  assert torch.allclose(
    env.sim.model.actuator_forcerange[1, [2, 3], 1],
    torch.tensor([150.0, 150.0], device=device),
  )

  assert torch.allclose(
    ideal_actuator.force_limit[1],
    torch.tensor([150.0, 150.0], device=device),
  )
