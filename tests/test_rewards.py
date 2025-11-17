"""Tests for reward manager functionality."""

from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.rewards import electrical_power_cost
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg

PARTIALLY_ACTUATED_ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="base_geom" type="cylinder" size="0.1 0.05" mass="1.0"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="actuated_joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="link1_geom" type="box" size="0.05 0.05 0.2" mass="0.5"/>
        <body name="link2" pos="0 0 0.4">
          <joint name="passive_joint" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
          <geom name="link2_geom" type="box" size="0.05 0.05 0.15" mass="0.3"/>
          <body name="link3" pos="0 0 0.3">
            <joint name="actuated_joint2" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
            <geom name="link3_geom" type="box" size="0.05 0.05 0.1" mass="0.2"/>
          </body>
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


class SimpleTestReward:
  """A simple class-based reward for testing that tracks state."""

  def __init__(self, cfg: RewardTermCfg, env):
    self.num_envs = env.num_envs
    self.device = env.device
    self.current_air_time = torch.zeros((self.num_envs, 1), device=self.device)

  def __call__(self, env, **kwargs):
    self.current_air_time += 0.01
    return torch.ones(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor | None = None, env=None):
    if env_ids is not None and len(env_ids) > 0:
      self.current_air_time[env_ids] = 0


class StatelessReward:
  """A stateless class-based reward without reset method."""

  def __init__(self, cfg: RewardTermCfg, env):
    pass

  def __call__(self, env, **kwargs):
    return torch.ones(env.num_envs)


@pytest.fixture
def mock_env():
  """Create a mock environment for testing."""
  env = Mock()
  env.num_envs = 4
  env.device = "cpu"
  env.step_dt = 0.01
  env.max_episode_length_s = 10.0
  robot = Mock()
  env.scene = {"robot": robot}
  env.command_manager.get_command = Mock(
    return_value=torch.tensor([[1.0, 0.0, 0.0]] * 4)
  )
  return env


@pytest.fixture
def class_reward_config():
  """Config with a class-based reward."""
  return {
    "term": RewardTermCfg(
      func=SimpleTestReward,
      weight=1.0,
      params={},
    )
  }


@pytest.fixture
def function_reward_config():
  """Config with a function-based reward."""
  return {
    "term": RewardTermCfg(
      func=lambda env: torch.ones(env.num_envs),
      weight=1.0,
      params={},
    )
  }


@pytest.fixture
def stateless_reward_config():
  """Config with a stateless class-based reward."""
  return {
    "term": RewardTermCfg(
      func=StatelessReward,
      weight=1.0,
      params={},
    )
  }


def test_class_based_reward_reset(mock_env, class_reward_config):
  """Test that class-based reward terms are tracked and have reset called."""
  manager = RewardManager(class_reward_config, mock_env)
  term = manager._class_term_cfgs[0].func

  for _ in range(10):
    manager.compute(dt=0.01)
  assert (term.current_air_time > 0).all()

  manager.reset(env_ids=torch.tensor([0, 2]))

  # Check that only specified envs were reset.
  assert term.current_air_time[0, 0] == 0
  assert term.current_air_time[1, 0] > 0
  assert term.current_air_time[2, 0] == 0
  assert term.current_air_time[3, 0] > 0


def test_function_based_reward_not_tracked(mock_env, function_reward_config):
  """Test that function-based reward terms are not tracked as class terms."""
  manager = RewardManager(function_reward_config, mock_env)
  assert len(manager._class_term_cfgs) == 0


def test_stateless_class_reward_no_reset(mock_env, stateless_reward_config):
  """Test that stateless class-based rewards without reset don't break reset."""
  manager = RewardManager(stateless_reward_config, mock_env)

  # Stateless rewards without reset method should not be tracked.
  assert len(manager._class_term_cfgs) == 0

  # Reset should work without errors.
  manager.reset(env_ids=torch.tensor([0, 2]))


def test_electrical_power_cost_partially_actuated(device):
  """Test electrical_power_cost on robots with passive joints.

  This test verifies that:
  1. The reward correctly identifies and uses only actuated joints (ignoring passive joints)
  2. Power cost is computed as sum of max(0, force*velocity) for each actuator,
     meaning only positive work is penalized (not regenerative braking)
  """
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(PARTIALLY_ACTUATED_ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          joint_names_expr=("actuated_joint1", "actuated_joint2"),
          effort_limit=10.0,
          stiffness=100.0,
          damping=10.0,
        ),
      )
    ),
  )
  entity = Entity(entity_cfg)
  model = entity.compile()

  num_envs = 2
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)

  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.scene = {"robot": entity}

  asset_cfg = SceneEntityCfg(name="robot", joint_names=("actuated.*",))

  reward_cfg = RewardTermCfg(
    func=electrical_power_cost, weight=1.0, params={"asset_cfg": asset_cfg}
  )

  reward = electrical_power_cost(reward_cfg, env)

  assert len(reward._joint_ids) == 2
  assert len(reward._actuator_ids) == 2

  # Test case 1: All forces and velocities aligned (all positive work).
  # actuated_joint1 (qvel[:, 0]), actuated_joint2 (qvel[:, 2]).
  # Note: qvel[:, 1] is the passive joint, not used in power calculation.
  sim.data.actuator_force[:] = torch.tensor(
    [[2.0, 3.0], [1.0, 4.0]], device=device, dtype=torch.float32
  )
  sim.data.qvel[:] = 0.0
  sim.data.qvel[:, 0] = torch.tensor([1.0, 2.0], device=device)
  sim.data.qvel[:, 2] = torch.tensor([2.0, 1.0], device=device)

  power_cost = reward(env, asset_cfg)

  # Expected: env0 = 2.0*1.0 + 3.0*2.0 = 8.0, env1 = 1.0*2.0 + 4.0*1.0 = 6.0.
  expected = torch.tensor([8.0, 6.0], device=device)
  assert torch.allclose(power_cost, expected)

  # Test case 2: Some negative forces (regenerative braking, not penalized).
  sim.data.actuator_force[:] = torch.tensor(
    [[-2.0, 3.0], [1.0, -4.0]], device=device, dtype=torch.float32
  )

  power_cost = reward(env, asset_cfg)

  # Expected: env0 = max(0,-2.0*1.0) + max(0,3.0*2.0) = 0 + 6.0 = 6.0.
  #           env1 = max(0,1.0*2.0) + max(0,-4.0*1.0) = 2.0 + 0 = 2.0.
  expected = torch.tensor([6.0, 2.0], device=device)
  assert torch.allclose(power_cost, expected)
