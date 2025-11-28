"""Tests for manager config immutability.

Managers should not mutate the original config objects passed to them.
This allows the same config to be reused across multiple manager instantiations.
"""

import inspect
from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import Entity, EntityCfg
from mjlab.managers.manager_term_config import (
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
)
from mjlab.managers.observation_manager import ObservationManager
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture
def device():
  return get_test_device()


@pytest.fixture
def mock_env(device):
  env = Mock()
  env.num_envs = 2
  env.device = device
  env.step_dt = 0.02
  env.max_episode_length_s = 10.0
  env.scene = {}
  return env


class ClassObsTerm:
  """Class-based observation term that triggers func mutation."""

  def __init__(self, cfg, env):
    self.device = env.device

  def __call__(self, env):
    return torch.zeros((env.num_envs, 3), device=self.device)


class ClassRewardTerm:
  """Class-based reward term that triggers func mutation."""

  def __init__(self, cfg, env):
    self.device = env.device

  def __call__(self, env):
    return torch.ones(env.num_envs, device=self.device)


def test_manager_preserves_class_func(mock_env):
  """Managers should not mutate class-based func to instance.

  When term_cfg.func is a class, _resolve_common_term_cfg instantiates it.
  Without copying term_cfg first, this mutates the original config.
  """
  term_cfg = ObservationTermCfg(func=ClassObsTerm, params={})

  assert inspect.isclass(term_cfg.func), "precondition: func should be a class"

  cfg = {"policy": ObservationGroupCfg(terms={"obs1": term_cfg})}
  _ = ObservationManager(cfg, mock_env)

  assert inspect.isclass(term_cfg.func), "func was mutated from class to instance"


def test_observation_shared_terms_between_groups(mock_env):
  """Terms shared between policy and critic groups get independent settings."""
  term_cfg = ObservationTermCfg(func=ClassObsTerm, params={})

  cfg = {
    "policy": ObservationGroupCfg(terms={"obs": term_cfg}, history_length=5),
    "critic": ObservationGroupCfg(terms={"obs": term_cfg}),  # no history
  }

  manager = ObservationManager(cfg, mock_env)

  policy_obs = manager.compute()["policy"]
  critic_obs = manager.compute()["critic"]

  # Policy has history (5 * 3 = 15), critic doesn't (3).
  assert isinstance(policy_obs, torch.Tensor)
  assert isinstance(critic_obs, torch.Tensor)
  assert policy_obs.shape[1] == 15  # 3 features * 5 history
  assert critic_obs.shape[1] == 3  # 3 features, no history


def test_scene_entity_cfg_in_params_not_mutated(device):
  """SceneEntityCfg in params should not be mutated when resolved."""
  # NOTE: 2 joints so selecting one won't optimize to slice(None).
  xml = """
  <mujoco>
    <worldbody>
      <body name="body1" pos="0 0 1">
        <joint name="joint1" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <body name="body2" pos="0.2 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  """
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))
  entity = Entity(entity_cfg)
  model = entity.compile()

  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)

  env = Mock()
  env.num_envs = 2
  env.device = device
  env.max_episode_length_s = 10.0
  env.scene = {"robot": entity}

  # Select only joint1 --> joint_ids will mutate from slice(None) to [0].
  asset_cfg = SceneEntityCfg(name="robot", joint_names=("joint1",))
  original_joint_ids = asset_cfg.joint_ids  # slice(None) before resolve

  def reward_func(env, asset_cfg):
    del asset_cfg  # Unused.
    return torch.ones(env.num_envs, device=env.device)

  term_cfg = RewardTermCfg(
    func=reward_func, params={"asset_cfg": asset_cfg}, weight=1.0
  )

  _ = RewardManager({"reward1": term_cfg}, env)

  assert asset_cfg.joint_ids == original_joint_ids, (
    f"SceneEntityCfg.joint_ids was mutated from {original_joint_ids} to "
    f"{asset_cfg.joint_ids}"
  )
