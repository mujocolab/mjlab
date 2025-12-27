"""Tests for encoder bias simulation.

We simulate this by storing `encoder_bias` per joint in EntityData:
  - Observations: policy sees (true_position + bias)
  - Actions: position commands are converted via (command - bias) before simulation

This ensures joint limits apply to the true physical position, not biased readings.
"""

from collections.abc import Callable
from functools import partial

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.manager_term_config import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg

# =============================================================================
# Test fixtures and helpers
# =============================================================================

SLIDING_MASS_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="mass" pos="0 0 0">
      <joint name="slide" type="slide" axis="1 0 0" range="-1 1" limited="true"/>
      <geom name="mass_geom" type="sphere" size="0.1" mass="1.0"/>
    </body>
  </worldbody>
  <sensor>
    <jointpos name="slide_pos" joint="slide"/>
  </sensor>
</mujoco>
"""


def _make_robot_cfg():
  return EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(SLIDING_MASS_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=(".*",), stiffness=1000.0, damping=100.0
        ),
      )
    ),
  )


def _make_env_cfg(
  obs_func: Callable | None = None,
  num_envs: int = 2,
  events: dict | None = None,
) -> ManagerBasedRlEnvCfg:
  if obs_func is None:
    obs_func = partial(mdp.joint_pos_rel, biased=True)
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),
      num_envs=num_envs,
      extent=1.0,
      entities={"robot": _make_robot_cfg()},
    ),
    observations={
      "policy": ObservationGroupCfg(
        terms={"obs": ObservationTermCfg(func=obs_func)},
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    events=events or {},
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002, iterations=1)),
    decimation=1,
    episode_length_s=10.0,
  )


@pytest.fixture(scope="module")
def device():
  return get_test_device()


# =============================================================================
# EntityData.encoder_bias field tests
# =============================================================================


def test_encoder_bias_initialized_to_zero(device):
  """encoder_bias should be zeros with shape (num_envs, num_joints)."""
  env = ManagerBasedRlEnv(cfg=_make_env_cfg(num_envs=4), device=device)
  env.reset()

  robot = env.scene["robot"]
  assert robot.data.encoder_bias.shape == (4, 1)
  assert (robot.data.encoder_bias == 0).all()

  env.close()


def test_encoder_bias_can_be_set_per_env(device):
  """encoder_bias can be set differently per environment."""
  env = ManagerBasedRlEnv(cfg=_make_env_cfg(), device=device)
  env.reset()

  robot = env.scene["robot"]
  robot.data.encoder_bias[0, 0] = 0.1
  robot.data.encoder_bias[1, 0] = -0.2

  assert robot.data.encoder_bias[0, 0].item() == pytest.approx(0.1)
  assert robot.data.encoder_bias[1, 0].item() == pytest.approx(-0.2)

  env.close()


def test_joint_pos_biased_equals_joint_pos_plus_bias(device):
  """joint_pos_biased property should return joint_pos + encoder_bias."""
  env = ManagerBasedRlEnv(cfg=_make_env_cfg(), device=device)
  env.reset()

  robot = env.scene["robot"]
  robot.data.encoder_bias[:, 0] = 0.25

  torch.testing.assert_close(
    robot.data.joint_pos_biased, robot.data.joint_pos + robot.data.encoder_bias
  )

  env.close()


# =============================================================================
# Observation tests
# =============================================================================


def test_joint_pos_rel_includes_encoder_bias(device):
  """joint_pos_rel observation should include encoder_bias."""
  env = ManagerBasedRlEnv(cfg=_make_env_cfg(), device=device)
  env.reset()

  robot = env.scene["robot"]
  bias = 0.5

  obs_before = env.observation_manager.compute()["policy"]
  assert isinstance(obs_before, torch.Tensor)
  obs_before = obs_before.clone()
  robot.data.encoder_bias[:, 0] = bias
  env.observation_manager._obs_buffer = None  # Invalidate cache.
  obs_after = env.observation_manager.compute()["policy"]

  # Observation should increase by bias amount.
  torch.testing.assert_close(obs_after, obs_before + bias, atol=1e-5, rtol=0)

  env.close()


def test_joint_vel_rel_ignores_encoder_bias(device):
  """joint_vel_rel should NOT include encoder_bias (bias is constant, d/dt = 0)."""
  env = ManagerBasedRlEnv(cfg=_make_env_cfg(obs_func=mdp.joint_vel_rel), device=device)
  env.reset()

  robot = env.scene["robot"]

  obs_before = env.observation_manager.compute()["policy"]
  assert isinstance(obs_before, torch.Tensor)
  obs_before = obs_before.clone()
  robot.data.encoder_bias[:, 0] = 0.5
  env.observation_manager._obs_buffer = None
  obs_after = env.observation_manager.compute()["policy"]

  torch.testing.assert_close(obs_before, obs_after, atol=1e-6, rtol=0)

  env.close()


# =============================================================================
# Action tests
# =============================================================================


def test_position_action_subtracts_encoder_bias(device):
  """JointPositionAction should subtract encoder_bias from commands."""
  env = ManagerBasedRlEnv(cfg=_make_env_cfg(), device=device)
  env.reset()

  robot = env.scene["robot"]
  robot.data.encoder_bias[0, 0] = 0.0
  robot.data.encoder_bias[1, 0] = 0.3

  # Same command to both envs.
  env.step(torch.tensor([[0.5], [0.5]], device=device))

  # Actual targets should differ by bias.
  assert robot.data.joint_pos_target[0, 0].item() == pytest.approx(0.5, abs=1e-5)
  assert robot.data.joint_pos_target[1, 0].item() == pytest.approx(0.2, abs=1e-5)

  env.close()


# =============================================================================
# End-to-end consistency test
# =============================================================================


def test_bias_compensation_produces_identical_physical_behavior(device):
  """Envs with different biases should have identical physics when compensated.

  If env0 has bias=0 and env1 has bias=0.3, and we want both to reach physical
  position 0.4, we command:
    - env0: 0.4 (in encoder frame)
    - env1: 0.7 (in encoder frame, which becomes 0.4 after bias subtraction)

  Both should reach the same physical position. Their observations should differ
  by the bias amount.
  """
  env = ManagerBasedRlEnv(cfg=_make_env_cfg(), device=device)
  env.reset()

  robot = env.scene["robot"]
  bias_env0, bias_env1 = 0.0, 0.3
  robot.data.encoder_bias[0, 0] = bias_env0
  robot.data.encoder_bias[1, 0] = bias_env1

  # Command same physical target via encoder frame.
  target_physical = 0.4
  action = torch.tensor(
    [[target_physical + bias_env0], [target_physical + bias_env1]], device=device
  )

  for _ in range(100):
    env.step(action)

  # Physical positions should match.
  pos_env0 = robot.data.joint_pos[0, 0].item()
  pos_env1 = robot.data.joint_pos[1, 0].item()
  assert pos_env0 == pytest.approx(pos_env1, abs=1e-4)

  # Observations should differ by bias.
  env.observation_manager._obs_buffer = None
  obs = env.observation_manager.compute()["policy"]
  assert isinstance(obs, torch.Tensor)
  obs_diff = obs[1, 0].item() - obs[0, 0].item()
  assert obs_diff == pytest.approx(bias_env1 - bias_env0, abs=1e-4)

  env.close()


# =============================================================================
# Event randomization test
# =============================================================================


def test_randomize_encoder_bias_event(device):
  """randomize_encoder_bias should sample values within specified range."""
  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),
      num_envs=100,
      extent=10.0,
      entities={"robot": _make_robot_cfg()},
    ),
    observations={
      "policy": ObservationGroupCfg(
        terms={"obs": ObservationTermCfg(func=partial(mdp.joint_pos_rel, biased=True))},
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    events={
      "randomize_bias": EventTermCfg(
        func=mdp.randomize_encoder_bias,
        mode="startup",
        params={"bias_range": (-0.1, 0.1), "asset_cfg": SceneEntityCfg("robot")},
      )
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002, iterations=1)),
    decimation=1,
    episode_length_s=10.0,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env.reset()

  biases = env.scene["robot"].data.encoder_bias[:, 0]
  assert (biases >= -0.1).all() and (biases <= 0.1).all()
  assert biases.std() > 0.01  # Not all identical.

  env.close()
