"""Tests for configurable manager classes on ManagerBasedRlEnvCfg (#1177)."""

import mujoco
import pytest
import torch
import tyro
from conftest import get_test_device

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.envs.manager_based_rl_env import _resolve_manager_class
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

SIMPLE_ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="box" type="box" size="0.1 0.1 0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


class CustomMultiRewardManager(RewardManager):
  """Custom reward manager subclass exposing a multi-column reward tensor."""

  def compute(self, dt: float) -> torch.Tensor:
    scalar = super().compute(dt)
    return torch.stack([scalar, self.term_rewards.sum(dim=-1)], dim=-1)


def _make_env_cfg(**kwargs) -> ManagerBasedRlEnvCfg:
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(SIMPLE_ROBOT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("slide",), stiffness=20.0, damping=2.0
        ),
      )
    ),
  )
  return ManagerBasedRlEnvCfg(
    decimation=1,
    scene=SceneCfg(
      num_envs=2,
      extent=1.0,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "qpos": ObservationTermCfg(
            func=lambda env: env.scene["robot"].data.joint_pos
          ),
        }
      )
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=("slide",), scale=1.0
      )
    },
    rewards={
      "alive": RewardTermCfg(
        func=lambda env: torch.ones(env.num_envs, device=env.device), weight=2.0
      )
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    episode_length_s=1.0,
    **kwargs,
  )


def test_custom_reward_manager_class_direct():
  """ManagerBasedRlEnv instantiates a custom RewardManager subclass passed directly."""
  device = get_test_device()
  cfg = _make_env_cfg(reward_manager_class=CustomMultiRewardManager)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)

  assert isinstance(env.reward_manager, CustomMultiRewardManager)
  env.reset()
  _, rew, _, _, _ = env.step(torch.zeros((2, 1), device=device))
  assert rew.shape == (2, 2)
  env.close()


def test_custom_reward_manager_class_import_string():
  """ManagerBasedRlEnv resolves 'module:ClassName' import strings for manager classes."""
  device = get_test_device()
  import_path = f"{__name__}:CustomMultiRewardManager"
  cfg = _make_env_cfg(reward_manager_class=import_path)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)

  assert isinstance(env.reward_manager, CustomMultiRewardManager)
  env.close()


def test_manager_class_cli_override_via_tyro():
  """tyro CLI can override manager class fields with an import path string."""
  cfg = _make_env_cfg()
  parsed = tyro.cli(
    ManagerBasedRlEnvCfg,
    default=cfg,
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
    args=["--reward-manager-class", f"{__name__}:CustomMultiRewardManager"],
  )
  assert parsed.reward_manager_class == f"{__name__}:CustomMultiRewardManager"


def test_resolve_manager_class_rejects_invalid_inputs():
  """_resolve_manager_class rejects malformed paths and wrong base classes."""
  with pytest.raises(ValueError, match="Invalid manager import path"):
    _resolve_manager_class("not_a_module_path", RewardManager)

  class Unrelated:
    pass

  with pytest.raises(TypeError, match="RewardManager"):
    _resolve_manager_class(Unrelated, RewardManager)
