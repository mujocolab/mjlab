"""Tests for RslRlVecEnvWrapper."""

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg


@pytest.fixture
def env():
  robot_xml = """
  <mujoco>
    <worldbody>
      <body name="base" pos="0 0 1">
        <freejoint name="free_joint"/>
        <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
        <body name="link1" pos="0 0 0">
          <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
          <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
        </body>
      </body>
    </worldbody>
    <actuator>
      <motor name="actuator1" joint="joint1" gear="1.0"/>
    </actuator>
  </mujoco>
  """
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=2,
      extent=1.0,
      entities={"robot": robot_cfg},
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "joint_pos": ObservationTermCfg(
            func=lambda env: env.scene["robot"].data.joint_pos
          ),
        },
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    # Episode-level reward metric written only on reset, never during compute().
    rewards={
      "const": RewardTermCfg(
        func=lambda env: torch.ones(env.num_envs, device=env.device), weight=1.0
      )
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=1,
    # Long episode with no terminations so the first step never resets.
    episode_length_s=100.0,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=get_test_device())
  yield env
  env.close()


def test_reset_only_log_keys_survive_nonreset_steps(env):
  """Keys written only on reset must keep appearing on non-reset steps.

  rsl_rl's logger derives its key set from the first rollout step, which
  typically has no resets. The wrapper caches the initial reset's keys and
  carries them forward so they remain in ``extras["log"]`` on every step.
  """
  wrapper = RslRlVecEnvWrapper(env)
  actions = torch.zeros(wrapper.num_envs, wrapper.num_actions, device=wrapper.device)
  _, _, dones, extras = wrapper.step(actions)

  assert not dones.any(), "test assumes no reset on the first step"
  assert "Episode_Reward/const" in extras["log"]
