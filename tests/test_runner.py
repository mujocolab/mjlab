"""Tests for MjlabOnPolicyRunner."""

import tempfile
from dataclasses import asdict
from pathlib import Path

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture
def env(device):
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
      actuators=(XmlMotorActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),
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
      "critic": ObservationGroupCfg(
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
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=1,
    episode_length_s=1.0,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  yield env
  env.close()


def test_runner_persists_common_step_counter(env, device, monkeypatch):
  """MjlabOnPolicyRunner should save and restore common_step_counter."""
  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )
    monkeypatch.setattr(runner.logger, "save_model", lambda *args, **kwargs: None)
    runner.logger.logger_type = "tensorboard"  # Normally set in learn().

    wrapped_env.unwrapped.common_step_counter = 12345
    checkpoint_path = str(Path(tmpdir) / "test_checkpoint.pt")
    runner.save(checkpoint_path)

    wrapped_env.unwrapped.common_step_counter = 0
    runner.load(checkpoint_path)

    assert wrapped_env.unwrapped.common_step_counter == 12345


def test_runner_handles_old_checkpoints_without_env_state(env, device):
  """Old checkpoints without env_state should load without crashing."""

  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )

    checkpoint_path = str(Path(tmpdir) / "old_checkpoint.pt")
    old_checkpoint = {
      "actor_state_dict": runner.alg.actor.state_dict(),
      "critic_state_dict": runner.alg.critic.state_dict(),
      "optimizer_state_dict": runner.alg.optimizer.state_dict(),
      "iter": 100,
      "infos": None,
    }
    torch.save(old_checkpoint, checkpoint_path)

    wrapped_env.unwrapped.common_step_counter = 999
    runner.load(checkpoint_path)

    assert wrapped_env.unwrapped.common_step_counter == 999
