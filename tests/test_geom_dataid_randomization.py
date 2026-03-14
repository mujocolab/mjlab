"""Tests for per-world mesh randomization via ``dr.geom_dataid``."""

import tempfile
from dataclasses import asdict

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

MESH_VARIANT_XML = """
<mujoco>
  <asset>
    <mesh name="m_small" vertex="0 0 0  0.04 0 0  0 0.04 0  0 0 0.04"
      face="0 1 2  0 1 3  0 2 3  1 2 3"/>
    <mesh name="m_medium" vertex="0 0 0  0.06 0 0  0 0.06 0  0 0 0.06"
      face="0 1 2  0 1 3  0 2 3  1 2 3"/>
    <mesh name="m_large" vertex="0 0 0  0.09 0 0  0 0.09 0  0 0 0.09"
      face="0 1 2  0 1 3  0 2 3  1 2 3"/>
  </asset>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="obj_geom" type="mesh" mesh="m_small" mass="1.0"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
        <geom name="link1_geom" type="box" size="0.08 0.08 0.08" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="actuator1" joint="joint1" gear="1.0"/>
  </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _make_env(device: str, *, num_envs: int, with_reset_event: bool) -> ManagerBasedRlEnv:
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(MESH_VARIANT_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlMotorActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

  events: dict[str, EventTermCfg] = {}
  if with_reset_event:
    events["randomize_object_mesh"] = EventTermCfg(
      mode="reset",
      func=dr.geom_dataid,
      params={
        "mesh_ids": ("m_small", "m_medium", "m_large"),
        "assignment_mode": "cycle",
        "shared_random": True,
        "asset_cfg": SceneEntityCfg("robot", geom_names=("obj_geom",)),
      },
    )

  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=num_envs,
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
    events=events,
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=1,
    episode_length_s=0.1,
  )
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def _obj_geom_global_id(env: ManagerBasedRlEnv) -> int:
  geom_cfg = SceneEntityCfg("robot", geom_names=("obj_geom",))
  geom_cfg.resolve(env.scene)
  geom_ids = env.scene["robot"].indexing.geom_ids[geom_cfg.geom_ids]
  return int(geom_ids[0].item())


def _mesh_support_or_skip(env: ManagerBasedRlEnv) -> None:
  if env.sim.model.geom_dataid.ndim != 2:
    pytest.skip(
      "This test requires per-world geom_dataid support from newer MuJoCo-Warp."
    )


def test_geom_dataid_randomizes_per_world(device):
  torch.manual_seed(0)
  env = _make_env(device, num_envs=12, with_reset_event=False)
  try:
    _mesh_support_or_skip(env)
    geom_id = _obj_geom_global_id(env)

    dr.geom_dataid(
      env,
      env_ids=None,
      mesh_ids=("m_small", "m_medium", "m_large"),
      asset_cfg=SceneEntityCfg("robot", geom_names=("obj_geom",)),
      assignment_mode="sample",
      shared_random=True,
    )

    values = env.sim.model.geom_dataid[:, geom_id]
    values_cpu = values.cpu()
    assert values_cpu.shape == (env.num_envs,)
    assert values_cpu.min().item() >= -1
    assert set(values_cpu.tolist()).issubset({0, 1, 2})
    assert torch.unique(values_cpu).numel() >= 2
  finally:
    env.close()


def test_geom_dataid_cycle_assignment(device):
  env = _make_env(device, num_envs=6, with_reset_event=False)
  try:
    _mesh_support_or_skip(env)
    geom_id = _obj_geom_global_id(env)

    dr.geom_dataid(
      env,
      env_ids=None,
      mesh_ids=("m_small", "m_medium", "m_large"),
      asset_cfg=SceneEntityCfg("robot", geom_names=("obj_geom",)),
      assignment_mode="cycle",
      shared_random=True,
    )
    values = env.sim.model.geom_dataid[:, geom_id].cpu()
    assert values.tolist() == [0, 1, 2, 0, 1, 2]
  finally:
    env.close()


@pytest.mark.slow
def test_multi_mesh_rl_training_smoke(device):
  env = _make_env(device, num_envs=6, with_reset_event=True)
  try:
    _mesh_support_or_skip(env)
    env.reset()
    geom_id = _obj_geom_global_id(env)
    initial = env.sim.model.geom_dataid[:, geom_id].clone()
    assert torch.unique(initial).numel() >= 2

    wrapped_env = RslRlVecEnvWrapper(env)
    agent_cfg = asdict(
      RslRlOnPolicyRunnerCfg(
        num_steps_per_env=4,
        max_iterations=2,
        save_interval=50,
      )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      runner = MjlabOnPolicyRunner(
        wrapped_env, agent_cfg, log_dir=tmpdir, device=device
      )
      runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)

    after = env.sim.model.geom_dataid[:, geom_id]
    assert after.shape[0] == env.num_envs
    assert torch.unique(after).numel() >= 2
  finally:
    env.close()
