"""Tests for entities with non-actuated ball joints (#475)."""

from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.events import reset_joints_by_offset
from mjlab.envs.mdp.observations import joint_pos_rel, joint_vel_rel
from mjlab.envs.mdp.rewards import joint_pos_limits, posture
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg

# hinge, ball, hinge: the second hinge is joint 2 but qpos column 5, qvel column 4.
BALL_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="a">
      <joint name="h1" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
      <body name="b" pos="0 0 -0.2">
        <joint name="ball" type="ball" range="0 1.2"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
        <body name="c" pos="0 0 -0.2">
          <joint name="h2" type="hinge" axis="1 0 0" range="-2 2"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _entity(device, actuated=False, num_envs=2):
  articulation = None
  if actuated:
    articulation = EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("h.*",), stiffness=10.0, damping=1.0
        ),
      )
    )
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(BALL_XML),
    init_state=EntityCfg.InitialStateCfg(joint_pos={"h1": 0.3, "h2": -0.4}),
    articulation=articulation,
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


def _env(entity):
  env = Mock()
  env.scene = {"robot": entity}
  env.device = entity.data.device
  env.num_envs = entity.data.joint_pos.shape[0]
  return env


def _asset_cfg(entity, names=(".*",)):
  cfg = SceneEntityCfg("robot", joint_names=names)
  cfg.resolve(_env(entity).scene)
  return cfg


def test_defaults_follow_qpos_and_qvel_layout(device):
  entity, sim = _entity(device)
  data = entity.data
  torch.testing.assert_close(
    data.default_joint_pos[0],
    torch.tensor([0.3, 1.0, 0.0, 0.0, 0.0, -0.4], device=device),
  )
  assert data.default_joint_vel.shape == (2, sim.mj_model.nv)
  # Limits stay one row per joint.
  assert data.soft_joint_pos_limits.shape == (2, 3, 2)
  assert entity.indexing.q_cols([2]) == [5]
  assert entity.indexing.v_cols([1, 2]) == [1, 2, 3, 4]


def test_write_by_joint_id_hits_the_joint_columns(device):
  entity, sim = _entity(device)
  entity.write_joint_state_to_sim(
    entity.data.default_joint_pos, entity.data.default_joint_vel
  )
  env_ids = torch.tensor([1], device=device)
  # Full (not unit) quaternion for the ball, scalar for h2.
  pos = torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.7]], device=device)
  vel = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device)
  entity.write_joint_position_to_sim(pos, joint_ids=[1, 2], env_ids=env_ids)
  entity.write_joint_velocity_to_sim(vel, joint_ids=[1, 2], env_ids=env_ids)
  torch.testing.assert_close(
    sim.data.qpos[1], torch.tensor([0.3, 0.0, 0.0, 1.0, 0.0, 0.7], device=device)
  )
  torch.testing.assert_close(
    sim.data.qvel[1], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], device=device)
  )
  torch.testing.assert_close(sim.data.qpos[0], entity.data.default_joint_pos[0])


def test_hinge_actuator_after_ball_reads_its_own_joint(device):
  entity, sim = _entity(device, actuated=True)
  sim.data.qpos[:] = torch.tensor([0.1, 1.0, 0.0, 0.0, 0.0, -0.5], device=device)
  torch.testing.assert_close(
    entity.data.joint_pos_by_joint[:, [0, 2]],
    torch.tensor([[0.1, -0.5]] * 2, device=device),
  )


def test_ball_joint_actuator_is_rejected(device):
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(BALL_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("ball",), stiffness=1.0, damping=1.0
        ),
      )
    ),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
  with pytest.raises(ValueError, match="Ball joints"):
    entity.initialize(model, sim.model, sim.data, device)


def test_joint_pos_rel_is_a_rotation_vector(device):
  entity, sim = _entity(device)
  angle = 0.3
  q = [0.3, np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0, -0.4]
  sim.data.qpos[:] = torch.tensor([q] * 2, device=device, dtype=torch.float)
  env = _env(entity)
  obs = joint_pos_rel(env, asset_cfg=_asset_cfg(entity))
  expected = torch.tensor([[0.0, 0.0, angle, 0.0, 0.0]] * 2, device=device)
  torch.testing.assert_close(obs, expected, atol=1e-6, rtol=0)
  # Matches mj_differentiatePos, the frame of qvel.
  mjd = np.zeros(sim.mj_model.nv)
  default = entity.data.default_joint_pos[0].cpu().double().numpy()
  mujoco.mj_differentiatePos(sim.mj_model, mjd, 1.0, default, np.array(q))
  np.testing.assert_allclose(obs[0].cpu().numpy(), mjd, atol=1e-6)
  assert joint_vel_rel(env, asset_cfg=_asset_cfg(entity)).shape == (2, 5)
  only_h2 = joint_pos_rel(env, asset_cfg=_asset_cfg(entity, ("h2",)))
  torch.testing.assert_close(only_h2, torch.zeros(2, 1, device=device))


def test_posture_and_limits_use_the_ball_angle(device):
  entity, sim = _entity(device)
  env = _env(entity)
  asset_cfg = _asset_cfg(entity)
  term = posture(
    RewardTermCfg(
      func=posture, weight=1.0, params={"asset_cfg": asset_cfg, "std": {".*": 1.0}}
    ),
    env,
  )
  angle = 1.5  # beyond the ball's 1.2 rad range
  q = [0.3, np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0, -0.4]
  sim.data.qpos[:] = torch.tensor([q] * 2, device=device, dtype=torch.float)
  reward = term(env, std=None, asset_cfg=asset_cfg)
  torch.testing.assert_close(
    reward, torch.full((2,), float(np.exp(-(angle**2) / 3)), device=device)
  )
  penalty = joint_pos_limits(env, asset_cfg=asset_cfg)
  torch.testing.assert_close(penalty, torch.full((2,), angle - 1.2, device=device))


def test_reset_by_offset_keeps_the_ball_orientation(device):
  entity, sim = _entity(device)
  env = _env(entity)
  reset_joints_by_offset(
    env,
    None,
    position_range=(0.1, 0.1),
    velocity_range=(0.0, 0.0),
    asset_cfg=_asset_cfg(entity),
  )
  torch.testing.assert_close(
    sim.data.qpos[0], torch.tensor([0.4, 1.0, 0.0, 0.0, 0.0, -0.3], device=device)
  )
