"""CartPole task environment configuration for testing."""

import math

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from .cartpole_constants import get_cartpole_robot_cfg

SCENE_CFG = SceneCfg(
  num_envs=64,
  extent=1.0,
  entities={"robot": get_cartpole_robot_cfg()},
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="pole",
  distance=3.0,
  elevation=10.0,
  azimuth=90.0,
)

SIM_CFG = SimulationCfg(
  mujoco=MujocoCfg(
    timestep=0.02,
    iterations=1,
  ),
)


def create_cartpole_actions() -> dict[str, JointPositionActionCfg]:
  """Create CartPole actions."""
  return {
    "joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      scale=20.0,
      use_default_offset=False,
    ),
  }


def create_cartpole_observations() -> dict[str, ObservationGroupCfg]:
  """Create CartPole observations."""
  policy_terms = {
    "angle": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, 1:2] / math.pi),
    "ang_vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, 1:2] / 5.0),
    "cart_pos": ObservationTermCfg(func=lambda env: env.sim.data.qpos[:, 0:1] / 2.0),
    "cart_vel": ObservationTermCfg(func=lambda env: env.sim.data.qvel[:, 0:1] / 20.0),
  }

  return {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
    ),
    "critic": ObservationGroupCfg(
      terms=policy_terms,  # Same observations for critic
      concatenate_terms=True,
    ),
  }


def compute_upright_reward(env):
  """Compute reward for keeping pole upright."""
  return env.sim.data.qpos[:, 1].cos()


def compute_effort_penalty(env):
  """Compute penalty for control effort."""
  return -0.01 * (env.sim.data.ctrl[:, 0] ** 2)


def create_cartpole_rewards() -> dict[str, RewardTermCfg]:
  """Create CartPole rewards."""
  return {
    "upright": RewardTermCfg(func=compute_upright_reward, weight=5.0),
    "effort": RewardTermCfg(func=compute_effort_penalty, weight=1.0),
  }


def random_push_cart(env, env_ids, force_range=(-5, 5)):
  """Apply random external forces to the cart."""
  n = len(env_ids)
  random_forces = (
    torch.rand(n, device=env.device) * (force_range[1] - force_range[0]) + force_range[0]
  )
  env.sim.data.qfrc_applied[env_ids, 0] = random_forces


def create_cartpole_events() -> dict[str, EventTermCfg]:
  """Create CartPole events."""
  return {
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "position_range": (-0.1, 0.1),
        "velocity_range": (-0.1, 0.1),
      },
    ),
    "random_push": EventTermCfg(
      func=random_push_cart,
      mode="interval",
      interval_range_s=(1.0, 2.0),
      params={"force_range": (-20.0, 20.0)},
    ),
  }


def check_pole_tipped(env):
  """Check if pole has tipped beyond threshold."""
  return env.sim.data.qpos[:, 1].abs() > math.radians(30)


def create_cartpole_terminations() -> dict[str, TerminationTermCfg]:
  """Create CartPole terminations."""
  return {
    "timeout": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "tipped": TerminationTermCfg(func=check_pole_tipped, time_out=False),
  }


def create_cartpole_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create CartPole environment configuration."""
  return ManagerBasedRlEnvCfg(
    scene=SCENE_CFG,
    observations=create_cartpole_observations(),
    actions=create_cartpole_actions(),
    rewards=create_cartpole_rewards(),
    events=create_cartpole_events(),
    terminations=create_cartpole_terminations(),
    sim=SIM_CFG,
    viewer=VIEWER_CONFIG,
    decimation=1,
    episode_length_s=10.0,
  )


# Module-level constant for gymnasium registration
CARTPOLE_ENV_CFG = create_cartpole_env_cfg()

