"""HalfCheetah-v5 composed from native mjlab articulation MDP terms."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim import MujocoCfg
from mjlab.tasks.gym_mujoco import mdp
from mjlab.tasks.gym_mujoco.config.common import (
  actor_critic_observations,
  joint_observation_terms,
  normal_velocity_joint_reset,
)
from mjlab.tasks.gym_mujoco.gym_mujoco_env_cfg import (
  make_base_env_cfg,
)
from mjlab.viewer import ViewerConfig


def half_cheetah_env_cfg(
  play: bool = False, *, num_envs: int | None = None
) -> ManagerBasedRlEnvCfg:
  if num_envs is None:
    num_envs = 1 if play else 4096
  cfg = make_base_env_cfg(
    "half_cheetah.xml",
    num_envs,
    sim=MujocoCfg(
      timestep=0.01,
      integrator="euler",
      ccd_iterations=35,
      jacobian="dense",
    ),
    decimation=5,
    max_episode_steps=1000,
    init_state=EntityCfg.InitialStateCfg(
      joint_pos={".*": 0.0},
      joint_vel={".*": 0.0},
    ),
  )
  cfg.events["reset_joints"] = normal_velocity_joint_reset(0.1, 0.1)
  # Gym omits horizontal translation from positions, but keeps all velocities.
  terms = joint_observation_terms()
  terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=("(?!rootx$).*",)
  )
  cfg.observations = actor_critic_observations(terms)
  cfg.rewards = {
    "forward": RewTerm(
      func=mdp.forward_velocity,
      weight=1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("rootx",))},
    ),
    "control": RewTerm(func=mdp.action_l2, weight=-0.1),
  }
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="torso",
    distance=4.0,
    elevation=-15.0,
    azimuth=90.0,
  )
  return cfg
