"""Hopper-v5 composed from native mjlab articulation MDP terms."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm
from mjlab.sim import MujocoCfg
from mjlab.tasks.gym_mujoco import mdp
from mjlab.tasks.gym_mujoco.config.common import (
  actor_critic_observations,
  joint_observation_terms,
  uniform_joint_reset,
)
from mjlab.tasks.gym_mujoco.gym_mujoco_env_cfg import (
  make_base_env_cfg,
)
from mjlab.viewer import ViewerConfig


def hopper_env_cfg(
  play: bool = False, *, num_envs: int | None = None
) -> ManagerBasedRlEnvCfg:
  if num_envs is None:
    num_envs = 1 if play else 2048
  cfg = make_base_env_cfg(
    "hopper.xml",
    num_envs,
    sim=MujocoCfg(
      timestep=0.002,
      integrator="rk4",
      ccd_iterations=35,
      jacobian="dense",
    ),
    decimation=4,
    max_episode_steps=1000,
    init_state=EntityCfg.InitialStateCfg(
      joint_pos={
        "rootx": 0.0,
        "rootz": 1.25,
        "rooty": 0.0,
        "thigh_joint": 0.0,
        "leg_joint": 0.0,
        "foot_joint": 0.0,
      },
      joint_vel={".*": 0.0},
    ),
  )
  cfg.events["reset_joints"] = uniform_joint_reset(0.005, 0.005)
  # Gym omits horizontal translation from positions, but keeps all velocities.
  terms = joint_observation_terms()
  terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=("(?!rootx$).*",)
  )
  terms["joint_vel"].clip = (-10.0, 10.0)
  cfg.observations = actor_critic_observations(terms)
  cfg.rewards = {
    "forward": RewTerm(
      func=mdp.forward_velocity,
      weight=1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("rootx",))},
    ),
    "control": RewTerm(func=mdp.action_l2, weight=-0.001),
    "survive": RewTerm(func=envs_mdp.is_alive, weight=1.0),
  }
  cfg.terminations["unhealthy"] = DoneTerm(
    func=mdp.planar_unhealthy,
    params={
      "z_range": (0.7, float("inf")),
      "angle_range": (-0.2, 0.2),
      "state_range": (-100.0, 100.0),
    },
  )
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="torso",
    distance=3.0,
    elevation=-20.0,
    azimuth=90.0,
  )
  return cfg
