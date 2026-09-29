"""InvertedPendulum-v5 composed from native mjlab articulation MDP terms."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
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


def inverted_pendulum_env_cfg(
  play: bool = False, *, num_envs: int | None = None
) -> ManagerBasedRlEnvCfg:
  if num_envs is None:
    num_envs = 1 if play else 4096
  cfg = make_base_env_cfg(
    "inverted_pendulum.xml",
    num_envs,
    sim=MujocoCfg(
      timestep=0.02,
      integrator="rk4",
      ccd_iterations=35,
      jacobian="dense",
    ),
    decimation=2,
    max_episode_steps=1000,
    init_state=EntityCfg.InitialStateCfg(
      joint_pos={".*": 0.0},
      joint_vel={".*": 0.0},
    ),
  )
  cfg.events["reset_joints"] = uniform_joint_reset(0.01, 0.01)
  cfg.observations = actor_critic_observations(joint_observation_terms())
  cfg.rewards = {
    "survive": RewTerm(func=envs_mdp.is_alive, weight=1.0),
  }
  cfg.terminations["unhealthy"] = DoneTerm(func=mdp.pendulum_unhealthy)
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    distance=2.5,
    elevation=-10.0,
    azimuth=90.0,
    lookat=(0.0, 0.0, 0.5),
  )
  return cfg
