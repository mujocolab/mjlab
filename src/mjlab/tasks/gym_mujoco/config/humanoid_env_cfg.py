"""Humanoid-v5 composed from native mjlab articulation MDP terms."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm
from mjlab.sim import MujocoCfg
from mjlab.tasks.gym_mujoco import mdp
from mjlab.tasks.gym_mujoco.config.common import (
  actor_critic_observations,
  floating_observation_terms,
  floating_root_reset,
  uniform_joint_reset,
)
from mjlab.tasks.gym_mujoco.gym_mujoco_env_cfg import (
  make_base_env_cfg,
)
from mjlab.viewer import ViewerConfig


def humanoid_env_cfg(
  play: bool = False, *, num_envs: int | None = None
) -> ManagerBasedRlEnvCfg:
  if num_envs is None:
    num_envs = 1 if play else 2048
  cfg = make_base_env_cfg(
    "humanoid.xml",
    num_envs,
    sim=MujocoCfg(
      timestep=0.003,
      integrator="rk4",
      iterations=50,
      ccd_iterations=35,
      jacobian="dense",
    ),
    decimation=5,
    max_episode_steps=1000,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 1.4),
      joint_pos={".*": 0.0},
      joint_vel={".*": 0.0},
    ),
  )
  cfg.events["reset_joints"] = uniform_joint_reset(0.01, 0.01)
  cfg.events["reset_root"] = floating_root_reset(0.01, 0.01)
  bodies = SceneEntityCfg("robot", body_names=(".*",))
  terms = floating_observation_terms()
  terms.update(
    {
      "body_inertia": ObsTerm(func=mdp.body_inertia, params={"asset_cfg": bodies}),
      "body_velocity": ObsTerm(
        func=mdp.body_spatial_velocity, params={"asset_cfg": bodies}
      ),
      "joint_actuator_force": ObsTerm(func=mdp.joint_actuator_force),
      "contact_wrench": ObsTerm(func=mdp.contact_wrench, params={"asset_cfg": bodies}),
    }
  )
  cfg.observations = actor_critic_observations(terms)
  cfg.rewards = {
    "forward": RewTerm(
      func=mdp.com_forward_velocity, weight=1.25, params={"asset_cfg": bodies}
    ),
    "control": RewTerm(func=mdp.action_l2, weight=-0.1),
    "survive": RewTerm(func=envs_mdp.is_alive, weight=5.0),
    "contact": RewTerm(
      func=mdp.contact_cost,
      weight=-1.0,
      params={"asset_cfg": bodies, "cost_weight": 5e-7, "max_cost": 10.0},
    ),
  }
  cfg.terminations["unhealthy"] = DoneTerm(
    func=mdp.root_unhealthy, params={"z_range": (1.0, 2.0)}
  )
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="torso",
    distance=4.0,
    elevation=-20.0,
    azimuth=110.0,
  )
  return cfg
