"""InvertedDoublePendulum-v5 composed from native mjlab articulation MDP terms."""

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
  normal_velocity_joint_reset,
)
from mjlab.tasks.gym_mujoco.gym_mujoco_env_cfg import (
  make_base_env_cfg,
)
from mjlab.viewer import ViewerConfig


def inverted_double_pendulum_env_cfg(
  play: bool = False, *, num_envs: int | None = None
) -> ManagerBasedRlEnvCfg:
  if num_envs is None:
    num_envs = 1 if play else 4096
  cfg = make_base_env_cfg(
    "inverted_double_pendulum.xml",
    num_envs,
    sim=MujocoCfg(
      gravity=(1e-5, 0.0, -9.81),
      timestep=0.01,
      integrator="rk4",
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
  slider = SceneEntityCfg("robot", joint_names=("slider",))
  hinges = SceneEntityCfg("robot", joint_names=("hinge", "hinge2"))
  tip = SceneEntityCfg("robot", site_ids=[0])
  terms = {
    "cart_position": ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": slider}),
    "pole_sin": ObsTerm(func=mdp.joint_pos_sin, params={"asset_cfg": hinges}),
    "pole_cos": ObsTerm(func=mdp.joint_pos_cos, params={"asset_cfg": hinges}),
    "joint_velocity": ObsTerm(func=mdp.joint_vel_rel, clip=(-10.0, 10.0)),
    "cart_constraint_force": ObsTerm(
      func=mdp.joint_constraint_force, params={"asset_cfg": slider}, clip=(-10.0, 10.0)
    ),
  }
  cfg.observations = actor_critic_observations(terms)
  cfg.rewards = {
    "survive": RewTerm(func=envs_mdp.is_alive, weight=10.0),
    "distance": RewTerm(func=mdp.tip_distance, weight=-1.0, params={"asset_cfg": tip}),
    "first_pole_velocity": RewTerm(
      func=envs_mdp.joint_vel_l2,
      weight=-1e-3,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("hinge",))},
    ),
    "second_pole_velocity": RewTerm(
      func=envs_mdp.joint_vel_l2,
      weight=-5e-3,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("hinge2",))},
    ),
  }
  cfg.terminations["tip_height"] = DoneTerm(
    func=mdp.tip_too_low, params={"minimum_height": 1.0, "asset_cfg": tip}
  )
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    distance=4.0,
    elevation=-10.0,
    azimuth=90.0,
    lookat=(0.0, 0.0, 1.0),
  )
  return cfg
