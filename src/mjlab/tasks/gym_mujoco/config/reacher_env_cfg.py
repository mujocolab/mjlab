"""Reacher-v5 composed from native mjlab articulation MDP terms."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim import MujocoCfg
from mjlab.tasks.gym_mujoco import mdp
from mjlab.tasks.gym_mujoco.config.common import (
  actor_critic_observations,
  uniform_joint_reset,
)
from mjlab.tasks.gym_mujoco.gym_mujoco_env_cfg import (
  make_base_env_cfg,
)
from mjlab.viewer import ViewerConfig


def reacher_env_cfg(
  play: bool = False, *, num_envs: int | None = None
) -> ManagerBasedRlEnvCfg:
  if num_envs is None:
    num_envs = 1 if play else 4096
  cfg = make_base_env_cfg(
    "reacher.xml",
    num_envs,
    sim=MujocoCfg(
      timestep=0.01,
      integrator="rk4",
      ccd_iterations=35,
      jacobian="dense",
    ),
    decimation=2,
    max_episode_steps=50,
    init_state=EntityCfg.InitialStateCfg(
      joint_pos={"joint0": 0.0, "joint1": 0.0, "target_x": 0.1, "target_y": -0.1},
      joint_vel={".*": 0.0},
    ),
  )
  cfg.events["reset_joints"] = uniform_joint_reset(0.1, 0.005)
  arm = SceneEntityCfg("robot", joint_names=("joint0", "joint1"))
  fingertip_target = SceneEntityCfg(
    "robot", body_names=("fingertip", "target"), preserve_order=True
  )
  terms = {
    "joint_cos": ObsTerm(func=mdp.joint_pos_cos, params={"asset_cfg": arm}),
    "joint_sin": ObsTerm(func=mdp.joint_pos_sin, params={"asset_cfg": arm}),
    "target_position": ObsTerm(
      func=mdp.generated_commands, params={"command_name": "target"}
    ),
    "joint_velocity": ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": arm}),
    "fingertip_to_target": ObsTerm(
      func=mdp.body_difference, params={"asset_cfg": fingertip_target, "dimensions": 2}
    ),
  }
  cfg.commands["target"] = mdp.ReacherTargetCommandCfg(
    # Gym holds a target for the whole episode; reset always resamples it.
    resampling_time_range=(1e9, 1e9),
  )
  cfg.observations = actor_critic_observations(terms)
  cfg.rewards = {
    "control": RewTerm(func=mdp.action_l2, weight=-1.0),
    "distance": RewTerm(
      func=mdp.body_distance, weight=-1.0, params={"asset_cfg": fingertip_target}
    ),
  }
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    distance=0.7,
    elevation=-90.0,
    azimuth=90.0,
    lookat=(0.0, 0.0, 0.0),
  )
  return cfg
