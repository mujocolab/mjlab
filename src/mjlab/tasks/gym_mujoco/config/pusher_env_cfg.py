"""Pusher-v5 composed from native mjlab articulation MDP terms."""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
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


def pusher_env_cfg(
  play: bool = False, *, num_envs: int | None = None
) -> ManagerBasedRlEnvCfg:
  if num_envs is None:
    num_envs = 1 if play else 4096
  cfg = make_base_env_cfg(
    "pusher_v5.xml",
    num_envs,
    sim=MujocoCfg(
      gravity=(0.0, 0.0, 0.0),
      timestep=0.01,
      integrator="euler",
      iterations=20,
      ccd_iterations=35,
      jacobian="dense",
    ),
    decimation=5,
    max_episode_steps=100,
    init_state=EntityCfg.InitialStateCfg(
      joint_pos={".*": 0.0},
      joint_vel={".*": 0.0},
    ),
  )
  cfg.events["reset_joints"] = uniform_joint_reset(0.0, 0.005)
  arm = SceneEntityCfg(
    "robot",
    joint_names=(
      "r_shoulder_pan_joint",
      "r_shoulder_lift_joint",
      "r_upper_arm_roll_joint",
      "r_elbow_flex_joint",
      "r_forearm_roll_joint",
      "r_wrist_flex_joint",
      "r_wrist_roll_joint",
    ),
  )
  object_target = SceneEntityCfg(
    "robot", body_names=("object", "goal"), preserve_order=True
  )
  object_tip = SceneEntityCfg(
    "robot", body_names=("object", "tips_arm"), preserve_order=True
  )
  terms = {
    "joint_position": ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": arm}),
    "joint_velocity": ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": arm}),
    "tip_position": ObsTerm(
      func=mdp.body_position,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("tips_arm",))},
    ),
    "object_position": ObsTerm(
      func=mdp.body_position,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("object",))},
    ),
    "target_position": ObsTerm(
      func=mdp.body_position,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("goal",))},
    ),
  }
  cfg.events["reset_object_goal"] = EventTermCfg(
    func=mdp.reset_pusher_object,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=("obj_slidey", "obj_slidex", "goal_slidey", "goal_slidex")
      )
    },
  )
  cfg.observations = actor_critic_observations(terms)
  cfg.rewards = {
    "control": RewTerm(func=mdp.action_l2, weight=-0.1),
    "distance": RewTerm(
      func=mdp.body_distance, weight=-1.0, params={"asset_cfg": object_target}
    ),
    "near": RewTerm(
      func=mdp.body_distance, weight=-0.5, params={"asset_cfg": object_tip}
    ),
  }
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    distance=2.0,
    elevation=-65.0,
    azimuth=90.0,
    lookat=(0.35, -0.25, 0.0),
  )
  return cfg
