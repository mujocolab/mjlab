"""Unitree G1 velocity-football environment configurations."""

from copy import deepcopy
from typing import Literal

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity_football.mdp.observations import (
  ball_visible_mask,
  masked_ball_pos_b,
  masked_ball_to_feet_vectors_b,
  perceived_ball_pos_b,
  perceived_ball_to_feet_vectors_b,
)
from mjlab.tasks.velocity_football.mdp.rewards import (
  ball_front_control,
  stop_ball_lin_vel_xy_exp,
)
from mjlab.tasks.velocity_football.mdp.velocity_command import (
  StopSkillVelocityReferenceCfg,
  UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity_football.velocity_football_env_cfg import (
  make_velocity_env_cfg,
)

from .pose import get_isaaclab_default_keyframe


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  # Add the robot without replacing the football created by the base scene.
  cfg.scene.entities["robot"] = get_g1_robot_cfg()
  cfg.scene.entities["robot"].init_state = get_isaaclab_default_keyframe()

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  ball_forbidden_contact_cfg = ContactSensorCfg(
    name="ball_forbidden_contact",
    primary=ContactMatch(
      mode="body",
      # Only knee links are forbidden from contacting the football.
      pattern=r"^(left_knee_link|right_knee_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="^ball$", entity="ball"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    ball_forbidden_contact_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15
  twist_cmd.zero_command_ramp_time_range = None
  twist_cmd.ball_relative_velocity_reference = None
  twist_cmd.stop_skill_velocity_reference = StopSkillVelocityReferenceCfg(
    maximum_velocity=1.0,
    rise_amplitude=0.2,
    rise_duration=0.3,
    fall_duration=0.3,
    trigger_window=5,
    acceleration_threshold=1.0,
    minimum_command_drop=0.12,
    persistence_frames=2,
    rearm_acceleration_threshold=0.2,
  )
  ball_velocity_reward = cfg.rewards["track_ball_lin_vel_xy_exp"]
  ball_velocity_reward.params["use_user_command"] = False
  ball_velocity_reward.params["use_ball_command"] = True

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["ball_forbidden_contacts"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-2.0,
    params={
      "sensor_name": ball_forbidden_contact_cfg.name,
      "force_threshold": 5.0,
    },
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  return cfg


def unitree_g1_temporal_flat_env_cfg(
  play: bool = False,
  history_length: int = 10,
) -> ManagerBasedRlEnvCfg:
  """E3 football task with a 10-frame TemporalCNN and masked ball vision."""
  if history_length <= 0:
    raise ValueError("history_length must be positive")
  cfg = unitree_g1_flat_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.zero_command_ramp_time_range = None
  twist_cmd.ball_relative_velocity_reference = None
  twist_cmd.stop_skill_velocity_reference = None

  ball_velocity_reward = cfg.rewards["track_ball_lin_vel_xy_exp"]
  ball_velocity_reward.params["use_user_command"] = True
  ball_velocity_reward.params["use_ball_command"] = False

  actor = cfg.observations["actor"]
  visual_params = {
    "x_range": (0.05, 1.50),
    "y_range": (-0.70, 0.70),
    "dropout_probability": 0.0,
    "bias_range": 0.0 if play else 0.10,
    "frame_noise_range": 0.0 if play else 0.06,
    "ball_cfg": SceneEntityCfg("ball"),
    "asset_cfg": SceneEntityCfg(
      "robot",
      body_names=(r".*_ankle_roll_link",),
    ),
  }
  actor.terms["ball_pos_b"] = ObservationTermCfg(
    func=masked_ball_pos_b,
    params=deepcopy(visual_params),
  )
  actor.terms["ball_to_feet_vectors_b"] = ObservationTermCfg(
    func=masked_ball_to_feet_vectors_b,
    params=deepcopy(visual_params),
  )
  actor.terms["ball_visible_mask"] = ObservationTermCfg(
    func=ball_visible_mask,
    params=deepcopy(visual_params),
  )
  actor.history_length = None
  actor.flatten_history_dim = True
  actor_history = deepcopy(actor)
  actor_history.history_length = history_length
  actor_history.flatten_history_dim = False
  cfg.observations["actor_history"] = actor_history

  critic = cfg.observations["critic"]
  critic.history_length = None
  critic.flatten_history_dim = True
  critic_history = deepcopy(critic)
  critic_history.history_length = history_length
  critic_history.flatten_history_dim = False
  cfg.observations["critic_history"] = critic_history

  return cfg


def unitree_g1_temporal_stop_reward_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Temporal football task with the isolated low-speed ball reward."""
  cfg = unitree_g1_temporal_flat_env_cfg(play=play)
  cfg.rewards["stop_ball_lin_vel_xy_exp"] = RewardTermCfg(
    func=stop_ball_lin_vel_xy_exp,
    weight=0.5,
    params={
      "std": 0.30,
      "command_name": "twist",
      "command_threshold": 0.10,
      "ball_cfg": SceneEntityCfg("ball"),
    },
  )
  return cfg


def unitree_g1_temporal_history_flat_env_cfg(
  history_length: int,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a TemporalCNN football task for history-length ablation."""
  return unitree_g1_temporal_flat_env_cfg(
    play=play,
    history_length=history_length,
  )


def unitree_g1_visual_mask_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Current-frame control task for the TemporalCNN history ablation."""
  cfg = unitree_g1_temporal_flat_env_cfg(play=play)
  cfg.observations.pop("actor_history")
  return cfg


RewardAblationVariant = Literal[
  "r0_isaaclab",
  "r1_e1",
  "r2_no_relative_velocity",
  "r3_no_relative_position",
]

FactorialBallRewardVariant = Literal["r0_isaaclab_ball", "r1_ball_center"]
B1_HISTORY_TERMS = (
  "ball_pos_b",
  "ball_to_feet_vectors_b",
  "ball_visible_mask",
)


def _configure_masked_ball_actor(cfg: ManagerBasedRlEnvCfg) -> None:
  actor = cfg.observations["actor"]
  visual_params = {
    "x_range": (0.05, 1.50),
    "y_range": (-0.70, 0.70),
    "dropout_probability": 0.0,
    "bias_range": 0.10,
    "frame_noise_range": 0.06,
    "ball_cfg": SceneEntityCfg("ball"),
    "asset_cfg": SceneEntityCfg(
      "robot",
      body_names=(r".*_ankle_roll_link",),
    ),
  }
  actor.terms["ball_pos_b"] = ObservationTermCfg(
    func=masked_ball_pos_b,
    params=deepcopy(visual_params),
  )
  actor.terms["ball_to_feet_vectors_b"] = ObservationTermCfg(
    func=masked_ball_to_feet_vectors_b,
    params=deepcopy(visual_params),
  )
  actor.terms["ball_visible_mask"] = ObservationTermCfg(
    func=ball_visible_mask,
    params=deepcopy(visual_params),
  )
  actor.history_length = None
  actor.flatten_history_dim = True


def _configure_privileged_critic_history(
  cfg: ManagerBasedRlEnvCfg,
  history_length: int,
) -> None:
  critic = cfg.observations["critic"]
  critic.history_length = None
  critic.flatten_history_dim = True
  critic_history = deepcopy(critic)
  critic_history.history_length = history_length
  critic_history.flatten_history_dim = False
  cfg.observations["critic_history"] = critic_history


def _configure_factorial_ball_reward(
  cfg: ManagerBasedRlEnvCfg,
  variant: FactorialBallRewardVariant,
) -> None:
  if variant == "r1_ball_center":
    cfg.rewards.pop("ball_front_control", None)
    return
  if variant != "r0_isaaclab_ball":
    raise ValueError(f"Unknown factorial ball reward variant: {variant}")

  cfg.rewards["track_ball_lin_vel_xy_exp"].weight = 1.0
  cfg.rewards["track_ball_lin_vel_xy_exp"].params = {
    "command_name": "twist",
    "std": 0.5,
    "gate_by_position": False,
    "use_user_command": True,
  }
  cfg.rewards["track_ball_relative_vel_xy_exp"].weight = 0.0
  cfg.rewards["track_ball_relative_pos_xy_exp"].weight = 0.0
  cfg.rewards["ball_outside_control_zone"].weight = 0.0
  cfg.rewards["ball_front_control"] = RewardTermCfg(
    func=ball_front_control,
    weight=0.5,
    params={"x_range": (0.10, 0.40), "y_abs": 0.15},
  )


def unitree_g1_factorial_flat_env_cfg(
  *,
  use_b1_history: bool,
  reward_variant: FactorialBallRewardVariant,
  play: bool = False,
  history_length: int = 10,
) -> ManagerBasedRlEnvCfg:
  """Create one frozen cell of the A0/A1 x R0/R1 factorial experiment."""
  if history_length <= 0:
    raise ValueError("history_length must be positive")
  cfg = unitree_g1_flat_env_cfg(play=play)
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.zero_command_ramp_time_range = None
  command.ball_relative_velocity_reference = None
  command.stop_skill_velocity_reference = None
  ball_reward = cfg.rewards["track_ball_lin_vel_xy_exp"]
  ball_reward.params["use_user_command"] = True
  ball_reward.params["use_ball_command"] = False

  _configure_masked_ball_actor(cfg)
  _configure_privileged_critic_history(cfg, history_length)
  if use_b1_history:
    actor = cfg.observations["actor"]
    actor_history = deepcopy(actor)
    actor_history.terms = {
      name: deepcopy(actor.terms[name]) for name in B1_HISTORY_TERMS
    }
    actor_history.history_length = history_length
    actor_history.flatten_history_dim = False
    cfg.observations["actor_history"] = actor_history
  _configure_factorial_ball_reward(cfg, reward_variant)
  return cfg


def unitree_g1_reward_ablation_flat_env_cfg(
  variant: RewardAblationVariant,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a controlled reward-ablation task with shared ball perception error."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.zero_command_ramp_time_range = None
  command.ball_relative_velocity_reference = None
  command.stop_skill_velocity_reference = None

  actor = cfg.observations["actor"]
  perception_range = 0.0 if play else 0.10
  frame_noise_range = 0.0 if play else 0.06
  ball_position = actor.terms["ball_pos_b"]
  ball_position.func = perceived_ball_pos_b
  ball_position.params = {
    "bias_range": perception_range,
    "frame_noise_range": frame_noise_range,
  }
  ball_position.noise = None

  ball_to_feet = actor.terms["ball_to_feet_vectors_b"]
  ball_to_feet.func = perceived_ball_to_feet_vectors_b
  ball_to_feet.params = {
    **ball_to_feet.params,
    "bias_range": perception_range,
    "frame_noise_range": frame_noise_range,
  }
  ball_to_feet.noise = None

  if variant == "r0_isaaclab":
    cfg.rewards["track_ball_lin_vel_xy_exp"].weight = 1.0
    cfg.rewards["track_ball_lin_vel_xy_exp"].params = {
      "command_name": "twist",
      "std": 0.5,
      "gate_by_position": False,
      "use_user_command": True,
    }
    cfg.rewards["track_linear_velocity"].weight = 1.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.5
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = 0.5
    cfg.rewards["track_ball_relative_vel_xy_exp"].weight = 0.0
    cfg.rewards["track_ball_relative_pos_xy_exp"].weight = 0.0
    cfg.rewards["ball_outside_control_zone"].weight = 0.0
    cfg.rewards["ball_front_control"] = RewardTermCfg(
      func=ball_front_control,
      weight=0.5,
      params={"x_range": (0.10, 0.40), "y_abs": 0.15},
    )
  elif variant == "r1_e1":
    pass
  elif variant == "r2_no_relative_velocity":
    cfg.rewards["track_ball_relative_vel_xy_exp"].weight = 0.0
  elif variant == "r3_no_relative_position":
    cfg.rewards["track_ball_relative_pos_xy_exp"].weight = 0.0
    cfg.rewards["ball_outside_control_zone"].weight = 0.0
  else:
    raise ValueError(f"Unknown reward-ablation variant: {variant}")

  return cfg
