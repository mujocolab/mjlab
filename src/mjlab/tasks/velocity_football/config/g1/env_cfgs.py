"""Unitree G1 velocity-football environment configurations."""

from copy import deepcopy
from typing import Literal

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_klavier_robot_cfg,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
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
from mjlab.tasks.velocity_football.mdp.curriculums import (
  lin_vel_cmd_levels,
  normal_control_lin_vel_cmd_levels,
  push_velocity_levels,
  scheduled_rough_terrain_levels,
  visibility_blend_task_levels,
)
from mjlab.tasks.velocity_football.mdp.events import kick_football_velocity
from mjlab.tasks.velocity_football.mdp.metrics import (
  ball_control_zone_success,
  command_velocity_envelope_violation,
  user_command_linear_velocity_error,
  user_command_yaw_velocity_error,
)
from mjlab.tasks.velocity_football.mdp.observations import (
  ball_visible_mask,
  episode_ball_observation_hidden,
  masked_ball_features_b,
  masked_ball_pos_b,
  masked_ball_to_feet_vectors_b,
  perceived_ball_pos_b,
  perceived_ball_to_feet_vectors_b,
)
from mjlab.tasks.velocity_football.mdp.rewards import (
  ball_front_control,
  command_velocity_envelope_l2,
  stop_ball_lin_vel_xy_exp,
  track_angular_velocity,
  track_hidden_angular_velocity,
  track_hidden_linear_velocity,
  track_linear_velocity,
  track_visibility_blended_angular_velocity,
  track_visibility_blended_linear_velocity,
  track_visible_recovery_angular_velocity,
  track_visible_recovery_linear_velocity,
)
from mjlab.tasks.velocity_football.mdp.velocity_command import (
  StopSkillVelocityReferenceCfg,
  UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity_football.velocity_football_env_cfg import (
  make_velocity_env_cfg,
)
from mjlab.terrains import HfRandomUniformTerrainCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg

from .pose import get_isaaclab_default_keyframe
from .velocity_env_cfgs import KLAVIER_JOINT_ORDER


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
    "x_range": (0.05, 1.00),
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


def _configure_masked_ball_actor(cfg: ManagerBasedRlEnvCfg) -> None:
  actor = cfg.observations["actor"]
  visual_params = {
    "x_range": (0.05, 1.00),
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
  # A0 keeps the original five-frame MLP contract (104 * 5 = 520).  The
  # visibility bit belongs exclusively to the B1 temporal branch.
  actor.history_length = 5
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
    ball_pos = deepcopy(actor.terms["ball_pos_b"])
    ball_to_feet = deepcopy(actor.terms["ball_to_feet_vectors_b"])
    actor_history.terms = {
      "ball_pos_b": ball_pos,
      "ball_to_feet_vectors_b": ball_to_feet,
      "ball_visible_mask": ObservationTermCfg(
        func=ball_visible_mask,
        params=deepcopy(ball_pos.params),
      ),
    }
    actor_history.history_length = history_length
    actor_history.flatten_history_dim = False
    cfg.observations["actor_history"] = actor_history
    # B1 assigns all football temporal processing to the CNN.  The main MLP
    # retains only the 98-dimensional proprioceptive/control prefix for five
    # frames (490 dimensions), avoiding duplicate ball-history pathways.
    actor.terms.pop("ball_pos_b")
    actor.terms.pop("ball_to_feet_vectors_b")
  _configure_factorial_ball_reward(cfg, reward_variant)
  return cfg


def unitree_g1_klavier_ball_temporal_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Scheme A: Klavier walk body plus an independent 10x7 BallCNN stream."""
  cfg = unitree_g1_factorial_flat_env_cfg(
    use_b1_history=True,
    reward_variant="r0_isaaclab_ball",
    play=play,
    history_length=10,
  )
  cfg.scene.num_envs = 1 if play else 4096
  cfg.scene.extent = 2.5
  cfg.scene.entities["robot"] = get_g1_klavier_robot_cfg()
  cfg.scene.entities["robot"].init_state = get_isaaclab_default_keyframe()

  ordered_joints = SceneEntityCfg(
    "robot",
    joint_names=KLAVIER_JOINT_ORDER,
    preserve_order=True,
  )
  # Preserve the Walk checkpoint's interleaved 29-joint observation contract.
  # actor_history deliberately contains only the seven football features.
  for group_name in ("actor", "critic", "critic_history"):
    group = cfg.observations[group_name]
    for term_name in ("joint_pos", "joint_vel", "joint_torques"):
      if term_name in group.terms:
        group.terms[term_name].params["asset_cfg"] = deepcopy(ordered_joints)

  action_obs = cfg.observations["actor"].terms["actions"]
  action_obs.delay_min_lag = 0
  action_obs.delay_max_lag = 2
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = KLAVIER_JOINT_ORDER
  action.preserve_order = True
  action.scale = G1_ACTION_SCALE

  # Reuse the exact startup dynamics randomization used by the copied Walk task.
  cfg.events["foot_friction"].params.update(
    {
      "asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_collision",)),
      "ranges": (0.3, 1.6),
      "shared_random": False,
    }
  )
  cfg.events["base_mass"] = EventTermCfg(
    func=envs_mdp.dr.body_mass,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "ranges": (-1.0, 5.0),
      "operation": "add",
    },
  )
  cfg.events["base_com"].params.update(
    {
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
    }
  )
  cfg.events["joint_default_pos"] = EventTermCfg(
    func=envs_mdp.dr.joint_default_pos,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
      "ranges": (-0.015, 0.015),
      "operation": "add",
    },
  )
  cfg.events["joint_friction"] = EventTermCfg(
    func=envs_mdp.dr.joint_friction,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
      "ranges": (0.5, 1.5),
      "operation": "scale",
    },
  )
  cfg.events["joint_armature"] = EventTermCfg(
    func=envs_mdp.dr.joint_armature,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
      "ranges": (0.5, 1.5),
      "operation": "scale",
    },
  )
  cfg.events["actuator_gains"] = EventTermCfg(
    func=envs_mdp.dr.pd_gains,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=r".*"),
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
      "operation": "scale",
    },
  )

  # Keep the coordinated football placement while matching Walk's reset/push
  # velocity perturbations. Ball placement and football rewards stay unchanged.
  cfg.events["reset_football"].params["robot_velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.5, 0.5),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
  }

  # Ten percent of all episodes lose the complete synchronized ball stream
  # after 2--6 s and remain blind until reset. Sample only from the 95%
  # non-standing episodes so long-dropout and standing modes remain disjoint.
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.standing_mode_per_episode = True
  transition_probability = 0.0 if play else 0.10 / 0.95
  actor_history = cfg.observations["actor_history"]
  for term in actor_history.terms.values():
    # Train from exact ball geometry while retaining the legacy 0--2 control
    # step observation latency.  Long-term visibility loss remains enabled.
    term.params["bias_range"] = 0.0
    term.params["frame_noise_range"] = 0.0
    term.noise = None
    term.delay_min_lag = 0
    term.delay_max_lag = 0
    term.params["dropout_probability"] = 0.0
    term.params["episode_dropout_probability"] = 0.0
    term.params["transition_dropout_probability"] = transition_probability
    term.params["transition_dropout_start_range_s"] = (2.0, 6.0)
    term.params["transition_dropout_duration_range_s"] = (0.2, 0.8)
    term.params["transition_dropout_until_end_probability"] = 0.0 if play else 1.0
    term.params["transition_excluded_standing_command_name"] = "twist"
    term.params["sensor_reward_fade_out_s"] = 0.5
    term.params["sensor_reward_fade_in_s"] = 0.5

  # One 7-D term means ball XY, both foot vectors, and visibility pass through
  # exactly one delay buffer and therefore always use the same 0--2-step frame.
  ball_features = deepcopy(actor_history.terms["ball_pos_b"])
  ball_features.func = masked_ball_features_b
  ball_features.delay_min_lag = 0
  ball_features.delay_max_lag = 2
  actor_history.terms = {"ball_features_b": ball_features}
  ball_termination = cfg.terminations["ball_out_of_control"].params
  ball_termination["ignore_episode_hidden"] = False
  ball_termination["ignore_when_sensor_hidden"] = not play
  cfg.rewards["track_ball_lin_vel_xy_exp"].params["gate_by_sensor_health"] = True
  cfg.rewards["ball_front_control"].params["gate_by_sensor_health"] = True

  if not play:
    cfg.curriculum["push_velocity_levels"] = CurriculumTermCfg(
      func=push_velocity_levels,
      params={
        "event_term_name": "push_robot",
        "unlock_command_name": "twist",
        "unlock_lin_vel_x": (-0.5, 2.0),
        "unlock_lin_vel_y": (-0.5, 0.5),
        "survival_threshold": 0.95,
        "max_velocity_range": {
          "x": (-1.5, 1.5),
          "y": (-1.0, 1.0),
          "z": (-0.5, 0.5),
          "roll": (-0.8, 0.8),
          "pitch": (-0.8, 0.8),
          "yaw": (-1.57, 1.57),
        },
      },
    )
  return cfg


def _align_isaaclab_actor_observation_randomization(
  cfg: ManagerBasedRlEnvCfg,
) -> None:
  """Match the original Isaac Lab 520-input Actor observation distribution."""
  cfg.events.pop("encoder_bias", None)
  cfg.observations["actor"].terms["joint_pos"].params["biased"] = False
  visual_group = cfg.observations.get("actor_history", cfg.observations["actor"])
  for term_name in ("ball_pos_b", "ball_to_feet_vectors_b", "ball_visible_mask"):
    if term_name not in visual_group.terms:
      continue
    visual_group.terms[term_name].params["bias_range"] = 0.0
    visual_group.terms[term_name].params["frame_noise_range"] = 0.0

  ball_pos = visual_group.terms["ball_pos_b"]
  ball_pos.noise = UniformNoiseCfg(n_min=-0.05, n_max=0.05)
  ball_pos.delay_min_lag = 0
  ball_pos.delay_max_lag = 2
  ball_to_feet = visual_group.terms["ball_to_feet_vectors_b"]
  ball_to_feet.noise = UniformNoiseCfg(n_min=-0.10, n_max=0.10)
  ball_to_feet.delay_min_lag = 0
  ball_to_feet.delay_max_lag = 2


def unitree_g1_isaaclab_aligned_flat_env_cfg(
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Original Isaac Lab-style 104-D observations stacked over five frames."""
  cfg = unitree_g1_factorial_flat_env_cfg(
    use_b1_history=False,
    reward_variant="r0_isaaclab_ball",
    play=play,
  )
  actor = cfg.observations["actor"]
  ball_pos = actor.terms["ball_pos_b"]
  actor.terms["ball_visible_mask"] = ObservationTermCfg(
    func=ball_visible_mask,
    params=deepcopy(ball_pos.params),
  )
  _align_isaaclab_actor_observation_randomization(cfg)
  return cfg


def unitree_g1_isaaclab_history5_long_dropout10_flat_env_cfg(
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """IsaacLab-aligned five-frame MLP with long sensor-loss episodes."""
  cfg = unitree_g1_isaaclab_aligned_flat_env_cfg(play=play)
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.standing_mode_per_episode = True

  # Long dropout is sampled only from the 95% non-standing episodes. This
  # produces mutually exclusive 85% moving, 5% visible-standing, and 10%
  # long-dropout episode shares without changing the History5 Actor contract.
  transition_probability = 0.0 if play else 0.10 / 0.95
  until_end_probability = 0.0 if play else 1.0
  actor = cfg.observations["actor"]
  for term_name in ("ball_pos_b", "ball_to_feet_vectors_b", "ball_visible_mask"):
    term = actor.terms[term_name]
    term.params["episode_dropout_probability"] = 0.0
    term.params["transition_dropout_probability"] = transition_probability
    term.params["transition_dropout_start_range_s"] = (2.0, 6.0)
    term.params["transition_dropout_duration_range_s"] = (0.2, 0.8)
    term.params["transition_dropout_until_end_probability"] = until_end_probability
    term.params["transition_excluded_standing_command_name"] = "twist"
    term.params["sensor_reward_fade_out_s"] = 0.5
    term.params["sensor_reward_fade_in_s"] = 0.5

  ball_termination = cfg.terminations["ball_out_of_control"].params
  ball_termination["ignore_episode_hidden"] = False
  ball_termination["ignore_when_sensor_hidden"] = not play
  cfg.rewards["track_ball_lin_vel_xy_exp"].params["gate_by_sensor_health"] = True
  cfg.rewards["ball_front_control"].params["gate_by_sensor_health"] = True
  return cfg


def unitree_g1_factorial_history30_flat_env_cfg(
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """A1R0 with one shared 30-frame, seven-dimensional football history."""
  cfg = unitree_g1_factorial_flat_env_cfg(
    use_b1_history=True,
    reward_variant="r0_isaaclab_ball",
    play=play,
    history_length=30,
  )
  # History30 noise ablation: use independent per-control-frame position
  # noise of +/-0.20 m during training.  Play/evaluation remains noise-free.
  frame_noise_range = 0.0 if play else 0.20
  for term in cfg.observations["actor_history"].terms.values():
    term.params["frame_noise_range"] = frame_noise_range
  # Keep the Critic's current 110-D privileged observation, but restrict its
  # temporal CNN input to the same masked 7-D football stream used by the
  # Actor.  This avoids convolving 30 frames of the entire privileged state.
  cfg.observations["critic_history"] = deepcopy(cfg.observations["actor_history"])
  return cfg


def unitree_g1_factorial_dropout_flat_env_cfg(
  *,
  play: bool = False,
  dropout_probability: float = 0.10,
  episode_dropout_probability: float = 0.0,
) -> ManagerBasedRlEnvCfg:
  """A1R0 with synchronized frame- and episode-level observation dropout."""
  if not 0.0 <= dropout_probability <= 1.0:
    raise ValueError("dropout_probability must be in [0, 1]")
  if not 0.0 <= episode_dropout_probability <= 1.0:
    raise ValueError("episode_dropout_probability must be in [0, 1]")
  cfg = unitree_g1_factorial_flat_env_cfg(
    use_b1_history=True,
    reward_variant="r0_isaaclab_ball",
    play=play,
  )
  effective_probability = 0.0 if play else dropout_probability
  effective_episode_probability = 0.0 if play else episode_dropout_probability
  for term in cfg.observations["actor_history"].terms.values():
    term.params["dropout_probability"] = effective_probability
    term.params["episode_dropout_probability"] = effective_episode_probability
  cfg.terminations["ball_out_of_control"].params["ignore_episode_hidden"] = (
    effective_episode_probability > 0.0
  )
  return cfg


def unitree_g1_visibility_blend_flat_env_cfg(
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """A1R0 fusion of blind command tracking and bounded visible-ball recovery."""
  cfg = unitree_g1_factorial_dropout_flat_env_cfg(
    play=play,
    dropout_probability=0.0,
    episode_dropout_probability=0.05,
  )

  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=track_visibility_blended_linear_velocity,
    weight=2.0,
    params={
      "std": 0.5,
      "command_name": "twist",
      "target_ball_x": 0.25,
      "recovery_gain_x": 1.0,
      "recovery_gain_y": 1.5,
      "min_tolerance_x": 0.10,
      "min_tolerance_y": 0.08,
      "relative_tolerance": 0.20,
    },
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=track_visibility_blended_angular_velocity,
    weight=2.0,
    params={
      "std": 0.5,
      "command_name": "twist",
      "recovery_gain_yaw": 1.5,
      "min_tolerance_yaw": 0.15,
      "relative_tolerance": 0.20,
    },
  )
  cfg.rewards["command_velocity_envelope"] = RewardTermCfg(
    func=command_velocity_envelope_l2,
    weight=-4.0,
    params={
      "command_name": "twist",
      "min_tolerance_x": 0.10,
      "min_tolerance_y": 0.08,
      "min_tolerance_yaw": 0.15,
      "relative_tolerance": 0.20,
    },
  )
  cfg.rewards["track_ball_lin_vel_xy_exp"].params["gate_by_visibility"] = True
  cfg.rewards["ball_front_control"].params["gate_by_visibility"] = True
  cfg.terminations["ball_out_of_control"].params["ignore_when_ball_unseen"] = not play

  reset_params = cfg.events["reset_football"].params
  reset_params["ball_velocity_range"] = (0.0, 0.0) if play else (-0.4, 0.4)
  reset_params["stationary_ball_probability"] = 1.0 if play else 0.80
  if not play:
    cfg.events["kick_football"] = EventTermCfg(
      func=kick_football_velocity,
      mode="interval",
      interval_range_s=(5.0, 8.0),
      params={
        "probability": 0.10,
        "velocity_delta_range": (-0.4, 0.4),
        "ball_cfg": SceneEntityCfg("ball"),
      },
    )
  return cfg


def unitree_g1_dropout5_envelope30_flat_env_cfg(
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Dropout5 baseline plus a 30%-relative command-velocity envelope."""
  cfg = unitree_g1_factorial_dropout_flat_env_cfg(
    play=play,
    dropout_probability=0.0,
    episode_dropout_probability=0.05,
  )
  cfg.rewards["command_velocity_envelope"] = RewardTermCfg(
    func=command_velocity_envelope_l2,
    weight=-1.0,
    params={
      "command_name": "twist",
      "min_tolerance_x": 0.10,
      "min_tolerance_y": 0.10,
      "min_tolerance_yaw": 0.15,
      "relative_tolerance": 0.30,
    },
  )
  return cfg


def unitree_g1_transition_dropout25_envelope30_flat_env_cfg(
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """FB-BASE-0 with mid-episode sensor-loss and recovery transitions."""
  cfg = unitree_g1_dropout5_envelope30_flat_env_cfg(play=play)
  transition_probability = 0.0 if play else 0.25
  until_end_probability = 0.0 if play else 0.40
  for term in cfg.observations["actor_history"].terms.values():
    term.params["episode_dropout_probability"] = 0.0
    term.params["transition_dropout_probability"] = transition_probability
    term.params["transition_dropout_start_range_s"] = (2.0, 6.0)
    term.params["transition_dropout_duration_range_s"] = (0.2, 0.8)
    term.params["transition_dropout_until_end_probability"] = until_end_probability
    term.params["sensor_reward_fade_out_s"] = 0.5
    term.params["sensor_reward_fade_in_s"] = 0.5

  ball_termination = cfg.terminations["ball_out_of_control"].params
  ball_termination["ignore_episode_hidden"] = False
  ball_termination["ignore_when_sensor_hidden"] = not play
  cfg.rewards["track_ball_lin_vel_xy_exp"].params["gate_by_sensor_health"] = True
  cfg.rewards["ball_front_control"].params["gate_by_sensor_health"] = True
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=mdp.action_acc_l2,
    weight=-0.1,
  )
  cfg.metrics["ball_control_success"] = MetricsTermCfg(
    func=ball_control_zone_success,
  )
  cfg.curriculum["lin_vel_cmd_levels"] = CurriculumTermCfg(
    func=normal_control_lin_vel_cmd_levels,
    params={
      "command_name": "twist",
      "reward_term_name": "track_linear_velocity",
      "ball_control_metric_name": "ball_control_success",
      "action_acc_metric_name": "mean_action_acc",
      "max_lin_vel_x": (-0.5, 2.0),
      "max_lin_vel_y": (-0.5, 0.5),
      "tracking_threshold": 0.7,
      "ball_control_threshold": 0.3,
      "survival_threshold": 0.7,
      "action_acc_threshold": 0.8,
      "range_step": 0.1,
      "min_normal_episodes": 256,
      "validation_interval_steps": 12_000,
      "consecutive_successes": 3,
    },
  )
  return cfg


def unitree_g1_transition_dropout25_envelope30_legacy_curriculum_flat_env_cfg(
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Transition-dropout task with the original velocity-only curriculum."""
  cfg = unitree_g1_transition_dropout25_envelope30_flat_env_cfg(play=play)
  cfg.metrics.pop("ball_control_success")
  cfg.curriculum["lin_vel_cmd_levels"] = CurriculumTermCfg(
    func=lin_vel_cmd_levels,
    params={
      "command_name": "twist",
      "reward_term_name": "track_linear_velocity",
      "max_lin_vel_x": (-0.5, 2.0),
      "max_lin_vel_y": (-0.5, 0.5),
      "success_threshold": 0.7,
      "range_step": 0.1,
    },
  )
  return cfg


def unitree_g1_long_dropout10_envelope30_legacy_curriculum_flat_env_cfg(
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Legacy curriculum with 10% externally injected long sensor loss."""
  cfg = unitree_g1_transition_dropout25_envelope30_legacy_curriculum_flat_env_cfg(
    play=play
  )
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  command.standing_mode_per_episode = True

  # Dropout is sampled only from the 95% non-standing episodes, so 10/95
  # produces an unconditional 10% long-dropout share.
  transition_probability = 0.0 if play else 0.10 / 0.95
  until_end_probability = 0.0 if play else 1.0
  for term in cfg.observations["actor_history"].terms.values():
    term.params["transition_dropout_probability"] = transition_probability
    term.params["transition_dropout_until_end_probability"] = until_end_probability
    term.params["transition_excluded_standing_command_name"] = "twist"

  # Keep the visibility mask and long-dropout mechanism required by this task,
  # while matching the original Isaac Lab Actor observation randomization.
  _align_isaaclab_actor_observation_randomization(cfg)
  return cfg


def unitree_g1_visible_only_envelope30_legacy_curriculum_flat_env_cfg(
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """LongDropout10-matched coordinate Teacher without artificial ball loss."""
  cfg = unitree_g1_long_dropout10_envelope30_legacy_curriculum_flat_env_cfg(play=play)
  for term in cfg.observations["actor_history"].terms.values():
    term.params["dropout_probability"] = 0.0
    term.params["episode_dropout_probability"] = 0.0
    term.params["transition_dropout_probability"] = 0.0
    term.params["transition_dropout_until_end_probability"] = 0.0

  ball_termination = cfg.terminations["ball_out_of_control"].params
  ball_termination["ignore_episode_hidden"] = False
  ball_termination["ignore_when_sensor_hidden"] = False
  cfg.rewards["track_ball_lin_vel_xy_exp"].params["gate_by_sensor_health"] = False
  cfg.rewards["ball_front_control"].params["gate_by_sensor_health"] = False
  return cfg


def unitree_g1_long_dropout10_rough_curriculum10mm_env_cfg(
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Long-dropout baseline on a five-stage, 10 mm rough-terrain curriculum."""
  cfg = unitree_g1_long_dropout10_envelope30_legacy_curriculum_flat_env_cfg(play=play)
  cfg.sim.njmax = 512
  cfg.sim.nconmax = 128
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 50
  rough = HfRandomUniformTerrainCfg(
    size=(10.0, 10.0),
    noise_range=(0.0, 0.02),
    noise_step=0.002,
    vertical_scale=0.001,
    horizontal_scale=0.1,
    downsampled_scale=0.2,
    platform_width=1.5,
    scale_with_difficulty=True,
  )
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=42,
      curriculum=True,
      size=(10.0, 10.0),
      border_width=5.0,
      num_rows=6,
      num_cols=1,
      sub_terrains={"random_rough": rough},
    ),
    max_init_terrain_level=0,
  )
  if play:
    cfg.curriculum.pop("terrain_levels", None)
    assert cfg.scene.terrain.terrain_generator is not None
    cfg.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
    cfg.scene.terrain.terrain_generator.num_rows = 1
  else:
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=scheduled_rough_terrain_levels,
      params={
        "steps_per_level": 24_000,
        "max_level": 5,
        "start_step": 69_998 * 24,
      },
    )
  return cfg


def unitree_g1_visibility_blend_curriculum_v2_flat_env_cfg(
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """VisibilityBlend with mode-separated, multi-objective task curriculum."""
  cfg = unitree_g1_visibility_blend_flat_env_cfg(play=play)
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)

  # Keep raw user-command tracking active in both modes, then expose the
  # visible-recovery and hidden-command objectives as explicit reward terms.
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=track_linear_velocity,
    weight=1.0,
    params={"std": 0.5, "command_name": "twist"},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=track_angular_velocity,
    weight=1.0,
    params={"std": 0.5, "command_name": "twist"},
  )
  cfg.rewards["visible_recovery_linear_velocity"] = RewardTermCfg(
    func=track_visible_recovery_linear_velocity,
    weight=1.0,
    params={
      "std": 0.5,
      "command_name": "twist",
      "target_ball_x": 0.25,
      "recovery_gain_x": 1.0,
      "recovery_gain_y": 1.5,
      "min_tolerance_x": 0.10,
      "min_tolerance_y": 0.08,
      "relative_tolerance": 0.20,
    },
  )
  cfg.rewards["visible_recovery_angular_velocity"] = RewardTermCfg(
    func=track_visible_recovery_angular_velocity,
    weight=1.0,
    params={
      "std": 0.5,
      "command_name": "twist",
      "recovery_gain_yaw": 1.5,
      "min_tolerance_yaw": 0.15,
      "relative_tolerance": 0.20,
    },
  )
  cfg.rewards["hidden_command_linear_velocity"] = RewardTermCfg(
    func=track_hidden_linear_velocity,
    weight=1.0,
    params={"std": 0.5, "command_name": "twist"},
  )
  cfg.rewards["hidden_command_angular_velocity"] = RewardTermCfg(
    func=track_hidden_angular_velocity,
    weight=1.0,
    params={"std": 0.5, "command_name": "twist"},
  )

  # Actor observes only the causal visual stream; the Critic additionally sees
  # the privileged whole-episode hidden-mode flag.
  critic_mode_params = deepcopy(
    cfg.observations["actor_history"].terms["ball_visible_mask"].params
  )
  cfg.observations["critic"].terms["episode_ball_observation_hidden"] = (
    ObservationTermCfg(
      func=episode_ball_observation_hidden,
      params=critic_mode_params,
    )
  )

  if play:
    command.ranges.lin_vel_x = (-0.5, 1.6)
    command.ranges.lin_vel_y = (-0.5, 0.5)
    command.ranges.ang_vel_z = (-1.0, 1.0)
    return cfg

  stages = [
    {
      "name": "S0_fusion",
      "lin_vel_x": (-0.25, 1.0),
      "lin_vel_y": (-0.25, 0.25),
      "ang_vel_z": (-0.5, 0.5),
      "episode_dropout_probability": 0.20,
      "ball_velocity_range": (0.0, 0.0),
      "stationary_ball_probability": 1.0,
      "kick_probability": 0.0,
      "kick_velocity_delta_range": (0.0, 0.0),
      "hidden_xy_error_max": 0.35,
      "hidden_yaw_error_max": 0.55,
      "visible_xy_error_max": 0.40,
      "visible_yaw_error_max": 0.60,
      "visible_ball_control_min": 0.10,
      "envelope_compliance_min": 0.35,
      "episode_completion_min": 0.75,
    },
    {
      "name": "S1_turn_and_ball",
      "lin_vel_x": (-0.25, 1.0),
      "lin_vel_y": (-0.25, 0.25),
      "ang_vel_z": (-1.0, 1.0),
      "episode_dropout_probability": 0.10,
      "ball_velocity_range": (-0.2, 0.2),
      "stationary_ball_probability": 0.90,
      "kick_probability": 0.05,
      "kick_velocity_delta_range": (-0.2, 0.2),
      "hidden_xy_error_max": 0.32,
      "hidden_yaw_error_max": 0.52,
      "visible_xy_error_max": 0.37,
      "visible_yaw_error_max": 0.57,
      "visible_ball_control_min": 0.15,
      "envelope_compliance_min": 0.40,
      "episode_completion_min": 0.80,
    },
    {
      "name": "S2_medium_speed",
      "lin_vel_x": (-0.35, 1.3),
      "lin_vel_y": (-0.35, 0.35),
      "ang_vel_z": (-1.0, 1.0),
      "episode_dropout_probability": 0.05,
      "ball_velocity_range": (-0.2, 0.2),
      "stationary_ball_probability": 0.90,
      "kick_probability": 0.05,
      "kick_velocity_delta_range": (-0.2, 0.2),
      "hidden_xy_error_max": 0.29,
      "hidden_yaw_error_max": 0.49,
      "visible_xy_error_max": 0.34,
      "visible_yaw_error_max": 0.54,
      "visible_ball_control_min": 0.20,
      "envelope_compliance_min": 0.45,
      "episode_completion_min": 0.85,
    },
    {
      "name": "S3_high_speed",
      "lin_vel_x": (-0.5, 1.6),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-1.0, 1.0),
      "episode_dropout_probability": 0.05,
      "ball_velocity_range": (-0.3, 0.3),
      "stationary_ball_probability": 0.85,
      "kick_probability": 0.075,
      "kick_velocity_delta_range": (-0.3, 0.3),
      "hidden_xy_error_max": 0.27,
      "hidden_yaw_error_max": 0.47,
      "visible_xy_error_max": 0.32,
      "visible_yaw_error_max": 0.52,
      "visible_ball_control_min": 0.25,
      "envelope_compliance_min": 0.50,
      "episode_completion_min": 0.88,
    },
    {
      "name": "S4_final",
      "lin_vel_x": (-0.5, 1.6),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-1.0, 1.0),
      "episode_dropout_probability": 0.05,
      "ball_velocity_range": (-0.4, 0.4),
      "stationary_ball_probability": 0.80,
      "kick_probability": 0.10,
      "kick_velocity_delta_range": (-0.4, 0.4),
      "hidden_xy_error_max": 0.25,
      "hidden_yaw_error_max": 0.45,
      "visible_xy_error_max": 0.30,
      "visible_yaw_error_max": 0.50,
      "visible_ball_control_min": 0.30,
      "envelope_compliance_min": 0.55,
      "episode_completion_min": 0.90,
    },
  ]

  first = stages[0]
  command.ranges.lin_vel_x = first["lin_vel_x"]
  command.ranges.lin_vel_y = first["lin_vel_y"]
  command.ranges.ang_vel_z = first["ang_vel_z"]
  for term in cfg.observations["actor_history"].terms.values():
    term.params["episode_dropout_probability"] = first["episode_dropout_probability"]
    term.params["visibility_rise_alpha"] = 0.20
    term.params["visibility_fall_alpha"] = 0.05
  cfg.observations["critic"].terms["episode_ball_observation_hidden"].params[
    "episode_dropout_probability"
  ] = first["episode_dropout_probability"]
  cfg.events["reset_football"].params["ball_velocity_range"] = first[
    "ball_velocity_range"
  ]
  cfg.events["reset_football"].params["stationary_ball_probability"] = first[
    "stationary_ball_probability"
  ]
  cfg.events["kick_football"].params["probability"] = first["kick_probability"]
  cfg.events["kick_football"].params["velocity_delta_range"] = first[
    "kick_velocity_delta_range"
  ]

  cfg.metrics.update(
    {
      "user_command_error_xy": MetricsTermCfg(
        func=user_command_linear_velocity_error,
        params={"command_name": "twist"},
      ),
      "user_command_error_yaw": MetricsTermCfg(
        func=user_command_yaw_velocity_error,
        params={"command_name": "twist"},
      ),
      "command_envelope_violation": MetricsTermCfg(
        func=command_velocity_envelope_violation,
        params={"command_name": "twist", "smoothing_alpha": 0.10},
      ),
      "ball_control_success": MetricsTermCfg(
        func=ball_control_zone_success,
      ),
    }
  )
  cfg.curriculum = {
    "visibility_blend_task_levels": CurriculumTermCfg(
      func=visibility_blend_task_levels,
      params={
        "command_name": "twist",
        "stages": stages,
        "validation_interval_steps": 12_000,
        "consecutive_successes": 3,
        "hidden_xy_error_max": 0.25,
        "hidden_yaw_error_max": 0.45,
        "visible_xy_error_max": 0.30,
        "visible_yaw_error_max": 0.50,
        "visible_ball_control_min": 0.75,
        "envelope_compliance_min": 0.55,
        "episode_completion_min": 0.90,
      },
    )
  }
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
