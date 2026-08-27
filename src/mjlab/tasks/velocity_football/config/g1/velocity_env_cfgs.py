"""Unitree G1 football-compatible velocity pretraining configurations."""

from copy import deepcopy

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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity_football import mdp as football_mdp
from mjlab.tasks.velocity_football.velocity_env_cfg import make_velocity_env_cfg

from .pose import get_isaaclab_default_keyframe

KLAVIER_JOINT_ORDER = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
)


def unitree_g1_velocity_pretrain_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the G1 velocity pretraining configuration."""
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
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
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


def unitree_g1_velocity_pretrain_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the flat G1 velocity pretraining configuration."""
  cfg = unitree_g1_velocity_pretrain_rough_env_cfg(play=play)

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


def unitree_g1_klavier_replica_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Reproduce Klavier ``unitree_g1_flat`` with its copied G1 MJCF model."""
  cfg = unitree_g1_velocity_pretrain_flat_env_cfg(play=play)
  cfg.scene.num_envs = 4096
  cfg.scene.extent = 2.5
  cfg.scene.entities["robot"] = get_g1_klavier_robot_cfg()
  cfg.scene.entities["robot"].init_state = get_isaaclab_default_keyframe()

  ordered_joints = SceneEntityCfg(
    "robot",
    joint_names=KLAVIER_JOINT_ORDER,
    preserve_order=True,
  )
  for group_name in ("actor", "critic"):
    group = cfg.observations[group_name]
    group.history_length = 5
    group.flatten_history_dim = True
    group.terms["joint_pos"].params["asset_cfg"] = deepcopy(ordered_joints)
    group.terms["joint_vel"].params["asset_cfg"] = deepcopy(ordered_joints)
  action_obs = cfg.observations["actor"].terms["actions"]
  action_obs.delay_min_lag = 0
  action_obs.delay_max_lag = 2
  cfg.observations["critic"].terms["joint_torques"] = ObservationTermCfg(
    func=football_mdp.joint_effort,
    params={"asset_cfg": deepcopy(ordered_joints)},
  )

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = KLAVIER_JOINT_ORDER
  action.preserve_order = True
  action.scale = G1_ACTION_SCALE

  cfg.events["reset_base"].params = {
    "pose_range": {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "yaw": (-3.14, 3.14),
    },
    "velocity_range": {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.5, 0.5),
      "roll": (-0.5, 0.5),
      "pitch": (-0.5, 0.5),
      "yaw": (-0.5, 0.5),
    },
  }
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

  undesired_cfg = ContactSensorCfg(
    name="klavier_undesired_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(?!.*(ankle_roll|wrist_yaw)).*$",
      entity="robot",
    ),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (undesired_cfg,)

  robot_all_joints = SceneEntityCfg("robot", joint_names=(r".*",))
  feet_bodies = SceneEntityCfg(
    "robot",
    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    preserve_order=True,
  )
  cfg.rewards = {
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=football_mdp.klavier_track_lin_vel_xy_exp,
      weight=1.0,
      params={"command_name": "twist", "std": 0.5},
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=football_mdp.klavier_track_ang_vel_z_exp,
      weight=1.0,
      params={"command_name": "twist", "std": 0.5},
    ),
    "lin_vel_z_l2": RewardTermCfg(func=football_mdp.klavier_lin_vel_z_l2, weight=-2.0),
    "ang_vel_xy_l2": RewardTermCfg(func=football_mdp.klavier_ang_vel_xy_l2, weight=-0.1),
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.flat_orientation_l2, weight=-1.0
    ),
    "body_orientation_l2": RewardTermCfg(
      func=football_mdp.klavier_body_orientation_l2,
      weight=-5.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    ),
    "joint_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-1e-5,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=r".*")},
    ),
    "joint_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-1e-7,
      params={"asset_cfg": deepcopy(robot_all_joints)},
    ),
    "default_joint_pos_l2": RewardTermCfg(
      func=football_mdp.klavier_joint_deviation_l2,
      weight=-0.05,
      params={"asset_cfg": deepcopy(robot_all_joints)},
    ),
    "joint_deviation_legs_exp": RewardTermCfg(
      func=football_mdp.klavier_joint_deviation_exp,
      weight=0.5,
      params={
        "std": 0.4,
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(r".*_hip_yaw.*", r".*_hip_roll.*", r"waist_.*"),
        ),
      },
    ),
    "joint_deviation_arms_exp": RewardTermCfg(
      func=football_mdp.klavier_joint_deviation_exp,
      weight=0.5,
      params={
        "std": 0.5,
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(
            r".*_shoulder_roll.*",
            r".*_shoulder_yaw.*",
            r".*_elbow.*",
            r".*_wrist.*",
          ),
        ),
      },
    ),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "joint_mirror": RewardTermCfg(
      func=football_mdp.klavier_joint_mirror,
      weight=-0.25,
      params={
        "mirror_joints": (
          ("left_hip_pitch_joint", "right_shoulder_pitch_joint"),
          ("right_hip_pitch_joint", "left_shoulder_pitch_joint"),
        )
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "undesired_contacts": RewardTermCfg(
      func=football_mdp.klavier_undesired_contacts,
      weight=-1.0,
      params={"sensor_name": undesired_cfg.name, "threshold": 1.0},
    ),
    "contact_forces": RewardTermCfg(
      func=football_mdp.klavier_contact_forces,
      weight=-1e-3,
      params={"sensor_name": "feet_ground_contact", "threshold": 300.0},
    ),
    "feet_slide": RewardTermCfg(
      func=football_mdp.klavier_feet_slide,
      weight=-0.25,
      params={"sensor_name": "feet_ground_contact", "asset_cfg": feet_bodies},
    ),
    "feet_gait": RewardTermCfg(
      func=football_mdp.klavier_feet_gait,
      weight=0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "period": 0.6,
        "offset": (0.0, 0.5),
        "threshold": 0.56,
        "command_name": "twist",
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=football_mdp.klavier_feet_air_time,
      weight=1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "threshold": 0.3,
      },
    ),
    "stand_still_without_cmd": RewardTermCfg(
      func=football_mdp.klavier_stand_still_without_cmd,
      weight=-1.0,
      params={"command_name": "twist", "asset_cfg": robot_all_joints},
    ),
  }
  cfg.curriculum = {
    "lin_vel_cmd_levels": CurriculumTermCfg(
      func=football_mdp.lin_vel_cmd_levels,
      params={
        "command_name": "twist",
        "reward_term_name": "track_lin_vel_xy_exp",
        "max_lin_vel_x": (-1.0, 2.0),
        "max_lin_vel_y": (-1.0, 1.0),
      },
    ),
    "push_velocity_levels": CurriculumTermCfg(
      func=football_mdp.push_velocity_levels,
      params={
        "event_term_name": "push_robot",
        "max_velocity_range": {
          "x": (-1.5, 1.5),
          "y": (-1.0, 1.0),
          "z": (-0.5, 0.5),
          "roll": (-0.8, 0.8),
          "pitch": (-0.8, 0.8),
          "yaw": (-1.57, 1.57),
        },
      },
    ),
  }

  cfg.sim.mujoco.timestep = 0.005
  cfg.decimation = 4
  cfg.episode_length_s = 20.0
  cfg.viewer.body_name = "torso_link"
  if play:
    cfg.scene.num_envs = 1
    cfg.curriculum = {}
  return cfg


def unitree_g1_current_velocity_pretrain_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the shared current-frame MLP walking pretraining task."""
  cfg = unitree_g1_velocity_pretrain_flat_env_cfg(play=play)
  for group_name in ("actor", "critic"):
    group = cfg.observations[group_name]
    group.history_length = None
    group.flatten_history_dim = True
  return cfg


def unitree_g1_temporal_velocity_pretrain_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the 10-frame TemporalCNN walking pretraining task."""
  cfg = unitree_g1_velocity_pretrain_flat_env_cfg(play=play)

  actor = cfg.observations["actor"]
  actor.history_length = None
  actor.flatten_history_dim = True
  actor_history = deepcopy(actor)
  actor_history.history_length = 10
  actor_history.flatten_history_dim = False
  cfg.observations["actor_history"] = actor_history

  critic = cfg.observations["critic"]
  critic.history_length = None
  critic.flatten_history_dim = True
  critic_history = deepcopy(critic)
  critic_history.history_length = 10
  critic_history.flatten_history_dim = False
  cfg.observations["critic_history"] = critic_history

  return cfg
