"""CCBR Leo robot velocity environment configurations."""

from mjlab.asset_zoo.robots.ccbr_leo.leo_constants import (
  LEO_ACTION_SCALE,
  get_leo_robot_cfg,
  get_leo_robot_cfg_learned,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def ccbr_leo_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create CCBR Leo robot rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_leo_robot_cfg()}

  # Leo robot uses Ball_End_Effector collision geoms as foot contacts
  # Mapping: back_left, back_right, front_right, front_left
  geom_names = (
    "Ball_End_Effector_collision",      # back_left
    "Ball_End_Effector_2_collision",    # back_right
    "Ball_End_Effector_4_collision",    # front_right
    "Ball_End_Effector_3_collision",    # front_left
  )
  # Foot site names matching the XML sites: BL, BR, FR, FL
  site_names = ("BL", "BR", "FR", "FL")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, nonfoot_ground_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = LEO_ACTION_SCALE

  cfg.viewer.body_name = "base"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Configure foot_height observation with foot sites
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  # Update joint name patterns for leo robot naming convention
  cfg.rewards["pose"].params["std_standing"] = {
    r"^(back|front)_(left|right)_hip_(roll|pitch)$": 0.05,
    r"^(back|front)_(left|right)_knee_pitch$": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r"^(back|front)_(left|right)_hip_(roll|pitch)$": 0.3,
    r"^(back|front)_(left|right)_knee_pitch$": 0.6,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r"^(back|front)_(left|right)_hip_(roll|pitch)$": 0.3,
    r"^(back|front)_(left|right)_knee_pitch$": 0.6,
  }
  cfg.rewards["pose"].weight = 0.25

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)

  # Configure foot site-based rewards with foot sites
  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    if reward_name in cfg.rewards:
      cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.0

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    assert cfg.commands is not None
    assert type(cfg.commands["twist"]) is UniformVelocityCommandCfg
    cfg.commands["twist"].ranges.lin_vel_x = (-0.0, 0.0)
    cfg.commands["twist"].ranges.lin_vel_y = (-0.0, 0.0)
    cfg.commands["twist"].resampling_time_range = (1.0, 3.0)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def ccbr_leo_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create CCBR Leo robot flat terrain velocity configuration."""
  cfg = ccbr_leo_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  if cfg.curriculum is not None and "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"][1]["step"] = 500 * 24
    cfg.curriculum["command_vel"].params["velocity_stages"][2]["step"] = 1000 * 24
  
  assert cfg.curriculum is not None
  del cfg.curriculum["terrain_levels"]

  return cfg


def ccbr_leo_flat_env_cfg_learned(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = ccbr_leo_flat_env_cfg(play=play)
  cfg.scene.entities["robot"] = get_leo_robot_cfg_learned()
  return cfg
