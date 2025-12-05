"""Booster T1 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  T1_ACTION_SCALE,
  get_t1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def booster_t1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_t1_robot_cfg()}

  # T1 foot geometries - using foot link and sphere contacts from holosoma
  geom_names = (
    "left_foot_link",
    "left_foot_sphere_1_link",
    "left_foot_sphere_2_link",
    "left_foot_sphere_3_link",
    "left_foot_sphere_4_link",
    "left_foot_sphere_5_link",
    "right_foot_link",
    "right_foot_sphere_1_link",
    "right_foot_sphere_2_link",
    "right_foot_sphere_3_link",
    "right_foot_sphere_4_link",
    "right_foot_sphere_5_link",
  )

  # Contact sensors using actual T1 body structure
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(left_foot_link|right_foot_link)$",
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
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = T1_ACTION_SCALE

  cfg.viewer.body_name = "Trunk"

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.0

  # Remove foot_height observation since we don't have foot sites
  cfg.observations["critic"].terms.pop("foot_height", None)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  # T1-specific pose standards (adjusted for 23 DOF humanoid)
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body
    r".*Hip_Pitch.*": 0.3,
    r".*Hip_Roll.*": 0.15,
    r".*Hip_Yaw.*": 0.15,
    r".*Knee_Pitch.*": 0.35,
    r".*Ankle_Pitch.*": 0.25,
    r".*Ankle_Roll.*": 0.1,
    # Waist
    r".*Waist.*": 0.15,
    # Arms
    r".*Shoulder_Pitch.*": 0.15,
    r".*Shoulder_Roll.*": 0.15,
    r".*Elbow_Pitch.*": 0.15,
    r".*Elbow_Yaw.*": 0.15,
    # Head
    r".*Head.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body
    r".*Hip_Pitch.*": 0.5,
    r".*Hip_Roll.*": 0.2,
    r".*Hip_Yaw.*": 0.2,
    r".*Knee_Pitch.*": 0.6,
    r".*Ankle_Pitch.*": 0.35,
    r".*Ankle_Roll.*": 0.15,
    # Waist
    r".*Waist.*": 0.2,
    # Arms
    r".*Shoulder_Pitch.*": 0.5,
    r".*Shoulder_Roll.*": 0.2,
    r".*Elbow_Pitch.*": 0.35,
    r".*Elbow_Yaw.*": 0.2,
    # Head
    r".*Head.*": 0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)

  # Remove foot-site-dependent rewards since T1 doesn't have foot sites defined
  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards.pop(reward_name, None)

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Apply play mode overrides
  if play:
    # Effectively infinite episode length
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def booster_t1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 flat terrain velocity configuration."""
  cfg = booster_t1_rough_env_cfg(play=play)

  # Switch to flat terrain
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
