"""Unitree Go1 velocity tracking environment configurations."""

from copy import deepcopy

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import (
  GO1_ACTION_SCALE,
  get_go1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import VIEWER_CONFIG, create_velocity_env_cfg
from mjlab.utils.retval import retval


@retval
def UNITREE_GO1_ROUGH_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 rough terrain velocity tracking configuration."""
  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

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
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  cfg = create_velocity_env_cfg(
    robot_cfg=get_go1_robot_cfg(),
    action_scale=GO1_ACTION_SCALE,
    viewer_body_name="trunk",
    site_names=site_names,
    feet_sensor_cfg=feet_ground_cfg,
    self_collision_sensor_cfg=nonfoot_ground_cfg,
    foot_friction_geom_names=geom_names,
    posture_std_standing={
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
    },
    posture_std_walking={
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    },
    posture_std_running={
      r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
      r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    },
    body_ang_vel_weight=0.0,
    angular_momentum_weight=0.0,
    self_collision_weight=0.0,
    air_time_weight=0.0,
  )

  cfg.viewer = deepcopy(VIEWER_CONFIG)
  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  assert cfg.terminations is not None
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "nonfoot_ground_touch"},
  )
  return cfg


@retval
def UNITREE_GO1_FLAT_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 flat terrain velocity tracking configuration."""
  # Start with rough terrain config.
  cfg = deepcopy(UNITREE_GO1_ROUGH_ENV_CFG)

  # Change to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  del cfg.curriculum["terrain_levels"]

  return cfg
