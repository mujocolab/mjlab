from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  G1_ROBOT_CFG,
)
from mjlab.tasks.tracking.tracking_env_cfg import (
  create_tracking_env_cfg,
)
from mjlab.utils.spec_config import ContactSensorCfg


def create_g1_flat_env_cfg():
  """Create configuration for Unitree G1 robot tracking on flat terrain."""
  # Configure self collision sensor.
  self_collision_sensor = ContactSensorCfg(
    name="self_collision",
    subtree1="pelvis",
    subtree2="pelvis",
    data=("found",),
    reduce="netforce",
    num=10,  # Report up to 10 contacts.
  )
  g1_cfg = replace(G1_ROBOT_CFG, sensors=(self_collision_sensor,))

  # Create configuration with G1-specific parameters.
  # Matching the original which always had motion command configured.
  cfg = create_tracking_env_cfg(
    robot_cfg=g1_cfg,
    action_scale=G1_ACTION_SCALE,
    viewer_body_name="torso_link",
    # Motion command parameters (matching original with empty file)
    motion_file="",  # Empty file - needs actual motion data to work.
    reference_body="torso_link",
    body_names=[
      "pelvis",
      "left_hip_roll_link",
      "left_knee_link",
      "left_ankle_roll_link",
      "right_hip_roll_link",
      "right_knee_link",
      "right_ankle_roll_link",
      "torso_link",
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_yaw_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_yaw_link",
    ],
    pose_range={
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    velocity_range={
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.2, 0.2),
      "roll": (-0.52, 0.52),
      "pitch": (-0.52, 0.52),
      "yaw": (-0.78, 0.78),
    },
    joint_position_range=(-0.1, 0.1),
    foot_friction_geom_names=[r"^(left|right)_foot[1-7]_collision$"],
    ee_body_names=[
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    ],
  )

  # Add base_com event with body ipos randomization (matching original)
  # Note: Using randomize_field since randomize_com doesn't exist.
  from mjlab.envs import mdp as envs_mdp
  from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  cfg.events["base_com"] = EventTerm(
    mode="startup",
    func=envs_mdp.randomize_field,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
      "operation": "add",
      "field": "body_ipos",
      "ranges": {
        0: (-0.025, 0.025),
        1: (-0.05, 0.05),
        2: (-0.05, 0.05),
      },
    },
  )

  return cfg


def create_g1_flat_env_cfg_play():
  """Create play configuration for Unitree G1 robot tracking on flat terrain."""
  cfg = create_g1_flat_env_cfg()

  # Disable corruption for play mode.
  assert "policy" in cfg.observations, "Policy observations must be configured"
  cfg.observations["policy"].enable_corruption = False

  # Remove push robot event.
  assert "push_robot" in cfg.events, (
    "Push robot event should be configured in base tracking config"
  )
  del cfg.events["push_robot"]

  # Update motion command randomization ranges for play mode.
  from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

  assert cfg.commands is not None, "Commands must be configured for tracking"
  assert "motion" in cfg.commands, "Motion command must be configured for tracking"

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg), (
    f"Expected MotionCommandCfg, got {type(motion_cmd)}"
  )

  # Set empty ranges to disable randomization in play mode.
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}

  # Effectively infinite episode length.
  cfg.episode_length_s = int(1e9)

  return cfg


# Create config instances.
G1_FLAT_ENV_CFG = create_g1_flat_env_cfg()
G1_FLAT_ENV_CFG_PLAY = create_g1_flat_env_cfg_play()
