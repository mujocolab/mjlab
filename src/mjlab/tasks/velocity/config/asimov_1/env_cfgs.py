"""Asimov 1 velocity environment configurations.

Based on the Unitree G1 config but adapted for the Asimov 1 robot.
Includes sim2real features from asimov legs-only config.

Key differences from G1:
- Asimov has waist_yaw only (no waist_roll/pitch)
- Asimov uses pelvis_link (not pelvis)
- Asimov uses waist_yaw_link as torso (not torso_link)
- Asimov foot is a single elongated link with four collision spheres

Key differences from asimov legs-only:
- Straight hips (not canted 45°)
- Upper body joints (waist, arms)
"""

import math
import os

from mjlab import MJLAB_SRC_PATH
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab.asset_zoo.robots import (
    ASIMOV_1_ACTION_SCALE,
    get_asimov_1_robot_cfg,
)


def asimov_1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Asimov 1 rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 70

    cfg.scene.entities = {"robot": get_asimov_1_robot_cfg()}

    site_names = ("left_foot", "right_foot")

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "pelvis_link"
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=site_name, entity="robot")
                for site_name in site_names
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

    # Foot collision: 4 G1-style spheres per foot at heel/toe corners.
    geom_names = (
        # Left foot
        "left_foot1_collision",
        "left_foot2_collision",
        "left_foot3_collision",
        "left_foot4_collision",
        # Right foot
        "right_foot1_collision",
        "right_foot2_collision",
        "right_foot3_collision",
        "right_foot4_collision",
    )

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
        primary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis_link", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    # Body-ground contact for termination (holosoma: terminate if non-foot body hits ground)
    # Only check bodies that should NEVER touch ground: knees, hips, torso, arms
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=(
                "pelvis_link",
                "left_knee_link", "right_knee_link",
                "left_hip_pitch_link", "right_hip_pitch_link",
                "waist_yaw_link",
                "left_elbow_link", "right_elbow_link",
            ),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
        body_ground_cfg,
    )

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = dict(ASIMOV_1_ACTION_SCALE)
    joint_pos_action.lpf_cutoff_freq = 10.0
    cfg.viewer.body_name = "waist_yaw_link"

    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 0.9  # Asimov torso height

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.8, 1.2)  # tighter DR
    # NOTE: base_com is created further below (it is not in the base cfg), so
    # its body_names are set there directly.

    # =========================================================================
    # STAGGERED OBSERVATION DELAYS (matching real CAN slot timing)
    # =========================================================================
    # All 5 buses read in parallel, delay based on SLOT position:
    #
    # | Slot | Bus 0 (L Leg)  | Bus 1 (R Leg)  | Bus 2 (L Arm)      | Bus 3 (R Arm)      | Bus 4 (Waist) |
    # |------|----------------|----------------|--------------------|--------------------|---------------|
    # | 0    | L_hip_pitch    | R_hip_pitch    | L_shoulder_pitch   | R_shoulder_pitch   | waist_yaw     |
    # | 1    | L_hip_roll     | R_hip_roll     | L_shoulder_roll    | R_shoulder_roll    | (neck_pitch)  |
    # | 2    | L_hip_yaw      | R_hip_yaw      | L_shoulder_yaw     | R_shoulder_yaw     | (neck_yaw)    |
    # | 3    | L_knee         | R_knee         | L_elbow            | R_elbow            | —             |
    # | 4    | L_ankle_pitch  | R_ankle_pitch  | L_wrist_yaw        | R_wrist_yaw        | —             |
    # | 5    | L_ankle_roll   | R_ankle_roll   | —                  | —                  | —             |
    #
    # Slot 0-1 (oldest, delay 2): hip_pitch, hip_roll, shoulder_pitch, shoulder_roll, waist_yaw
    # Slot 2-3 (middle, delay 1): hip_yaw, knee, shoulder_yaw, elbow
    # Slot 4-5 (fresh, delay 0):  ankle_pitch, ankle_roll, wrist_yaw
    #
    SLOT_0_1 = (
        # Slot 0
        "left_hip_pitch_joint", "right_hip_pitch_joint",
        "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
        "waist_yaw_joint",
        # Slot 1
        "left_hip_roll_joint", "right_hip_roll_joint",
        "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    )
    SLOT_2_3 = (
        # Slot 2
        "left_hip_yaw_joint", "right_hip_yaw_joint",
        "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
        # Slot 3
        "left_knee_joint", "right_knee_joint",
        "left_elbow_joint", "right_elbow_joint",
    )
    SLOT_4_5 = (
        # Slot 4
        "left_ankle_pitch_joint", "right_ankle_pitch_joint",
        "left_wrist_yaw_joint", "right_wrist_yaw_joint",
        # Slot 5
        "left_ankle_roll_joint", "right_ankle_roll_joint",
    )
    # TOE_JOINTS removed — toes removed from sim XML

    # Save existing terms we want to keep
    actor_base_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=0.25,
            delay_min_lag=0,
            delay_max_lag=1,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity_imu,
            params={"sensor_name": "robot/imu_quat"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            delay_min_lag=0,
            delay_max_lag=2,
        ),
        "command": cfg.observations["actor"].terms["command"],
    }
    actor_actions = cfg.observations["actor"].terms["actions"]

    critic_base_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=0.25,
            delay_min_lag=0,
            delay_max_lag=1,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity_imu,
            params={"sensor_name": "robot/imu_quat"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "command": cfg.observations["critic"].terms["command"],
    }
    critic_actions = cfg.observations["critic"].terms["actions"]
    critic_privileged = {
        "base_lin_vel": cfg.observations["critic"].terms["base_lin_vel"],
        "foot_height": cfg.observations["critic"].terms["foot_height"],
        "foot_air_time": cfg.observations["critic"].terms["foot_air_time"],
        "foot_contact": cfg.observations["critic"].terms["foot_contact"],
        "foot_contact_forces": cfg.observations["critic"].terms["foot_contact_forces"],
        # toe_joint_pos/vel removed — toes removed from sim
    }

    # Build actor terms with staggered delays by CAN slot.
    cfg.observations["actor"].terms = {
        **actor_base_terms,
        # Joint position by slot timing
        "joint_pos_slot01": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SLOT_0_1)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        ),
        "joint_pos_slot23": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SLOT_2_3)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        ),
        "joint_pos_slot45": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SLOT_4_5)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        ),
        # Joint velocity by slot timing
        "joint_vel_slot01": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SLOT_0_1)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.1,
        ),
        "joint_vel_slot23": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SLOT_2_3)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.1,
        ),
        "joint_vel_slot45": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SLOT_4_5)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.1,
        ),
        "actions": actor_actions,
        "gait_clock": ObservationTermCfg(
            func=mdp.gait_clock,
            params={"command_name": "twist"},
        ),
    }

    # Build critic terms with the same slot structure but no added obs lag.
    cfg.observations["critic"].terms = {
        **critic_base_terms,
        # Joint position by slot timing
        "joint_pos_slot01": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SLOT_0_1)},
            scale=1.0,
            delay_min_lag=0,
            delay_max_lag=0,
        ),
        "joint_pos_slot23": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SLOT_2_3)},
            scale=1.0,
            delay_min_lag=0,
            delay_max_lag=0,
        ),
        "joint_pos_slot45": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SLOT_4_5)},
            scale=1.0,
            delay_min_lag=0,
            delay_max_lag=0,
        ),
        # Joint velocity by slot timing
        "joint_vel_slot01": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SLOT_0_1)},
            scale=1.0,
            delay_min_lag=0,
            delay_max_lag=0,
        ),
        "joint_vel_slot23": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SLOT_2_3)},
            scale=1.0,
            delay_min_lag=0,
            delay_max_lag=0,
        ),
        "joint_vel_slot45": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SLOT_4_5)},
            scale=1.0,
            delay_min_lag=0,
            delay_max_lag=0,
        ),
        "actions": critic_actions,
        "gait_clock": ObservationTermCfg(
            func=mdp.gait_clock,
            params={"command_name": "twist"},
        ),
        # Privileged info at the end
        **critic_privileged,
    }

    # =========================================================================
    # DOMAIN RANDOMIZATION (from asimov legs-only)
    # =========================================================================

    # Default joint position randomization (calibration error)
    # ±0.02 rad (~1.1 deg) offset - simulates encoder zero offset.
    # mjlab dr.qpos0 randomizes the model's qpos0 (default joint pose) field.
    cfg.events["qpos0_rand"] = EventTermCfg(
        mode="startup",
        func=dr.qpos0,
        params={
            "ranges": (-0.02, 0.02),  # match legs-only
            "operation": "add",
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )

    # PD gains DR: ×[0.8, 1.2] (light)
    cfg.events["pd_gains_rand"] = EventTermCfg(
        mode="reset",
        func=dr.pd_gains,
        params={
            "kp_range": (0.8, 1.2),
            "kd_range": (0.8, 1.2),
            "operation": "scale",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Torso COM offset (payload on torso). mjlab dr.body_com_offset takes a
    # per-axis ranges dict like the ported body_ipos randomization.
    cfg.events["base_com"] = EventTermCfg(
        mode="startup",
        func=dr.body_com_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("waist_yaw_link",)),
            "operation": "add",
            "ranges": {
                0: (0.0, 0.05),
                1: (0.0, 0.0),
                2: (0.03, 0.07),
            },
        },
    )

    # Joint reset — keep default offset (0, 0), matching legs-only

    # Joint reset randomization (match G1: ±0.5 on pos and vel)
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.5, 0.5)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.5, 0.5)

    # Velocity-injection push (G1/atom01/TienKung style) — directly overrides base
    # velocity so PD cannot passively resist. Forces reactive stepping, including
    # lateral (vy) which is critical for sidestep learning.
    cfg.events.pop("push_robot", None)
    cfg.events["push_robot"] = EventTermCfg(
        func=envs_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),   # lateral — forces sidestep
                "z": (-0.3, 0.3),
                "roll": (-0.4, 0.4),
                "pitch": (-0.4, 0.4),
                "yaw": (-0.5, 0.5),
            }
        },
    )
    # =========================================================
    # POSE REWARD (speed-adaptive, proven in legs-only)
    # Relaxed upper body stds to match G1 (waist, shoulder_roll, wrist)
    # =========================================================
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot",
        joint_names=(
            ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint",
            ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
            "waist_yaw_joint",
            ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_yaw_joint",
        ),
    )
    cfg.rewards["pose"].params["walking_threshold"] = 0.1
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.15, r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.5, r".*ankle_pitch.*": 0.15, r".*ankle_roll.*": 0.1,
        r".*waist_yaw.*": 0.15,
        r".*shoulder_pitch.*": 0.15, r".*shoulder_roll.*": 0.1,
        r".*shoulder_yaw.*": 0.1, r".*elbow.*": 0.1, r".*wrist.*": 0.1,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.25, r".*hip_yaw.*": 0.25,
        r".*knee.*": 0.5, r".*ankle_pitch.*": 0.25, r".*ankle_roll.*": 0.1,
        r".*waist_yaw.*": 0.25,
        r".*shoulder_pitch.*": 0.25, r".*shoulder_roll.*": 0.1,
        r".*shoulder_yaw.*": 0.1, r".*elbow.*": 0.1, r".*wrist.*": 0.1,
    }

    # =========================================================================
    # BODY-SPECIFIC REWARDS - Use waist_yaw_link as Asimov's "torso"
    # =========================================================================
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("waist_yaw_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("waist_yaw_link",)

    for reward_name in ["foot_clearance", "foot_slip", "foot_swing_height"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # =========================================================================
    # ASIMOV 1 TRAINING REWARDS (23 DOF)
    # Based on the working legs-only config, adapted for the full robot
    # =========================================================================

    # =========================================================================
    # v171 reward stack adapted to the current API.
    # =========================================================================

    # Tracking
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.5
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = 0.7071

    # Positive task rewards
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["pose"].weight = 1.0
    cfg.rewards["air_time"].weight = 0.5

    # Penalties
    cfg.rewards["foot_clearance"].weight = -2.0
    cfg.rewards["foot_swing_height"].weight = -0.25
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards["body_ang_vel"].weight = -0.08
    cfg.rewards["angular_momentum"].weight = -0.03
    cfg.rewards["dof_pos_limits"].weight = -1.0
    cfg.rewards["soft_landing"].weight = -1e-5

    # Contact shaping
    cfg.rewards["feet_stumble"] = RewardTermCfg(
        func=mdp.feet_stumble,
        weight=-1.25,
        params={"sensor_name": feet_ground_cfg.name, "ratio_threshold": 4.0},
    )
    cfg.rewards["feet_contact_force_limit"] = RewardTermCfg(
        func=mdp.feet_contact_force_limit,
        weight=-5e-4,
        params={"sensor_name": feet_ground_cfg.name, "max_force": 350.0},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )
    ref_gait_csv = os.path.join(
        MJLAB_SRC_PATH,
        "asset_zoo",
        "robots",
        "asimov_1",
        "imitation_data",
        "reference_gait_forward.csv",
    )
    leg_joints = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )
    cfg.rewards["imitation_gait"] = RewardTermCfg(
        func=mdp.imitation_reward,
        weight=1.0,
        params={
            "csv_path": ref_gait_csv,
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", joint_names=leg_joints),
            "scale": 3.0,
            "command_threshold": 0.1,
            "forward_only": True,
            "lateral_yaw_tol": 0.2,
        },
    )
    cfg.rewards["knee_swing"] = RewardTermCfg(
        func=mdp.knee_swing_reward,
        weight=0.5,
        params={
            "sensor_name": feet_ground_cfg.name,
            "command_name": "twist",
            "command_threshold": 0.1,
            "target_flexion": 0.87,
            "forward_only": True,
            "lateral_yaw_tol": 0.2,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=("left_knee_joint", "right_knee_joint")
            ),
        },
    )

    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(70.0)

    # =========================================================================
    # VELOCITY - Full speed from start (matches working legs-only config)
    # =========================================================================
    twist_cmd.ranges.lin_vel_x = (-0.8, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.6, 0.6)
    twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)

    twist_cmd.rel_standing_envs = 0.3
    twist_cmd.gait_freq_base = 0.5
    twist_cmd.gait_freq_speed_scale = 1.5

    # Velocity curriculum removed — full range from start.
    assert cfg.curriculum is not None
    cfg.curriculum.pop("command_vel", None)

    # No force curriculum — fixed ±50N from start

    # Use default push_robot and reset_base (same as legs-only)

    # Apply play mode overrides
    if play:
        cfg.episode_length_s = int(1e9)

        # Disable observation noise/delay corruption
        cfg.observations["actor"].enable_corruption = False

        # Disable all domain randomization so the robot loads in its nominal pose
        for _ev in ("push_robot", "qpos0_rand", "pd_gains_rand", "base_com", "foot_friction"):
            cfg.events.pop(_ev, None)

        # Zero out joint reset jitter so init pose is exactly the nominal crouch
        cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
        cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)

        # Zero out base pose/velocity randomization on reset
        if "reset_base" in cfg.events:
            params = cfg.events["reset_base"].params
            params["pose_range"] = {k: (0.0, 0.0) for k in params.get("pose_range", {})}
            params["velocity_range"] = {k: (0.0, 0.0) for k in params.get("velocity_range", {})}

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


def asimov_1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Asimov 1 flat terrain velocity configuration."""
    cfg = asimov_1_rough_env_cfg(play=play)

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum.
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
