from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import rl_lab.tasks.locomotion.velocity.mdp as mdp

"""
IsaacLab人形机器人带球速度跟踪任务完整环境配置
任务目标：机器人跟随速度指令行走，同时保持足球在身前可控范围
框架结构：场景+指令+动作+观测(策略/价值分离)+域随机事件+奖励+终止条件+课程学习
"""

##
# 场景资源定义：地面、足球、机器人、灯光、足端接触力传感器
##
@configclass
class MySceneCfg(InteractiveSceneCfg):
    # 平面地面地形
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.05,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # 足球刚体配置
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.1098,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.5,
                dynamic_friction=0.15,
                restitution=0.5,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.55, 0.10)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, 0.1098)),
    )
    # 人形机器人模型，外部传入
    robot: ArticulationCfg = MISSING
    # 平行太阳光
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    # 天空环境光
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    # 全身接触力传感器，记录接触时长与力大小
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )


##
# MDP 决策流程配置
##
@configclass
class CommandsCfg:
    """速度指令生成器：随机线速度X/Y + 偏航角速度"""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 6.0), # 指令5~6秒随机更换
        rel_standing_envs=0.05,           # 5%环境静止不动
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.25, 1.0), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-1.0, 1.0), heading=(-3.14, 3.14)
        ),
    )


@configclass
class ActionsCfg:
    """动作空间：所有关节位置增量控制"""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,          # 动作缩放幅度
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """观测分为两组：Policy带噪声用于推理；Critic无噪声特权信息用于训练价值网络"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        phase = ObsTerm(func=mdp.phase, params={"period": 0.6, "command_name": "base_velocity"}) # 步态周期相位
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(
            func=mdp.delayed_last_action,
            params={"delay_steps": 2}, # 延迟2步历史动作
            clip=(-10.0, 10.0),
        )
        ball_pos_b = ObsTerm( # 机身坐标系足球位置
            func=mdp.ball_pos_b,
            params={"delay_steps": 2},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        ball_to_feet_vectors_b = ObsTerm( # 机身坐标系球到各脚踝向量
            func=mdp.ball_to_feet_vectors_b,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
                "delay_steps": 2,
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.enable_corruption = True   # 开启观测噪声
            self.concatenate_terms = True
            self.history_length = 5         # 观测序列长度5帧

    @configclass
    class CriticCfg(ObsGroup):
        # 价值网络无噪声特权观测，包含球速度等额外信息
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        phase = ObsTerm(func=mdp.phase, params={"period": 0.6, "command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))
        ball_vel_b = ObsTerm(func=mdp.ball_vel_b)
        ball_pos_b = ObsTerm(func=mdp.ball_pos_b)
        ball_to_feet_vectors_b = ObsTerm(
            func=mdp.ball_to_feet_vectors_b,
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False  # 关闭噪声
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """域随机化 & 回合重置事件"""
    # 环境初始化一次：机器人全身摩擦随机
    randomize_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )
    # 初始化：足球物理材质随机
    randomize_ball_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": (0.25, 0.5),
            "dynamic_friction_range": (0.05, 0.15),
            "restitution_range": (0.25, 0.5),
            "num_buckets": 32,
        },
    )
    # 每回合重置：关节微小偏移随机
    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.0, 0.0),
        },
    )
    # 每回合重置机器人与足球相对位置
    randomize_reset_football = EventTerm(
        func=mdp.reset_football,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
            "ball_radius": 0.1098,
            "robot_xy_noise_range": (-0.05, 0.05),
            "robot_yaw_range": (-3.14, 3.14),
            "ball_forward_range": (0.1, 0.5),
            "ball_lateral_range": (-0.15, 0.15),
            "ball_velocity_range": (-1.5, 1.5),
        },
    )
    # 周期外力扰动机器人机身
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 6.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.3, 0.3),
                "z": (-0.2, 0.2),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            }
        },
    )


@configclass
class RewardsCfg:
    """全套奖励函数：生存惩罚、姿态正则、速度跟踪、带球控制"""
    # 摔倒/终止大额惩罚
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # 机身姿态惩罚，防止倾倒
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    # 关节功耗、加速度、偏离默认姿态惩罚
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-1e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-1e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    default_joint_pos_l2 = RewTerm(
        func=mdp.joint_deviation_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    # 腿部、手臂姿态正则指数奖励
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.5,
        params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*", "waist_.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_exp,
        weight=0.5,
        params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_elbow.*", ".*_wrist.*"])},
    )
    # 左右关节对称约束惩罚
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["left_hip_pitch.*", "right_shoulder_pitch.*"],
                ["right_hip_pitch.*", "left_shoulder_pitch.*"],
            ],
        },
    )
    # 关节限位惩罚
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    # 非足/手部位意外接触惩罚
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[r"(?!.*(ankle_roll|wrist_yaw)).*"]),
            "threshold": 1.0,
        },
    )

    # 动作平滑惩罚，抑制剧烈抖动
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

    # 核心：机身速度跟踪奖励
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # 跟随指令同步带动足球速度奖励
    track_ball_lin_vel_xy_exp = RewTerm(
        func=mdp.track_ball_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # 步态约束：绊倒、打滑惩罚
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # 保持足球在身前可控区域主奖励
    ball_front_control = RewTerm(
        func=mdp.ball_front_control,
        weight=0.5,
        params={
            "x_range": (0.1, 0.4),
            "y_abs": 0.15,
        },
    )


@configclass
class TerminationsCfg:
    """回合终止判定条件"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True) # 时长耗尽
    bad_orientation = DoneTerm( # 机身倾斜过大摔倒
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": 0.8
        },
    )
    ball_out_of_control = DoneTerm( # 足球距离过远失去控制
        func=mdp.ball_out_of_control,
        params={
            "max_distance": 1.5,
            "min_forward": -0.5,
            "max_lateral": 0.5,
            "max_height": 0.5,
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )


@configclass
class CurriculumCfg:
    """课程学习：随性能逐步放大允许最大前进速度"""
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "max_lin_vel_x": (-0.5, 2.0),
            "max_lin_vel_y": (-0.5, 0.5),
        },
    )


@configclass
class LocomotionVelocityFootballEnvCfg(ManagerBasedRLEnvCfg):
    # 顶层环境总配置，整合所有模块
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5) # 并行环境数量、环境间距
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4                # 策略每4次物理步进执行一次
        self.episode_length_s = 20.0       # 单回合最大20秒
        self.sim.dt = 0.005                # PhysX物理步长5ms
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
