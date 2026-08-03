"""
速度跟踪任务基础环境配置工厂函数
功能说明：
1. 基于MjLab Mujoco强化学习框架 ManagerBasedRlEnvCfg 实现通用四足/双足机器人速度跟踪训练模板
2. 完整封装RL全链路模块：传感器、观测器（策略/价值网络分离特权信息）、动作、速度指令、奖励、终止条件、域随机事件、课程学习、仿真与可视化
3. 设计为公共基础模板：机器人专属配置调用本函数生成基础cfg，再通过dataclass.replace()覆盖空占位参数（机身/足端帧、关节/刚体/碰撞组名称、奖励权重、动作缩放、地形难度等）

模块分块介绍：
1. 传感器：环境地形雷达扫描 + 足端离地高度探测，用于地形感知与步态奖励计算
2. 观测组：
   - Actor策略观测：带噪声本体感知、速度指令、地形雷达（推理时使用）
   - Critic价值观测：无噪声特权信息（真实关节角度、足端接触/腾空时间/接触力）仅训练价值网络使用
3. 动作空间：关节位置增量控制，通配符匹配机器人全部执行器
4. 速度指令生成器：随机采样线速度X/Y + 偏航角速度指令，按比例分配静止/原地转向/直行环境，定时重新采样
5. 域随机化事件（DR）：
   - 回合重置：机身位姿随机、关节复位至中立位
   - 周期扰动：间隔施加随机外力/力矩干扰机器人
   - 启动一次性随机：足端摩擦系数、编码器零偏误差、机身质心偏移
6. 奖励函数：
   - 主奖励：线速度/角速度跟踪、机身直立保持
   - 姿态正则：根据行走速度自适应关节姿态代价
   - 惩罚项：关节限位、动作剧烈抖动、足端打滑、抬脚过高、落地冲击
   - 步态辅助奖励：足端腾空时长正则（权重需分机型自定义）
7. 回合终止条件：训练时长耗尽、机身倾斜超阈值（摔倒）、跑出训练地形区域
8. 课程学习渐进训练：
   - 地形难度自动提升：根据行走性能提高地形粗糙等级
   - 指令速度范围分阶段扩大，循序渐进提升训练难度
9. 仿真与场景配置：粗糙程序化地形、Mujoco求解器参数、控制频率降采样、最大单回合时长限制
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity_football import mdp
from mjlab.tasks.velocity_football.mdp.velocity_command import (
  UniformVelocityCommandCfg,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from .football import get_football_cfg


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """
  生成基础速度跟踪强化学习环境配置模板
  所有空白占位参数（机身frame、刚体/关节/足端名称、奖励权重、动作缩放等）
  需要在机器人专属配置文件中通过 dataclass.replace() 自定义覆盖
  返回一套完整包含传感器/观测/奖励/课程学习的通用足式机器人行走环境。
  """

  ##
  # 传感器模块：地形雷达扫描 + 足端高度探测器，实现地形感知
  ##
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # 机身坐标系，分机型单独配置
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # 仅检测地形碰撞组，忽略机器人自身
    debug_vis=True,
  )

  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(),  # 足端探测坐标系，分机型配置frame与探测点阵
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # 仅检测地形
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),  # 可视化射线：品红色
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )

  ##
  # 观测空间：分为Actor策略观测（带噪声）、Critic价值观测（特权无噪声信息）
  ##
  actor_terms = {
    # 参考足球任务不向Actor提供机身线速度；需要时可恢复：
    # "base_lin_vel": ObservationTermCfg(
    #     func=mdp.builtin_sensor,
    #     params={"sensor_name": "robot/imu_lin_vel"},
    #     noise=Unoise(n_min=-0.5, n_max=0.5),
    # ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),  # IMU角速度观测噪声
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),  # 机身投影重力噪声
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},  # Actor带关节零偏噪声
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      clip=(-10.0, 10.0),
      delay_min_lag=0,
      delay_max_lag=2,
    ),
    "ball_pos_b": ObservationTermCfg(
      func=mdp.ball_pos_b_with_fixed_bias,
      params={"bias_range": 0.10},
      noise=Unoise(n_min=-0.06, n_max=0.06),
      delay_min_lag=0,
      delay_max_lag=2,
    ),
    "ball_to_feet_vectors_b": ObservationTermCfg(
      func=mdp.ball_to_feet_vectors_b,
      params={
        "ball_cfg": SceneEntityCfg("ball"),
        "asset_cfg": SceneEntityCfg("robot", body_names=(r".*_ankle_roll_link",)),
      },
      noise=Unoise(n_min=-0.1, n_max=0.1),
      delay_min_lag=0,
      delay_max_lag=2,
    ),
    # 无限平面不需要Actor地形扫描；恢复粗糙地形时可恢复：
    # "height_scan": ObservationTermCfg(
    #     func=envs_mdp.height_scan,
    #     params={"sensor_name": "terrain_scan"},
    #     noise=Unoise(n_min=-0.1, n_max=0.1),
    #     scale=1 / terrain_scan.max_distance,
    # ),
  }

  critic_terms = {
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action, clip=(-10.0, 10.0)),
    "ball_vel_b": ObservationTermCfg(func=mdp.ball_vel_b),
    "ball_pos_b": ObservationTermCfg(func=mdp.ball_pos_b),
    "ball_to_feet_vectors_b": ObservationTermCfg(
      func=mdp.ball_to_feet_vectors_b,
      params={
        "ball_cfg": SceneEntityCfg("ball"),
        "asset_cfg": SceneEntityCfg("robot", body_names=(r".*_ankle_roll_link",)),
      },
    ),
    # 以下为原生速度任务特权观测，足球参考方案暂不使用：
    # 恢复 height_scan 时，同时恢复 mjlab.envs.mdp 的 envs_mdp import。
    # "height_scan": ObservationTermCfg(
    #     func=envs_mdp.height_scan,
    #     params={"sensor_name": "terrain_scan"},
    #     scale=1 / terrain_scan.max_distance,
    # ),
    # "foot_height": ObservationTermCfg(
    #     func=mdp.foot_height,
    #     params={"sensor_name": "foot_height_scan"},
    # ),
    # "foot_air_time": ObservationTermCfg(
    #     func=mdp.foot_air_time,
    #     params={"sensor_name": "feet_ground_contact"},
    # ),
    # "foot_contact": ObservationTermCfg(
    #     func=mdp.foot_contact,
    #     params={"sensor_name": "feet_ground_contact"},
    # ),
    # "foot_contact_forces": ObservationTermCfg(
    #     func=mdp.foot_contact_forces,
    #     params={"sensor_name": "feet_ground_contact"},
    # ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,  # 开启观测噪声
      history_length=5,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,  # 价值网络无观测噪声
      history_length=5,
      flatten_history_dim=True,
    ),
  }

  ##
  # 指标统计：训练过程诊断指标
  ##
  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # 动作空间：关节位置增量控制
  ##
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),  # 匹配机器人所有关节执行器
      scale=0.5,  # 动作缩放系数，分机型覆盖
      use_default_offset=True,
    )
  }

  ##
  # 速度指令生成器：随机生成线速度+偏航角速度指令
  ##
  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(5.0, 6.0),  # 指令重新采样间隔5~6秒
      rel_standing_envs=0.05,  # 5%环境保持静止
      zero_command_ramp_time_range=(0.3, 0.5),
      rel_heading_envs=1.0,  # 所有非站立环境使用目标朝向
      rel_forward_envs=0.0,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.25, 1.0),  # X向线速度范围
        lin_vel_y=(-0.25, 0.25),  # Y向线速度范围
        ang_vel_z=(-1.0, 1.0),  # 偏航角速度范围
        heading=(-math.pi, math.pi),  # 目标朝向角度
      ),
    )
  }

  ##
  # 事件管理器：回合重置、周期扰动、启动域随机化DR
  ##
  events = {
    # 同时重置机器人和足球，保证足球位置始终相对机器人生成。
    "reset_football": EventTermCfg(
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
    ),
    # 原生独立机身重置备选；恢复时需停用上面的协调重置：
    # "reset_base": EventTermCfg(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "z": (0.01, 0.05),
    #             "yaw": (-3.14, 3.14),
    #         },
    #         "velocity_range": {},
    #     },
    # ),
    # 关节在默认姿态附近按参考范围随机重置。
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # 周期性随机外力扰动机器人
    "push_robot": EventTermCfg(
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
        },
      },
      # 强扰动备选：interval=(1.0, 3.0)，速度范围分别为
      # x±0.5、y±0.5、z±0.4、roll/pitch±0.52、yaw±0.78。
    ),
    # 环境初始化：随机足端地面摩擦系数
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # 分机型指定足端碰撞体
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # 所有足端使用同一个摩擦随机值
      },
    ),
    # 环境初始化：仅随机足球滑动摩擦，扭转和滚动摩擦保持资产基线值。
    "ball_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_collision",)),
        "operation": "abs",
        "ranges": (0.05, 0.15),
        "axes": [0],
        "shared_random": True,
      },
    ),
    # 初始化：关节编码器零偏噪声
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    # 初始化：随机机身质心偏移
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # 指定机身刚体
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }

  ##
  # 奖励函数集合：主跟踪奖励、姿态正则、各类惩罚项
  ##
  rewards = {
    # 非超时终止惩罚；当前覆盖摔倒，后续也覆盖足球失控终止。
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    # 执行器力矩L2正则，限制高能耗和过激触球。
    "joint_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-1e-5,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=r".*"),
      },
    ),
    # 关节加速度L2正则，抑制高频抖动。
    "joint_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-1e-7,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",))},
    ),
    # 足球全局平面速度跟踪：控球任务的主运动目标。
    "track_ball_lin_vel_xy_exp": RewardTermCfg(
      func=mdp.track_ball_lin_vel_xy_exp,
      weight=2.0,
      params={
        "command_name": "twist",
        "std": 0.8,
        "control_x_range": (0.05, 0.45),
        "control_y_abs": 0.15,
        "gate_std_x": 0.10,
        "gate_std_y": 0.05,
        "use_user_command": True,
      },
    ),
    # 机器人线速度跟踪：保留为较低权重的辅助项，避免机器人原地控球。
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    # 角速度跟踪奖励
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.5,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    # 足球与机器人 pelvis 的瞬时相对平面速度趋近于零（无时间窗口）。
    "track_ball_relative_vel_xy_exp": RewardTermCfg(
      func=mdp.track_ball_relative_vel_xy_exp,
      weight=0.25,
      params={"std": 0.5},
    ),
    # 足球保持在 pelvis 前方固定锚点附近。
    "track_ball_relative_pos_xy_exp": RewardTermCfg(
      func=mdp.track_ball_relative_pos_xy_exp,
      weight=0.5,
      params={
        "command_name": "twist",
        "anchor_x": 0.19,
        "anchor_x_speed_gain": 0.0,
        "anchor_x_range": (0.19, 0.19),
        "std_x": 0.5,
        "std_y": 0.5,
      },
    ),
    # 超出软控球区域时惩罚；区域内成本为零。
    "ball_outside_control_zone": RewardTermCfg(
      func=mdp.ball_outside_control_zone_l2,
      weight=-0.5,
      params={
        "x_range": (0.05, 0.45),
        "y_abs": 0.15,
        "std_x": 0.10,
        "std_y": 0.05,
      },
    ),
    # 机身直立保持奖励
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # 指定机身刚体
      },
    ),
    # 自适应关节姿态正则（静止/行走/快跑三套代价）
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {},  # 静止姿态标准差，分机型配置
        "std_walking": {},  # 行走姿态标准差
        "std_running": {},  # 快跑姿态标准差
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    # 机身多余角速度惩罚（默认权重0，分机型开启）
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=0.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
    ),
    # 角动量惩罚（默认关闭）
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=0.0,
      params={"sensor_name": "robot/root_angmom"},
    ),
    # 关节超出限位惩罚
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    # 动作变化率L2惩罚，抑制剧烈抖动
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.2),
    # 足端合理腾空时长奖励（步态辅助）
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,  # 分机型自定义权重
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.5,
      },
    ),
    # 抬脚过高惩罚
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.1,
        "height_sensor_name": "foot_height_scan",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # 足端探测点
      },
    ),
    # 摆动足高度偏差惩罚
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    # 足端打滑惩罚
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
    ),
    # 落地冲击惩罚（软着陆）
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
  }

  ##
  # 回合终止条件
  ##
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),  # 达到最大时长
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": 0.8},  # 参考阈值，约45.84度
      # 若早期运球动作频繁误终止，可恢复：math.radians(70.0)
    ),
    "ball_out_of_control": TerminationTermCfg(
      func=mdp.ball_out_of_control,
      params={
        "max_distance": 1.5,
        "min_forward": 0.0,
        "max_forward": 1.0,
        "max_lateral": 0.5,
        "max_height": 0.5,
        "ball_cfg": SceneEntityCfg("ball"),
      },
    ),
    "out_of_terrain_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds,
      time_out=True,  # 跑出训练地形区域直接结束回合
    ),
  }

  ##
  # 课程学习：渐进式提升训练难度
  ##
  curriculum = {
    # 根据行走性能自动提升地形粗糙等级
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    # 与 Isaac Lab 参考一致：跟踪奖励达标后逐级扩大线速度范围。
    "lin_vel_cmd_levels": CurriculumTermCfg(
      func=mdp.lin_vel_cmd_levels,
      params={
        "command_name": "twist",
        "reward_term_name": "track_linear_velocity",
        "max_lin_vel_x": (-0.5, 2.0),
        "max_lin_vel_y": (-0.5, 0.5),
        "success_threshold": 0.7,
        "range_step": 0.1,
      },
    ),
  }

  ##
  # 组装完整环境配置并返回
  ##
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      # MuJoCo 平面在 XY 方向无限延伸。
      terrain=TerrainEntityCfg(terrain_type="plane"),
      # 后续恢复粗糙地形时，将上面一行替换为：
      # terrain=TerrainEntityCfg(
      #     terrain_type="generator",
      #     terrain_generator=replace(ROUGH_TERRAINS_CFG),
      #     max_init_terrain_level=5,
      # ),
      # 并恢复 dataclasses.replace 和 ROUGH_TERRAINS_CFG 的 import。
      entities={"ball": get_football_cfg()},
      sensors=(terrain_scan, foot_height_scan),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # 跟随观察的机身，分机型配置
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,  # 最大碰撞约束数量
      njmax=1500,  # 最大关节约束数量
      mujoco=MujocoCfg(
        timestep=0.005,  # Mujoco物理步长 5ms
        iterations=10,  # 求解迭代次数
        ls_iterations=20,  # 线搜索迭代次数
      ),
    ),
    decimation=4,  # 控制降采样：每4次物理步执行一次RL策略
    episode_length_s=20.0,  # 单回合最大时长20秒
  )
