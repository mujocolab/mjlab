.. _environment_config:

环境配置
========

单个 ``ManagerBasedRlEnvCfg`` dataclass 完整描述一个 mjlab 环境：物理
世界、智能体与世界的接口，以及其上定义的 MDP。因为所有内容都在一个
扁平对象里，环境可以被检视、复制和修改，而无需在类层次结构中穿行。

在阅读本页之前想对 mjlab 建立整体认识，请从 :ref:`architecture_overview`
开始。


.. _env-config-skeleton:

带注释的骨架
------------

``ManagerBasedRlEnvCfg`` 的全部字段如下所示，附有行内注释。标记为
``...`` 的字段必须提供；其余字段都有默认值。

.. code-block:: python

    from dataclasses import dataclass, field

    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.managers.action_manager import ActionTermCfg
    from mjlab.managers.command_manager import CommandTermCfg
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene.scene import SceneCfg
    from mjlab.sim.sim import SimulationCfg
    from mjlab.viewer.viewer_config import ViewerConfig


    @dataclass
    class MyEnvCfg(ManagerBasedRlEnvCfg):

        # --- Physics ---

        decimation: int = 4
        # Number of physics steps per policy step.
        # Environment step duration = sim.mujoco.timestep * decimation.

        sim: SimulationCfg = field(default_factory=SimulationCfg)
        # Physics parameters: timestep, integrator, solver, contact settings.
        # Default timestep is 0.002 s (500 Hz). Override with MujocoCfg.

        scene: SceneCfg = ...
        # Terrain, entities, and sensors. Also sets num_envs.
        # Required; there is no default.

        # --- Episode ---

        episode_length_s: float = 20.0
        # Episode duration in seconds.
        # Steps = ceil(episode_length_s / (sim.mujoco.timestep * decimation)).

        is_finite_horizon: bool = False
        # False (default): time limit is an artificial cutoff. The agent
        #   receives a truncated signal and bootstraps value beyond the limit.
        # True: time limit defines the task boundary. The agent receives a
        #   terminal done signal with no future value beyond it.

        scale_rewards_by_dt: bool = True
        # When True (default), each reward term is multiplied by step_dt so
        # that cumulative episodic sums are invariant to simulation frequency.
        # Set to False for algorithms that expect unscaled reward signals.

        # --- Managers ---

        observations: dict[str, ObservationGroupCfg] = field(default_factory=dict)
        # Observation groups. Each key is a group name (e.g. "actor", "critic").
        # Groups can differ in noise, history, delay, and concatenation.

        actions: dict[str, ActionTermCfg] = field(default_factory=dict)
        # Action terms. Each term controls one slice of the policy output
        # and routes it to a specific entity's actuators.

        rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
        # Reward terms. The manager computes a weighted sum each step.

        terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
        # Termination conditions. If empty, episodes never terminate early.
        # Add a time_out term to enforce the episode length limit.

        events: dict[str, EventTermCfg] = field(
            default_factory=lambda: {
                "reset_scene_to_default": EventTermCfg(
                    func=reset_scene_to_default,
                    mode="reset",
                )
            }
        )
        # Event terms for domain randomization and state resets.
        # The default includes reset_scene_to_default, which resets all
        # entities to their initial pose each episode. Override this dict
        # to replace or extend the default reset behavior.

        commands: dict[str, CommandTermCfg] = field(default_factory=dict)
        # Command generators (e.g. velocity targets for locomotion).
        # Commands are resampled at configurable intervals and on reset.

        curriculum: dict[str, CurriculumTermCfg] = field(default_factory=dict)
        # Curriculum terms that adjust training conditions based on performance.

        metrics: dict[str, MetricsTermCfg] = field(default_factory=dict)
        # Custom metrics logged as episode averages alongside reward terms.

        # --- Misc ---

        seed: int | None = None
        # Random seed for reproducibility. If None, a random seed is chosen
        # and stored back into this field after initialization.

        viewer: ViewerConfig = field(default_factory=ViewerConfig)
        # Camera position, resolution, and tracking target for rendering.


.. _env-config-term-pattern:

项配置模式
----------

所有管理器字典都遵循同一模式。每个条目把字符串名称映射到一个项配置
对象。配置至少携带指向实现该可调用对象的 ``func`` 字段，以及转发给该
可调用对象的额外关键字参数字典 ``params``。

管理器每步调用 ``func(env, **params)``（当 ``func`` 是已实例化的类时
则调用 ``term(env, **params)``）。项名是任意的；它们出现在训练日志中，
仅用于标识。

.. rubric:: 奖励项

.. code-block:: python

    from mjlab.envs import mdp
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    rewards = {
        "alive": RewardTermCfg(
            func=mdp.is_alive,
            weight=1.0,
        ),
        "joint_torques": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-1e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "action_rate": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.1,
        ),
    }

``weight`` 在函数输出累加进总奖励之前对其缩放。负权重产生惩罚。

``params`` 映射到函数的关键字参数。例如
``mdp.joint_torques_l2(env, asset_cfg=...)`` 的 ``asset_cfg`` 来自
``params`` 字典。任何未在 ``params`` 中列出的参数必须在函数签名中带
默认值。

.. rubric:: 终止项

.. code-block:: python

    from mjlab.envs import mdp
    from mjlab.managers.termination_manager import TerminationTermCfg

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,   # marks this as a truncation, not a failure
        ),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.22},   # ~70 degrees in radians
        ),
    }

``TerminationTermCfg`` 的 ``time_out`` 标志告诉管理器把该条件当作截断
而非终止失败。截断映射到 Gym 接口的 ``truncated`` 信号；失败映射到
``terminated``。这一区分对 RL 算法中的价值自举很重要。

.. rubric:: 事件项

.. code-block:: python

    from mjlab.managers.event_manager import EventTermCfg

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"yaw": (-3.14, 3.14)},
                "velocity_range": {},
            },
        ),
    }

``EventTermCfg`` 的 ``mode`` 字段控制各项何时触发：启动时、回合重置时
或按固定间隔。生命周期模式、内置事件函数以及事件与域随机化关系的完整
论述见 :ref:`events`。

.. rubric:: 基于函数 vs 基于类的项

项可以是普通函数或类。函数适合无状态计算；当项需要缓存昂贵的初始化或
跨步维护状态时，类更有用。

基于函数的项签名为 ``func(env, **params) -> Tensor``。基于类的项以
``(cfg, env)`` 实例化一次，之后以相同签名调用。类可以选择实现
``reset(env_ids)`` 钩子来清除回合内状态。

.. code-block:: python

    # Function-based (stateless)
    RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.01)

    # Class-based (caches joint indices at init)
    class MyReward:
        def __init__(self, cfg, env):
            self.joint_ids = resolve_joint_ids(cfg.params, env)

        def __call__(self, env) -> torch.Tensor:
            return compute_reward(env, self.joint_ids)

    RewardTermCfg(func=MyReward, weight=1.0)


.. _env-config-timing:

时序：decimation、timestep 与回合长度
--------------------------------------

三个参数共同决定环境的时间结构。

``sim.mujoco.timestep``
    物理积分步长，单位秒。默认 0.002 s（500 Hz）。这是任何环境中最
    重要的参数之一：取值越小物理越稳定，但仿真速度越慢。选择时间步长
    与求解器设置的实用建议见 MuJoCo 的
    `性能调优 <https://mujoco.readthedocs.io/en/stable/modeling.html#performance-tuning>`_
    指南。

``decimation``
    每个策略步执行的物理步数。策略运行频率为
    ``1 / (timestep * decimation)`` Hz。

``episode_length_s``
    回合时长（秒）。每回合的最大策略步数为
    ``ceil(episode_length_s / (timestep * decimation))``。

**具体示例。** 速度任务使用 ``timestep=0.005``（200 Hz 物理）和
``decimation=4``，策略频率为 50 Hz。``episode_length_s=20.0`` 时，每个
回合恰好运行 1000 个策略步。

.. code-block:: python

    physics_dt  = 0.005        # seconds per physics step (200 Hz)
    decimation  = 4            # physics steps per policy step
    step_dt     = 0.005 * 4   # = 0.02 s per policy step (50 Hz)
    episode_len = 20.0 / 0.02  # = 1000 policy steps per episode

运行时读取这些值可使用环境属性：

.. code-block:: python

    env.physics_dt          # = cfg.sim.mujoco.timestep
    env.step_dt             # = cfg.sim.mujoco.timestep * cfg.decimation
    env.max_episode_length  # steps (int)
    env.max_episode_length_s  # seconds (float)

``scale_rewards_by_dt=True``（默认）时，每个奖励项在返回前乘以
``step_dt``。返回常量 1.0 的奖励函数每步贡献 ``step_dt``，整个回合
约贡献 ``episode_length_s``，无论 ``decimation`` 和 ``timestep`` 如何
设置。在不关闭该缩放的情况下改变仿真频率，奖励量级保持不变。


.. _env-config-subclassing:

子类化模式
----------

mjlab 使用普通 dataclass 继承，而不是深层嵌套的类层次结构。构建任务
专用配置的方式是子类化 ``ManagerBasedRlEnvCfg`` 并覆盖字段。

推荐做法是在工厂函数中定义完整配置，然后由仅覆盖差异字段的机器人
专用配置调用它。速度任务就使用这一模式：``make_velocity_env_cfg``
返回一个组装完毕的 ``ManagerBasedRlEnvCfg``，每个机器人配置调用该
工厂并修补机器人专属的值，如场景、关节名模式和动作缩放。

工厂函数的精简版展示了完整的组装模式：

.. code-block:: python

    import math
    from dataclasses import replace

    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp import dr
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.tasks.velocity import mdp
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.terrains.config import ROUGH_TERRAINS_CFG
    from mjlab.viewer import ViewerConfig


    def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:

        observations = {
            "actor": ObservationGroupCfg(
                terms={
                    "base_lin_vel": ObservationTermCfg(
                        func=mdp.builtin_sensor,
                        params={"sensor_name": "robot/imu_lin_vel"},
                    ),
                    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                    "command": ObservationTermCfg(
                        func=mdp.generated_commands,
                        params={"command_name": "twist"},
                    ),
                    # additional terms omitted for brevity
                },
                concatenate_terms=True,
                enable_corruption=True,
            ),
            "critic": ObservationGroupCfg(
                terms={...},
                concatenate_terms=True,
                enable_corruption=False,
            ),
        }

        actions = {
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.5,
                use_default_offset=True,
            )
        }

        commands = {
            "twist": UniformVelocityCommandCfg(
                entity_name="robot",
                resampling_time_range=(3.0, 8.0),
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-1.0, 1.0),
                    ang_vel_z=(-0.5, 0.5),
                    heading=(-math.pi, math.pi),
                ),
            )
        }

        events = {
            "reset_base": EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {},
                },
            ),
            "foot_friction": EventTermCfg(
                mode="startup",
                func=dr.geom_friction,
                params={
                    "asset_cfg": SceneEntityCfg("robot", geom_names=[]),
                    "operation": "abs",
                    "ranges": (0.3, 1.2),
                },
            ),
            "push_robot": EventTermCfg(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(1.0, 3.0),
                params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
            ),
        }

        rewards = {
            "track_linear_velocity": RewardTermCfg(
                func=mdp.track_linear_velocity,
                weight=2.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            ),
            "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
            "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        }

        terminations = {
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "fell_over": TerminationTermCfg(
                func=mdp.bad_orientation,
                params={"limit_angle": math.radians(70.0)},
            ),
        }

        return ManagerBasedRlEnvCfg(
            decimation=4,
            episode_length_s=20.0,
            sim=SimulationCfg(
                nconmax=35,
                njmax=1500,
                mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
            ),
            scene=SceneCfg(
                terrain=TerrainEntityCfg(
                    terrain_type="generator",
                    terrain_generator=replace(ROUGH_TERRAINS_CFG),
                    max_init_terrain_level=5,
                ),
                num_envs=1,
            ),
            observations=observations,
            actions=actions,
            commands=commands,
            events=events,
            rewards=rewards,
            terminations=terminations,
        )

机器人专用配置调用该工厂并用 ``dataclasses.replace`` 或直接赋值修补
字段。常见的按机器人覆盖包括 ``scene``（添加机器人实体与传感器）、
``SceneEntityCfg`` 内的关节名模式、动作 ``scale``，以及奖励项的 body
名称。

.. note::

   Isaac Lab 使用深层嵌套的 ``__post_init__`` 覆盖来实现配置继承。
   mjlab 刻意避开那种模式：每个 ``ManagerBasedRlEnvCfg`` 都是扁平、
   可检视的 dataclass。拼错的字段名会在构造时抛出 ``TypeError``，而不是
   悄悄创建一个新属性。完整对比见 :ref:`migration_isaac_lab`。


下一步
------

管理器层一节的其余页面逐一深入介绍每个管理器：

- :ref:`observations`：观测组、处理流水线（裁剪、缩放、噪声、延迟、
  历史）与内置观测函数。
- :ref:`actions`：动作类型，以及动作管理器如何把策略输出路由到执行器。
- :ref:`rewards`：奖励项与按 dt 缩放。
- :ref:`terminations`：回合结束条件与截断/失败之分。
- :ref:`commands`：指令生成器与目标条件任务设置。
- :ref:`events`：事件管理器生命周期（启动、重置、间隔）。
- :ref:`domain_randomization`：用于域随机化的完整 ``dr`` 模块。
- :ref:`curriculum`：基于策略表现的难度递进。
- :ref:`metrics`：以回合平均值记录的自定义每步指标。
