.. _observations:

观测
====

观测定义智能体在每步感知到什么。观测管理器把各个观测项组装成策略接收
的输入张量。每个观测项都要经过一条可配置的处理流水线：噪声注入、裁剪、
缩放、传感器延迟和历史堆叠。


观测组
------

每个组是一个 ``ObservationGroupCfg``，持有把字符串名称映射到
``ObservationTermCfg`` 条目的 ``terms`` 字典。管理器按注册顺序沿最后一维
拼接各观测项的输出。

.. code-block:: python

    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.envs.mdp import observations as obs_fns

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                "projected_gravity": ObservationTermCfg(
                    func=obs_fns.projected_gravity
                ),
                "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel),
                "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                "last_action": ObservationTermCfg(func=obs_fns.last_action),
            },
            enable_corruption=True,
        ),
    }

该字典传给 ``ManagerBasedRlEnvCfg(observations=...)``。观测管理器在
初始化时解析各项函数，并在此刻分配所需的历史或延迟缓冲区。

默认情况下，组内各项的输出沿最后一维拼接为单个 ``[num_envs, D]``
张量。设置 ``concatenate_terms=False`` 可改为接收把项名映射到独立
张量的字典。

``enable_corruption`` 标志控制整个组的噪声开关：为 ``False`` 时各项上
的噪声配置被忽略。这使得在带噪声的 actor 组与无噪声的 critic 组之间
共享项定义变得非常直接，见下文 :ref:`非对称 actor-critic <obs-asymmetric>`
一节。

历史与延迟也可以在组级别设置，统一作用于组内所有项；见
:ref:`obs-history-delay`。


处理流水线
----------

每一步，每个组的每一项都依次通过以下流水线：

.. code-block:: text

    compute → noise → clip → scale → delay → history

1. **compute**：调用观测项函数。必须返回 ``[num_envs, D]`` 张量。

2. **noise**：若组上 ``enable_corruption=True`` 且该项配置了 ``noise``，
   则施加噪声。无状态噪声（``NoiseCfg``）直接施加；有状态噪声
   （``NoiseModelCfg``）由管理器跨步维护。

3. **clip**：若项上设置了 ``clip=(lo, hi)``，数值被截断到该范围。

4. **scale**：若设置了 ``scale``，输出按元素相乘。接受标量、元组或
   张量。

5. **delay**：若 ``delay_max_lag > 0``，该项的输出存入环形缓冲区，
   返回较早一步的值。见 :ref:`obs-history-delay`。

6. **history**：若 ``history_length > 0``，堆叠过去的输出。见
   :ref:`obs-history-delay`。

.. note::

   延迟先于历史施加。这模拟了真实系统中旧传感器读数被缓冲的情形：
   历史堆叠的是延迟后的观测，而不是未来的观测。


.. _obs-history-delay:

观测历史与延迟
--------------

观测支持两个时序特性：历史与延迟。历史堆叠过去的帧，给策略提供时间
上下文；延迟通过返回较早时间步的观测来建模传感器延迟。

两者都通过 ``ObservationTermCfg`` 上的字段按项配置，也可以在
``ObservationGroupCfg`` 的组级别设置、统一作用于组内所有项。项级设置
覆盖组级设置。

历史
^^^^

设置 ``history_length=N`` 会堆叠某项最近 N 次的输出。
``flatten_history_dim=True``（默认）时，历史维度被折叠进特征维度，
产生适合 MLP 的 ``[num_envs, N * D]`` 张量；
``flatten_history_dim=False`` 时，输出保留时间维 ``[num_envs, N, D]``，
适合 RNN。

历史缓冲区在环境重置时清空。重置后的第一个观测会回填所有历史槽位，
因此策略从第零步就拿到有效数据。

``flatten_history_dim=True`` 且 ``concatenate_terms=True`` 时，mjlab
使用 **term 优先** （term-major）排序：每个项的完整历史先展平，再跨项
拼接。

.. code-block:: text

    Term A (D=4, history=3), Term B (D=2, history=3):
    [A_t0, A_t1, A_t2, B_t0, B_t1, B_t2]
     └─ A history ──┘  └─ B history ─┘

一些框架使用 **时间优先** （time-major）排序：先在每个时间步拼出完整帧，
再跨时间拼接。在不同排序的框架之间迁移策略时，需要对观测向量重新
排索引。

延迟
^^^^

设置 ``delay_max_lag > 0`` 启用一个环形缓冲区，存储过去的输出并返回
较早一步的值。滞后量以整数步从 ``[delay_min_lag, delay_max_lag]``
均匀采样。滞后为零返回当前观测；滞后为二返回两步前的观测。

.. code-block:: text

    50Hz control (20ms/step), lag=2:

    Sensor captures:  A     B     C     D     E     F     G     H
    Control steps:    0     1     2     3     4     5     6     7

    Policy sees:      A     A     A     B     C     D     E     F
                      └clamp┘     └ 40ms delay from here on

    Steps 0-1: lag clamped because the buffer is not yet full.
    Step 2 onward: each step returns the observation from 2 steps ago.

把现实世界延迟换算为滞后步数：``lag = latency_seconds / step_dt``。
50 Hz 控制（每步 20 ms）下，40 ms 的传感器延迟对应滞后 2。延迟按整数步
量化；要近似落在两步之间的延迟，把 ``delay_min_lag`` 和
``delay_max_lag`` 设为最接近的两个整数。

默认情况下每个环境独立采样自己的滞后（``delay_per_env=True``）。
其他参数控制重采样频率（``delay_update_period``）、保持概率
（``delay_hold_prob``）与相位错开（``delay_per_env_phase``）。

历史与延迟缓冲区只在启用时分配；使用默认设置的项没有任何开销。


内置观测函数
------------

下列函数位于 ``mjlab.envs.mdp.observations``（也被重导出为
``mjlab.envs.mdp``）。全部返回 ``[num_envs, D]`` 张量。

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - 函数
     - 描述
   * - ``base_lin_vel``
     - 机器人基座在基座系中的线速度。
   * - ``base_ang_vel``
     - 机器人基座在基座系中的角速度。
   * - ``projected_gravity``
     - 投影到基座系的重力向量。无需显式姿态表示即可提供横滚与俯仰
       信息。
   * - ``joint_pos_rel``
     - 相对默认位姿的关节位置。传 ``biased=True`` 获取带编码器偏置的
       位置（配合 ``dr.encoder_bias`` 做 sim2real）。
   * - ``joint_vel_rel``
     - 相对默认速度的关节速度。
   * - ``last_action``
     - 最近一次的动作张量。可选传 ``action_name`` 选择单个动作项。
   * - ``generated_commands``
     - 指定指令项的当前指令张量。需要
       ``params={"command_name": "<name>"}``。
   * - ``builtin_sensor``
     - 指定 ``BuiltinSensor`` 的原始数据（MuJoCo ``sensordata`` 切片）。
       需要 ``params={"sensor_name": "<entity>/<sensor>"}``。
   * - ``height_scan``
     - 距 ``RayCastSensor`` 各射线命中点的高度。需要
       ``params={"sensor_name": "<name>"}``。

``builtin_sensor`` 和 ``height_scan`` 的 ``sensor_name`` 参数必须与场景
中注册的传感器匹配。传感器配置方法见 :ref:`sensors`。


.. _obs-asymmetric:

非对称 actor-critic
-------------------

多个观测组可以支持非对称 actor-critic 架构。actor 组只包含真实硬件上
可获得的观测；critic 组可以包含只在训练期间可访问的特权仿真状态。

速度运动任务就使用了这一模式：actor 组接收带噪声的 IMU 读数和关节
状态；critic 组额外加入无噪声的高度扫描数据和脚部接触信息。
``enable_corruption`` 标志让这种分离非常干净：actor 项带噪声配置，而
critic 组完全禁用噪声。

.. code-block:: python

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,   # Noise active during training.
        ),
        "critic": ObservationGroupCfg(
            terms={**actor_terms, **privileged_terms},
            concatenate_terms=True,
            enable_corruption=False,  # No noise on critic.
        ),
    }

训练框架接收两组观测。策略网络在推理时读取 ``obs["actor"]``；价值网络
只在训练期间读取 ``obs["critic"]``。


编写自定义观测函数
------------------

观测函数第一个参数接收 ``env``，返回 ``[num_envs, D]`` 张量。额外参数
声明为函数参数，通过 ``ObservationTermCfg(params={...})`` 提供。

.. code-block:: python

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg


    def my_observation(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        robot = env.scene[asset_cfg.name]
        return robot.data.root_lin_vel_b

当观测项需要缓存初始化工作或维护回合内状态时，把它实现为带有
``__init__(self, cfg, env)`` 和 ``__call__(self, env, ...)`` 的类。若该
类有 ``reset(env_ids)`` 方法，管理器会在回合重置时自动调用。通用模式见
:ref:`env-config-term-pattern`。
