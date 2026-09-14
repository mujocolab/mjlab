.. _architecture_overview:

架构概览
========

mjlab 分为两层：**仿真层**负责对机器人和世界建模，**管理器层**在其上定义
强化学习问题。理解这一分层是建立系统全局认知的最快途径。

.. figure:: _static/architecture_diagram.png
   :width: 60%
   :align: center
   :alt: mjlab 架构图

   多个实体组合为 MjSpec，编译后传输给 MuJoCo Warp 进行 GPU 仿真。
   ManagerBasedRlEnv 负责编排 MDP；RSL-RL 负责训练。


仿真层
------

**场景流水线。**
mjlab 通过把实体描述组合成单一的
`MjSpec <https://mujoco.readthedocs.io/en/stable/programming/modeledit.html>`_
来构建场景。每个实体都从一个 MJCF 文件起步，通过 ``MjSpec.from_file()``
加载。完全用 XML 定义一切的用户可以直接使用这种方式；需要更强控制力的
用户则可以用 Python dataclass 扩展或覆盖已加载 spec 的属性：执行器、
碰撞规则、材质、传感器和初始状态。这种混合方式让用户能够从现有 MuJoCo
模型出发，叠加任务相关配置，而无需修改原始 XML。组合后的规格说明在 CPU
上编译为 ``MjModel``，然后经
`MuJoCo Warp <https://mujoco.readthedocs.io/en/stable/mjwarp/index.html>`_
传输到 GPU；MuJoCo Warp 构建在
`NVIDIA Warp <https://nvidia.github.io/warp/>`_ 之上。

**MuJoCo Warp。**
MuJoCo Warp 是 MuJoCo 的 GPU 加速后端。它保留了 MuJoCo 的
``MjModel``/``MjData`` 范式，但增加了一个关键的 *world*（世界）维度：
单个 ``MjData`` 对象可以并行持有 N 个独立仿真实例的状态，从而支持数千个
环境同时步进。模型参数默认在所有 world 之间共享；当域随机化需要时，
单个字段可以展开为按 world 独立取值。mjlab 把仿真步进捕获为
`CUDA graph <https://developer.nvidia.com/blog/cuda-graphs>`_：内核执行
序列只录制一次，后续调用直接重放，消除了 CPU 侧的分发开销。

.. note::

   CUDA graph 捕获是环境启动时的一次性开销。回合级重置和域随机化事件在
   图重放之间以普通 Python 代码运行，不会破坏捕获。

**核心组件。**
仿真层提供四个核心组件，各自拥有独立的文档页面：

- :ref:`entity`：机器人、被操作的对象，或
  :ref:`terrain <terrain>` 之类的静态物体，由 MJCF 描述加上可选的 Python
  配置（执行器、碰撞规则、初始状态）定义。
- :ref:`actuators`：实体的受控方式。既可包装 MJCF 中已定义的执行器，
  也可通过 Python 配置创建新执行器。
- :ref:`sensors`：世界的观测方式。包括 MuJoCo 原生传感器，以及 RGB-D
  相机、光线投射器等自定义传感器。
- :ref:`scene`：场景组合与环境摆放。


管理器层
--------

在仿真层之上，mjlab 采用了 Isaac Lab 提出的基于管理器的环境设计。用户
通过组合小型、自包含的 *项*（term，如奖励函数、观测计算、域随机化事件）
来定义环境，并将它们注册到相应的管理器。每个管理器负责其各项的生命周期：
在仿真循环的正确时机调用它们、聚合输出并暴露诊断信息。

项可以是为无状态计算准备的普通函数，也可以是继承 ``ManagerTermBase``
的类，后者适用于需要缓存昂贵初始化结果（例如把正则模式解析为关节索引）
或需要通过 ``reset()`` 钩子维护回合内状态的场景。

环境通过 ``ManagerBasedRlEnvCfg`` 配置，它是一个普通 dataclass，持有
每个管理器的项配置字典。

.. code-block:: python

    from mjlab.envs import ManagerBasedRlEnvCfg

    cfg = ManagerBasedRlEnvCfg(
        decimation=4,           # 4 physics steps per policy step
        episode_length_s=20.0,
        scene=...,              # SceneCfg: terrain, entities, sensors
        sim=...,                # SimulationCfg: timestep, solver, integrator
        observations={...},     # ObservationManager terms
        actions={...},          # ActionManager terms
        rewards={...},          # RewardManager terms
        terminations={...},     # TerminationManager terms
        events={...},           # EventManager terms (resets, DR)
        commands={...},         # CommandManager terms (velocity targets, etc.)
        curriculum={...},       # CurriculumManager terms
        metrics={...},          # MetricsManager terms
    )

.. rubric:: 八个管理器

- **ObservationManager**：组装观测组，支持可配置的后处理（裁剪、噪声、
  延迟、历史）。支持非对称 actor-critic。见 :ref:`observations`。
- **ActionManager**：把策略输出的张量路由到实体执行器，处理缩放与偏移。
  见 :ref:`actions`。
- **RewardManager**：计算奖励项的加权和，并按步长时间缩放以保证频率
  无关性。见 :ref:`rewards`。
- **TerminationManager**：评估终止条件，区分终止重置与超时。见
  :ref:`terminations`。
- **EventManager**：在生命周期节点（启动、重置、按间隔）触发各项。
  域随机化即通过事件项实现。见 :ref:`events` 和 :ref:`domain_randomization`。
- **CommandManager**：生成并重采样目标信号（速度目标、位姿目标）。见
  :ref:`commands`。
- **CurriculumManager**：依据策略表现调整训练条件。见 :ref:`curriculum`。
- **MetricsManager**：以回合平均值的形式记录自定义的每步数值。见
  :ref:`metrics`。

涵盖全部管理器的完整配置参考见 :ref:`environment_config`。


环境生命周期
------------

每个环境实例都会经历四个阶段。

1. **构建。** ``Scene`` 通过 ``MjSpec`` 组合实体的 MJCF 文件，并在 CPU
   上编译 ``MjModel``。``Simulation`` 经 MuJoCo Warp 把模型上传到 GPU，
   分配单个携带 N 个并行 world 的 ``MjData``，并捕获 ``step``、
   ``forward``、``reset`` 和 ``sense`` 的 CUDA 图。

2. **初始化。** 各管理器依据项配置字典构建。正则模式被匹配到关节、body
   和 geom 索引。观测历史与延迟缓冲区完成分配。域随机化项所需的模型字段
   从共享存储展开为按 world 存储，CUDA 图随新布局重建。启动事件触发一次。

3. **重置。** 在训练开始时以及环境终止或超时后调用。``EventManager``
   触发 ``reset`` 项，把场景恢复到初始状态（可带随机化）。指令目标被
   重新采样。观测历史缓冲区被清空。

4. **步进。** 策略动作由 ``ActionManager`` 处理。物理仿真推进
   ``decimation`` 次，每个子步都应用执行器指令并更新实体状态。decimation
   循环结束后，``TerminationManager`` 检查终止条件，``RewardManager``
   计算奖励信号。按计划触发的 step 与 interval 事件在重置前的状态上执行。
   随后所有已终止的环境被重置。一次 ``forward()`` 调用为全部环境刷新
   派生量。``CommandManager`` 推进或重采样目标。传感器更新。
   ``ObservationManager`` 组装供下一次策略查询使用的观测。

步进序列的完整顺序：

.. code-block:: text

    action_manager.process_action(action)
    for _ in range(decimation):
        action_manager.apply_action()
        sim.step()
        scene.update()
    termination_manager.compute()
    reward_manager.compute()
    metrics_manager.compute()
    event_manager.apply(mode="step")
    event_manager.apply(mode="interval")
    [reset terminated envs]
    sim.forward()
    command_manager.compute()  # dt=0 for envs just reset
    sim.sense()
    observation_manager.compute()

建立这一思维模型后，"核心概念"页面将逐一深入介绍仿真层的每个组件，
"管理器层"页面则逐个讲解每个管理器的配置与内置项。如果你来自 Isaac Lab，
:ref:`migration_isaac_lab` 描述了两者关键的 API 差异。
