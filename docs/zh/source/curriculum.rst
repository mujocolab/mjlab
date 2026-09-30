.. _curriculum:

课程
====

课程管理器依据策略表现调整训练条件。训练从较简单的问题开始，随着策略
证明自己能应对当前条件，难度逐步提升。常见用途包括把机器人推进到更难的
地形、拓宽指令速度范围，以及在整个训练过程中渐进提高奖励惩罚权重。

课程项在每次环境重置时被调用。每项接收环境和被重置环境的 ID 集合，
考察某个表现信号，然后直接修改环境参数。

.. code-block:: python

    from mjlab.managers.curriculum_manager import CurriculumTermCfg

    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "twist"},
        ),
    }

课程函数的返回值会以 ``Curriculum/<term_name>`` 记录在训练指标中。


内置课程函数
------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 函数
     - 描述
   * - ``terrain_levels_vel``
     - 度量每个机器人在回合内移动的距离。走过足够距离的机器人在地形
       网格中上升一行，未达标的下降。见下文地形课程一节。
   * - ``commands_vel``
     - 依据训练步数拓宽速度指令范围。每个阶段指定一个步数阈值和超过
       该阈值后启用的新范围。
   * - ``reward_curriculum``
     - 按训练步数阈值调整奖励项的权重和/或参数。取代较旧的
       ``reward_weight`` 函数，还支持修改奖励函数参数。
   * - ``termination_curriculum``
     - 按训练步数阈值调整终止项的参数。适合随训练推进逐步收紧终止条件
       （如能量限制）。


奖励课程
--------

``reward_curriculum`` 按训练进度调度奖励项权重或关键字参数的变更。每个
阶段指定一个 ``step`` 阈值和可选的 ``weight`` 或 ``params`` 更新。阶段
按顺序评估，所有阈值已达的阶段都会被应用。

**渐进提高惩罚权重**

常见模式是训练早期以低权重引入惩罚项，等策略学会基础动作后再加大：

.. code-block:: python

    from mjlab.managers.curriculum_manager import CurriculumTermCfg

    curriculum = {
        "joint_vel_hinge_weight": CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={
                "reward_name": "joint_vel_hinge",
                "stages": [
                    {"step": 0, "weight": -0.01},
                    {"step": 12000, "weight": -0.1},
                    {"step": 24000, "weight": -1.0},
                ],
            },
        ),
    }

**调整奖励参数**

也可以修改传给奖励函数的参数。例如随训练推进收紧跟踪容差：

.. code-block:: python

    curriculum = {
        "track_lin_vel_tighten": CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={
                "reward_name": "track_linear_velocity",
                "stages": [
                    {"step": 0, "params": {"std": 0.5}},
                    {"step": 20000, "params": {"std": 0.3}},
                    {"step": 50000, "params": {"std": 0.1}},
                ],
            },
        ),
    }

**同时调整权重与参数**

单个阶段可以同时更新权重和参数：

.. code-block:: python

    {"step": 24000, "weight": -1.0, "params": {"max_vel": 1.0}}


终止课程
--------

``termination_curriculum`` 按训练进度调度终止项参数的变更。这在策略
学会基本行为后逐步收紧终止条件时非常有用。

**收紧能量限制**

从宽松的能量阈值开始，随训练逐步减小：

.. code-block:: python

    from mjlab.managers.curriculum_manager import CurriculumTermCfg

    curriculum = {
        "energy_threshold": CurriculumTermCfg(
            func=mdp.termination_curriculum,
            params={
                "termination_name": "energy",
                "stages": [
                    {"step": 12000, "params": {"threshold": 1000.0}},
                    {"step": 24000, "params": {"threshold": 700.0}},
                    {"step": 36000, "params": {"threshold": 400.0}},
                ],
            },
        ),
    }

如有需要，``TerminationTermCfg`` 的 ``time_out`` 字段同样可以通过阶段
切换，不过实践中并不常见。


地形课程
--------

配合程序化地形使用的地形网格是 ``num_rows x num_cols`` 的地形块矩阵。
列代表地形类型变体；行代表难度等级，第 0 行最简单，第 ``num_rows - 1``
行最难。``TerrainGeneratorCfg.curriculum=True`` 时每列恰好分配一种
地形类型，使难度沿行单调递增。

环境构建时，每个环境在 ``[0, max_init_terrain_level]`` 内随机分配一个
起始行。``terrain_levels_vel`` 课程项在每次重置时依据回合内移动距离
提升或降级环境。到达最高等级的环境会被随机重新指派到任意一行，保持
所有难度等级上的覆盖。地形网格本身的配置细节见 :ref:`terrain`。


编写自定义课程函数
------------------

课程函数接收 ``env`` 和 ``env_ids``，施加参数变更，并返回一个待记录的
值（标量张量、张量字典或 ``None``）。典型实现读取某个表现指标，决定
提高还是降低难度，就地修改相关配置，并返回当前难度等级。通用模式见
:ref:`env-config-term-pattern`。
