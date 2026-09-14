.. _commands:

指令
====

指令规定策略在每个时刻应当达成的目标：目标速度、参考轨迹、目标位置。
指令管理器生成这些信号、按可配置的间隔重新采样，并经观测系统传给
策略。


注册
----

指令在 ``ManagerBasedRlEnvCfg`` 中以字典形式注册，把字符串名称映射到
``CommandTermCfg`` 实例。与其他管理器使用的基于函数的项不同，每个指令
项都是一个继承自 ``CommandTerm`` 的类。

``resampling_time_range`` 字段控制指令多久变化一次。每次重采样后，该项
以秒为单位从给定的 ``(min, max)`` 区间均匀抽取一个新的计时值。每次回合
重置时指令也会无条件重采样。

.. code-block:: python

    commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-0.5, 0.5),
            ),
        ),
    }

``generated_commands`` 观测函数按名称读取当前指令张量并传给策略：

.. code-block:: python

    ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "twist"},
    )

环境没有指令时，管理器对所有操作做空操作并返回空张量，无需任何特殊
处理。


内置指令项
----------

每个任务都随附针对其目标定制的指令项。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 指令项
     - 描述
   * - ``UniformVelocityCommand``
     - 生成平面速度指令 ``[v_x, v_y, omega_z]``，从可配置范围均匀采样。
       支持站立模式（一部分环境收到零速度）和朝向模式（偏航角速率改由
       比例控制器跟踪采样的朝向角）。速度任务使用。
   * - ``LiftingCommand``
     - 为被操作物体生成 3D 目标位置。支持固定与动态难度模式。跟踪位置
       误差与回合成功率等指标。操作任务使用。
   * - ``MotionCommand``
     - 从预录的 ``.npz`` 运动片段流式提供参考关节位置、速度和 body
       位姿。支持三种起始帧采样模式：``"start"``（始终第 0 帧）、
       ``"uniform"``（随机）和 ``"adaptive"``（偏向困难区域）。重置时
       机器人从采样帧初始化，可带扰动。跟踪任务使用。

在配置中设置 ``debug_vis=True`` 后，各指令项可以在交互式查看器中渲染
调试可视化。下图是 ``MotionCommand`` 的幻影可视化——在参考位姿处渲染
一个半透明的机器人副本，与真实机器人并列。

.. figure:: _static/ghost_visualization.png
   :align: center
   :width: 100%

   Viser 中 G1 跟踪任务的参考运动可视化。


编写自定义指令项
----------------

自定义指令项是一个继承 ``CommandTerm`` 的类，搭配一个继承
``CommandTermCfg`` 的配置 dataclass。指令项必须实现四个方法：
``_resample_command(env_ids)`` 采样新目标、``_update_command(env_ids)``
做每步更新、``_update_metrics()`` 负责记录，以及一个返回当前目标张量的
``command`` 属性。基类自动管理重采样计时器与重置逻辑。

``_update_command`` 在两种情况下被调用。每个环境步它收到
``env_ids=None``，意味着更新所有环境。重置之后它会再次被调用，参数是
刚被重置的环境 ID，以便在计算观测之前把这些环境的指令状态更新到位。

当你的更新会推进状态时（例如递增参考运动的帧索引），这一区分就很重要。
这类推进只应作用于 ``env_ids``（为 ``None`` 时是全部环境）；否则重置
少数几个环境会把其他所有环境也推进。而只是从当前仿真状态重算数值的
更新（如朝向误差），无论调用多少次结果都一样，可以安全地忽略
``env_ids``。

配置必须实现 ``build(env)`` 方法来构造配对的指令项实例。
