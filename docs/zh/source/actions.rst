.. _actions:

动作
====

动作定义策略如何控制仿真。动作管理器每步接收策略的输出张量，把它切分
给各个已注册的动作项，并把每个切片路由到相应实体的执行器。每个动作项
把策略输出的一个连续片段映射到一组关节、肌腱或 site 上的某种控制模式
（位置、速度、力矩）。

.. code-block:: python

    from mjlab.envs.mdp.actions import JointPositionActionCfg

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),   # regex matching actuator names
            scale=0.5,
            use_default_offset=True,  # action 0 = default pose
        ),
    }


公共参数
--------

所有动作类型共享继承自 ``BaseActionCfg`` 的基础参数集。

``entity_name`` 标识要控制的场景实体。``actuator_names`` 是正则模式
元组，与执行器（或肌腱/site）名称匹配以选出受控目标。

``scale`` 在施加任何偏移之前乘在原始策略输出上。它接受标量，或把执行器
名称模式映射到逐目标取值的字典。这使策略输出保持在归一化范围内，同时
映射到有物理意义的单位。``offset`` 在缩放之后相加；关节类动作还提供
``use_default_offset``，自动加载实体的默认关节位置或速度作为偏移，使
原始输出为零即产生默认位姿。

``clip`` 可选地在处理后的动作（缩放与偏移之后）到达执行器之前进行截断。
它接受把执行器名称模式映射到 ``(min, max)`` 元组的字典，解析方式与
``scale`` 和 ``offset`` 相同。

.. code-block:: python

    JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=0.5,
        clip={".*_hip_.*": (-1.0, 1.0), ".*_knee_.*": (-0.5, 2.0)},
    )

动作在每个 decimation 子步（物理步）都会写入执行器目标，而不是每个
策略步只写一次。与之对比，观测延迟以策略步为单位。


动作类型
--------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 类型
     - 描述
   * - ``JointPositionAction``
     - 设置关节位置目标。``use_default_offset=True``（默认）时策略输出
       为零即命令默认位姿。``dr.encoder_bias`` 的编码器偏置会被自动
       减去，使随机化偏移正确传递到控制指令。
   * - ``RelativeJointPositionAction``
     - 相对当前关节位置设置关节位置目标。目标为
       ``current_pos + action * scale``，因此策略输出为零时无论机器人
       当前构型如何都会原地保持。
   * - ``JointVelocityAction``
     - 设置关节速度目标。``use_default_offset=True`` 使用默认关节速度
       （通常为零）。
   * - ``JointEffortAction``
     - 直接设置关节 effort（力矩）目标。无默认偏移。
   * - ``TendonLengthAction``
     - 设置肌腱长度目标。目标通过 ``actuator_names`` 与肌腱名匹配解析。
   * - ``TendonVelocityAction``
     - 设置肌腱速度目标。
   * - ``TendonEffortAction``
     - 设置肌腱 effort 目标。
   * - ``SiteEffortAction``
     - 在命名 site 上施加力和力矩。适用于四旋翼和无人机——推力施加在
       旋翼 site 上，而不是通过关节执行器。


任务空间动作
------------

``DifferentialIKAction`` 通过阻尼最小二乘逆 kinematics 把笛卡尔位置与
姿态指令转换为关节空间位置目标。每个 decimation 子步执行一次 IK 迭代，
因此末端执行器在整个子步区间内连续跟踪目标，而不只是在策略频率上。

动作维度根据配置自动选择：

- ``orientation_weight == 0``：**3D** （仅位置）
- ``orientation_weight > 0, use_relative_mode=True``：**6D** （位置增量
  + 轴角增量）
- ``orientation_weight > 0, use_relative_mode=False``：**7D** （绝对位置
  + 四元数）

所有目标（位置、姿态、关节限位、姿态保持）都堆叠进同一个 DLS 系统。
把某个权重设为零即禁用该目标，求解过程零开销。

``compute_dq()`` 方法返回关节位移而不写入执行器目标，支持在 RL 训练
之外的独立脚本中做多迭代 IK。


动作维度与历史
--------------

呈现给策略的总动作维度是各注册项 ``action_dim`` 之和。对关节、肌腱和
site 动作，它等于匹配到的目标数量；对 ``DifferentialIKAction``，依
激活的目标为 3、6 或 7。

动作管理器跟踪最近三个动作向量：``action``、``prev_action`` 和
``prev_prev_action``。``last_action`` 之类的观测项和 ``action_rate_l2``、
``action_acc_l2`` 之类的奖励项从这些缓冲区读取。动作历史在环境重置时
清零，回合边界不会泄漏信息。


多个动作项
----------

环境可以注册任意数量的动作项。动作管理器按注册顺序拼接它们的维度，在
相应边界切分策略输出张量，并独立路由每个切片。

.. code-block:: python

    from mjlab.envs.mdp.actions import (
        JointPositionActionCfg,
        JointVelocityActionCfg,
    )

    actions = {
        "arm_joints": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*_arm_.*",),
            scale=0.5,
        ),
        "wheel_joints": JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=(".*_wheel_.*",),
            scale=10.0,
        ),
    }

策略输出的张量宽度等于所有项匹配目标的总数。各动作项还可以指向不同的
实体，例如一项控制机器人、另一项控制被操作的对象。
