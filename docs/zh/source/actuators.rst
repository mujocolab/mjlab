.. _actuators:

执行器
======

执行器把高层指令（位置、速度、力矩）转换为驱动关节的低层 effort。
它们通过 :ref:`EntityCfg <entity>` 的 ``articulation`` 字段配置。
mjlab 提供 **内置** 执行器——利用物理引擎的隐式积分获得最佳稳定性——
以及用于自定义控制律和执行器动力学的 **显式** 执行器。


快速上手
--------

以 ``BuiltinPositionActuator`` 实现基础 PD 控制，这是最常见的起点。

.. code-block:: python

    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

    robot_cfg = EntityCfg(
        spec_fn=lambda: load_robot_spec(),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_hip_.*", ".*_knee_.*"),
                    stiffness=80.0,
                    damping=10.0,
                    effort_limit=100.0,
                ),
            ),
        ),
    )

在任何执行器配置上直接添加延迟字段即可建模通信延迟。

.. code-block:: python

    from mjlab.actuator import BuiltinPositionActuatorCfg

    BuiltinPositionActuatorCfg(
        target_names_expr=(".*",),
        stiffness=80.0,
        damping=10.0,
        delay_min_lag=2,  # Minimum 2 physics steps
        delay_max_lag=5,  # Maximum 5 physics steps
    )


内置 vs 显式执行器
------------------

配置执行器时的关键设计决策是选择 **内置** 还是 **显式** 类型。两者的
差异归结为 MuJoCo 积分器如何处理依赖速度的力。

**内置执行器** （``BuiltinPositionActuator``、``BuiltinVelocityActuator``、
``BuiltinMotorActuator``、``BuiltinPdActuator``、
``BuiltinDcMotorActuator``、``BuiltinMuscleActuator``）在 MjSpec 中创建
MuJoCo 原生执行器元素。物理引擎负责计算控制律，并隐式积分依赖速度的
阻尼力。这带来最佳的数值稳定性，在高增益或大时间步长下尤其明显。

**显式执行器** （``IdealPdActuator``、``DcMotorActuator``、
``LearnedMlpActuator``）在用户代码中计算力矩，再经由作为直通件的
``<motor>`` 执行器转发。由于积分器无法把这些外部计算的力纳入速度导数
的考量，其数值鲁棒性不如内置类型。当需要内置类型无法表达的自定义控制律
或执行器动力学（如依赖速度的力矩限制、学习式执行器网络）时，使用显式
执行器。

在小时间步长、线性无约束区间内，两种方式的结果非常接近。在更大时间步长
或更高增益下，内置执行器的容错性更好。

**积分器选择。** mjlab 把阻尼放在执行器里而不是关节里。``euler`` 积分器
对关节阻尼做隐式处理、对执行器阻尼做显式处理，稳定性受限。
``implicitfast`` 积分器对所有已知的依赖速度的力做隐式处理，无需额外
开销即可同时处理执行器的比例项与阻尼项。

.. note::

     mjlab 默认使用 ``implicitfast``：它是 MuJoCo 官方推荐的积分器，
     对执行器侧阻尼提供更优的稳定性。


执行器类型
----------

所有执行器配置都共享继承自 ``ActuatorCfg`` 的几个公共字段：

- ``target_names_expr``：正则模式元组，与关节名匹配（使用其他
  ``transmission_type`` 时则与肌腱名/site 名匹配）。
- ``armature``：加到目标关节上的反射转子惯量。
- ``frictionloss``：以约束形式建模在目标关节上的静摩擦。见 MuJoCo 的
  `frictionloss <https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-frictionloss>`_。

内置执行器
^^^^^^^^^^

内置执行器经由 MjSpec API 使用 MuJoCo 的原生执行器类型。

**BuiltinPositionActuator**：创建 ``<position>`` 执行器，用于 PD 控制。

**BuiltinVelocityActuator**：创建 ``<velocity>`` 执行器，用于速度控制。

**BuiltinMotorActuator**：创建 ``<motor>`` 执行器，用于直接力矩控制。

**BuiltinPdActuator**：原生 PD，同时闭合位置目标和速度目标，实现为成对
的 ``<position>`` + ``<velocity>`` 执行器，求和为
``kp * (p_target - q) + kd * (v_target - qdot)``。
``BuiltinPositionActuator`` 把 kd 放在 ``<position>`` 元素上并隐式假定
速度参考为零；当策略输出非零速度目标时应使用本类型。原生传递让
``implicit`` / ``implicitfast`` 能在其速度更新中看到 kd 项，而
``IdealPdActuator`` 只是把 Python 计算的力矩经由不透明的 ``<motor>``
转发，做不到这一点。

**BuiltinDcMotorActuator**：包装 MuJoCo 原生的
`<dcmotor> <https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-dcmotor>`_
元素。力矩为 ``tau = K * (V - K * omega) / R``；反电动势走原生 bias
路径，因此 ``implicit`` / ``implicitfast`` 把它的速度导数视为有效阻尼。
三种输入模式决定 ``ctrl`` 携带什么：VOLTAGE 直接驱动电机；POSITION /
VELOCITY 对单一设定点闭合内部 PID（含抗积分饱和与斜率限制），其经
Vmax 截断的输出成为力矩。POSITION 模式固定 v_target = 0（kd 项作用于
原始速度）。可选物理特性：电感、含 I^2R 发热的热模型、齿槽转矩纹波、
LuGre 摩擦。``DcMotorActuator``（显式版本）是 ``<motor>`` 之上叠加
依赖速度力矩限幅的软件 PD；本类型才是真实的电气模型。

**BuiltinMuscleActuator**：创建 ``<muscle>`` 执行器，以力-长度-速度
特性模拟仿生肌肉动力学。

.. code-block:: python

    from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg

    # Mobile manipulator: PD for arm joints, velocity control for wheels.
    actuators = (
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"),
            stiffness=100.0,
            damping=10.0,
            effort_limit=150.0,
        ),
        BuiltinVelocityActuatorCfg(
            target_names_expr=(".*_wheel_.*",),
            damping=20.0,
            effort_limit=50.0,
        ),
    )


显式执行器
^^^^^^^^^^

显式执行器自行计算 effort 并转发给作为直通件的底层 ``<motor>`` 执行器。
稳定性影响见上文 `内置 vs 显式执行器`_。

**IdealPdActuator**：实现理想 PD 控制器，按
``tau = Kp * pos_error + Kd * vel_error`` 计算力矩。

**DcMotorActuator**：在 ``IdealPdActuator`` 基础上增加依赖速度的力矩
饱和，以建模直流电机的转矩-转速曲线（反电动势效应）。实现线性
转矩-转速曲线：零速时力矩最大，最高转速时力矩为零。

**LearnedMlpActuator**：基于神经网络的执行器，用训练好的 MLP 从关节
状态历史预测力矩输出。当解析模型无法刻画延迟、非线性和摩擦等复杂执行器
动力学时非常有用。继承直流电机基于速度的力矩限制。

.. code-block:: python

    from mjlab.actuator import IdealPdActuatorCfg, DcMotorActuatorCfg

    # Ideal PD for hips, DC motor model with torque-speed curve for knees.
    actuators = (
        IdealPdActuatorCfg(
            target_names_expr=(".*_hip_.*",),
            stiffness=80.0,
            damping=10.0,
            effort_limit=100.0,
        ),
        DcMotorActuatorCfg(
            target_names_expr=(".*_knee_.*",),
            stiffness=80.0,
            damping=10.0,
            effort_limit=25.0,       # Continuous torque limit
            saturation_effort=50.0,  # Peak torque at stall
            velocity_limit=30.0,     # No-load speed (rad/s)
        ),
    )


XML 执行器
^^^^^^^^^^

XML 执行器包装机器人 XML 文件中已经定义好的执行器。配置通过把
``target_names_expr`` 模式与执行器的 ``target`` 关节名匹配来定位现有的
执行器。每个关节必须恰好命中一个执行器。

**XmlActuator**：包装 XML 中已定义的任意执行器。执行器类型（position、
velocity、motor、muscle）从 XML 元素自动检测，也可以显式设置
``command_field``。

.. code-block:: python

    from mjlab.actuator import XmlActuatorCfg

    # Robot XML already has:
    # <actuator>
    #   <position name="hip_joint" joint="hip_joint" kp="100"/>
    # </actuator>

    # Wrap existing XML actuators.
    actuators = (
        XmlActuatorCfg(target_names_expr=("hip_joint",)),
    )

执行器延迟
^^^^^^^^^^

任何执行器配置都支持内联延迟字段来建模指令延迟。真实机器人上，板载 PD
环路以 KHz 频率运行并直接读取编码器，但来自策略的位置目标会因推理耗时
和通信总线周期而迟到。执行器延迟建模的正是这一点：指令目标被延迟，
而控制律看到的仍是新鲜的关节状态。

它与观测延迟不同——观测延迟建模的是传感器链路延迟（进入策略的过期
状态）。两者合起来覆盖了"传感器 → 策略 → 电机"往返的两条腿。

.. code-block:: python

    from mjlab.actuator import IdealPdActuatorCfg

    # Add 2-5 step delay to position commands.
    actuators = (
        IdealPdActuatorCfg(
            target_names_expr=(".*",),
            stiffness=80.0,
            damping=10.0,
            delay_min_lag=2,
            delay_max_lag=5,
            delay_hold_prob=0.3,         # 30% chance to keep current lag
            delay_update_period=10,      # Resample lag every 10 steps
        ),
    )

每一步都会从 ``[delay_min_lag, delay_max_lag]`` 中均匀采样一个滞后量。
延迟按物理时间步量化。例如物理频率 500Hz（每步 2ms）时，
``delay_min_lag=2`` 表示最小 4ms 的延迟。


编写执行器配置
--------------

由于执行器参数在每个配置内是统一的，需要不同参数的关节请使用单独的
执行器配置：

.. code-block:: python

    from mjlab.actuator import BuiltinPositionActuatorCfg

    # G1 humanoid with different gains per joint group.
    G1_ACTUATORS = (
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_.*", "waist_yaw_joint"),
            stiffness=180.0,
            damping=18.0,
            effort_limit=88.0,
            armature=0.0015,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("left_hip_pitch_joint", "right_hip_pitch_joint"),
            stiffness=200.0,
            damping=20.0,
            effort_limit=88.0,
            armature=0.0015,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_knee_joint",),
            stiffness=150.0,
            damping=15.0,
            effort_limit=139.0,
            armature=0.0025,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_ankle_.*",),
            stiffness=40.0,
            damping=5.0,
            effort_limit=25.0,
            armature=0.0008,
        ),
    )

这一设计选择体现了 mjlab 的一次刻意简化：每个 ``ActuatorCfg`` 代表
一种执行器类型（例如某个具体的电机/减速箱型号），并把它统一应用到其
驱动的所有关节。``armature``（反射转子惯量）和 ``gear`` 这类硬件参数
描述的是执行器硬件的属性，尽管它们在 MuJoCo 中实现为关节或执行器
字段。在其他框架（如 Isaac Lab）中，这些字段可能接受
``float | dict[str, float]`` 以支持逐关节差异。mjlab 则鼓励每个执行器
类型或每个关节组一个配置，让硬件模型保持物理一致且显式。主要的代价是
特殊情况（如平行连杆）下配置会显得冗长——本来逐关节覆盖会更方便——
但换来的是更清晰的语义和更简单的维护。

动作项如何把策略输出路由到执行器（包括用于任务空间控制的
DifferentialIK）见 :ref:`actions`；随机化增益与力矩限制见
:ref:`domain_randomization`。


计算硬件参数
------------

本节适用于从真实电机数据手册配置执行器的场景。如果你使用手工调参，
可以跳过。

mjlab 在 ``mjlab.utils.actuator`` 中提供了根据电机物理规格计算执行器
参数的工具。这对计算反射惯量（``armature``）以及从硬件数据手册推导
合适的控制增益尤其有用。

**示例：Unitree G1 电机配置**

.. code-block:: python

    from math import pi

    from mjlab.utils.actuator import (
        reflected_inertia_from_two_stage_planetary,
        ElectricActuator
    )

    # Motor specs from manufacturer datasheet.
    ROTOR_INERTIAS_7520_14 = (
        0.489e-4,  # Motor rotor inertia (kg*m**2)
        0.098e-4,  # Planet carrier inertia
        0.533e-4,  # Output stage inertia
    )
    GEARS_7520_14 = (
        1,            # First stage (motor to planet)
        4.5,          # Second stage (planet to carrier)
        1 + (48/22),  # Third stage (carrier to output)
    )

    # Compute reflected inertia at joint output.
    # J_reflected = J_motor*(N1*N2)**2 + J_carrier*N2**2 + J_output.
    ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
        ROTOR_INERTIAS_7520_14, GEARS_7520_14
    )

    # Create motor spec container.
    ACTUATOR_7520_14 = ElectricActuator(
        reflected_inertia=ARMATURE_7520_14,
        velocity_limit=32.0,   # rad/s at joint
        effort_limit=88.0,     # N*m continuous torque
    )

    # Derive PD gains from natural frequency and damping ratio.
    NATURAL_FREQ = 10 * 2*pi  # 10 Hz bandwidth.
    DAMPING_RATIO = 2.0       # Overdamped, see note below.
    STIFFNESS = ARMATURE_7520_14 * NATURAL_FREQ**2
    DAMPING = 2 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ

    # Use in actuator config.
    from mjlab.actuator import BuiltinPositionActuatorCfg

    actuator = BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_pitch_joint",),
        stiffness=STIFFNESS,
        damping=DAMPING,
        effort_limit=ACTUATOR_7520_14.effort_limit,
        armature=ACTUATOR_7520_14.reflected_inertia,
    )

.. note::

     示例使用 ``DAMPING_RATIO = 2.0`` （过阻尼）而不是临界阻尼值 1.0。
     原因在于反射惯量的计算只考虑了电机转子惯量，没有考虑被驱动连杆的
     表观惯量。实践中关节处的总有效惯量高于反射电机惯量，因此在真实
     系统惯量被低估时，过阻尼比能提供更好的稳定裕度。

**平行连杆近似：**

对由平行连杆驱动的关节（如 G1 的双电机脚踝），标称构型下的有效
armature 可以近似为各电机 armature 之和：

.. code-block:: python

    # Two 5020 motors driving ankle through parallel linkage.
    G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
        target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
        stiffness=STIFFNESS_5020 * 2,
        damping=DAMPING_5020 * 2,
        effort_limit=ACTUATOR_5020.effort_limit * 2,
        armature=ACTUATOR_5020.reflected_inertia * 2,
    )


扩展：自定义执行器
------------------

所有执行器都实现统一的 ``compute()`` 接口：接收一个 ``ActuatorCmd``
（包含位置、速度和 effort 目标），返回驱动各关节的低层 MuJoCo 执行器
控制信号。

**核心接口：**

.. code-block:: python

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        """Convert high-level commands to control signals.

        Args:
            cmd: Command containing position_target, velocity_target,
                effort_target (each is a [num_envs, num_targets] tensor
                or None)

        Returns:
            Control signals for this actuator
            ([num_envs, num_targets] tensor)
        """

**生命周期钩子：**

- ``edit_spec``：编译前修改 MjSpec（添加执行器、设置增益）
- ``initialize``：编译后初始化（解析索引、分配缓冲区）
- ``reset``：按环境重置的逻辑
- ``update``：步进前的更新
- ``compute``：把指令转换为控制信号

**属性：**

- ``target_ids``：本执行器控制的本地目标索引张量
- ``target_names``：本执行器控制的目标名称列表
- ``ctrl_ids``：本执行器的全局控制输入索引张量

``IdealPdActuator`` 是编写自定义显式执行器时推荐的基类。
``DcMotorActuator`` 和 ``LearnedMlpActuator`` 都构建在它之上，可以作为
扩展模式的示例。
