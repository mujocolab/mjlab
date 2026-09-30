.. _sensors:

传感器
======

如 :ref:`entity` 所述，传感器位于 mjlab 数据访问层级中 ``EntityData``
与原始仿真数组之间。最简单的情况下，它们把 MuJoCo 传感器原语包装成
映射真实机器人硬件的干净接口。除此之外，它们还是把仿真数据转换为结构化
输出的通用抽象：``ContactSensor`` 以归约和腾空时间跟踪聚合接触对，
``RayCastSensor`` 执行 GPU 加速的地形扫描，``CameraSensor`` 在 GPU 上
渲染 RGB 与深度图像，基础 ``Sensor`` 类可以被子类化以实现自定义测量
逻辑。

传感器配置在 **场景层级**，而不是在单个实体上。传感器可以引用实体的
某个元素（机器人脚上的接触传感器、附着在 body site 上的加速度计），
但也可以完全独立于任何实体。这就是传感器放在 ``SceneCfg`` 而不是
``EntityCfg`` 中的原因。

.. code-block:: python

    from mjlab.sensor import (
        BuiltinSensorCfg, ContactSensorCfg, ContactMatch, ObjRef,
    )

    # A robot with an IMU accelerometer and foot contact detection.
    scene_cfg = SceneCfg(
        entities={"robot": robot_cfg},
        sensors=(
            BuiltinSensorCfg(
                name="imu_acc",
                sensor_type="accelerometer",
                obj=ObjRef(type="site", name="imu_site", entity="robot"),
            ),
            ContactSensorCfg(
                name="feet_contact",
                primary=ContactMatch(
                    mode="geom", pattern=r".*_foot$", entity="robot",
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
            ),
        ),
    )

    # Access at runtime.
    imu = env.scene["robot/imu_acc"].data        # [B, 3] acceleration
    feet = env.scene["feet_contact"].data         # ContactData
    feet.found                                    # [B, P] contact count per foot
    feet.force                                    # [B, P, 3] contact force per foot

mjlab 提供四种传感器类型：用于 MuJoCo 原生测量的 ``BuiltinSensor``、
结构化接触检测的 ``ContactSensor``、GPU 加速光线投射的
``RayCastSensor``，以及 RGB-D 渲染的 ``CameraSensor``。基础 ``Sensor``
类可以被子类化以实现自定义测量逻辑；见下文 `扩展：自定义传感器`_。


BuiltinSensor
-------------

``BuiltinSensor`` 包装 MuJoCo 的原生传感器类型。每个传感器通过
``ObjRef`` 附着到一个 MuJoCo 元素（site、关节、body 等），返回形状为
``[num_envs, dim]`` 的 ``torch.Tensor``，其中 ``dim`` 取决于传感器类型
（向量 3、四元数 4、标量 1）。

+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------+
| 类别      | 可用传感器                                                                                                                                         |
+===========+====================================================================================================================================================+
| **Site**  | ``accelerometer``, ``velocimeter``, ``gyro``, ``force``, ``torque``, ``magnetometer``, ``rangefinder``                                             |
+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------+
| **Joint** | ``jointpos``, ``jointvel``, ``jointlimitpos``, ``jointlimitvel``, ``jointlimitfrc``, ``jointactuatorfrc``                                          |
+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------+
| **Frame** | ``framepos``, ``framequat``, ``framexaxis``, ``frameyaxis``, ``framezaxis``, ``framelinvel``, ``frameangvel``, ``framelinacc``, ``frameangacc``    |
+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------+
| **Other** | ``actuatorpos``, ``actuatorvel``, ``actuatorfrc``, ``subtreecom``, ``subtreelinvel``, ``subtreeangmom``, ``clock``, ``e_potential``, ``e_kinetic`` |
+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------+

``ObjRef`` 标识传感器附着到哪个 MuJoCo 元素。``entity`` 字段把查找
限定到特定实体的命名空间，传感器名称会相应地自动加前缀（例如实体
``"robot"`` 上的 ``"imu_acc"`` 变成 ``"robot/imu_acc"``）。

.. code-block:: python

    from mjlab.sensor import BuiltinSensorCfg, ObjRef

    # Accelerometer attached to a site.
    BuiltinSensorCfg(
        name="imu_acc",
        sensor_type="accelerometer",
        obj=ObjRef(type="site", name="imu_site", entity="robot"),
    )

    # Joint limit sensor with output clamping.
    BuiltinSensorCfg(
        name="knee_limit",
        sensor_type="jointlimitpos",
        obj=ObjRef(type="joint", name="knee_joint", entity="robot"),
        cutoff=0.1,
    )

    # Relative frame position (end-effector w.r.t. base).
    BuiltinSensorCfg(
        name="ee_pos",
        sensor_type="framepos",
        obj=ObjRef(type="body", name="end_effector", entity="robot"),
        ref=ObjRef(type="body", name="base", entity="robot"),
    )


自动发现
^^^^^^^^

实体 XML 中已经定义的传感器会在场景组合阶段被自动发现，并加上实体名
前缀。无需为它们创建 ``BuiltinSensorCfg``。

.. code-block:: xml

    <!-- In robot.xml -->
    <sensor>
        <accelerometer name="trunk_imu" site="imu_site"/>
        <jointpos name="hip_sensor" joint="hip_joint"/>
    </sensor>

.. code-block:: python

    # Access by prefixed name.
    imu = env.scene["robot/trunk_imu"]
    hip = env.scene["robot/hip_sensor"]


ContactSensor
-------------

每个物理步，MuJoCo 都会为整个场景产出一组扁平、非结构化的接触对列表。
单个脚部 geom 可能同时与地面产生多个接触，并与其他实体的接触交错在一起。
``ContactSensor`` 把这份原始列表过滤到你关心的接触对，把每个元素的
多个接触归约为固定数量，再把结果打包成策略可以直接消费的干净批处理
张量。它构建在 MuJoCo 原生的
`contact 传感器 <https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact>`_
之上。

primary 与 secondary
^^^^^^^^^^^^^^^^^^^^

接触是成对的：你通常想知道的是"机器人脚有没有碰到地形"，而不是
"有没有什么东西碰到什么东西"。``primary`` 定义你测量的元素（脚）。
``secondary`` 可选地限制它们接触的对象（地形）。``secondary`` 为
``None`` 时，与 primary 元素的任何接触都算数。

每一侧都用 ``ContactMatch`` 指定。``mode`` 选择 MuJoCo 元素类型
（``"geom"``、``"body"`` 或 ``"subtree"``），``pattern`` 接受一个正则
或正则元组，在实体范围内与元素名匹配。

.. code-block:: python

    from mjlab.sensor import ContactSensorCfg, ContactMatch

    # Foot geoms contacting the terrain body.
    ContactSensorCfg(
        name="feet_ground",
        primary=ContactMatch(
            mode="geom", pattern=r".*_foot$", entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
    )

    # Self-collision: pelvis subtree against itself.
    ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree", pattern="pelvis", entity="robot",
        ),
        secondary=ContactMatch(
            mode="subtree", pattern="pelvis", entity="robot",
        ),
        fields=("found",),
    )

输出形状
^^^^^^^^

``r".*_foot$"`` 这样的模式会解析为 ``P`` 个 primary 元素（例如四足
机器人的四只脚）。每个 primary 在输出张量的接触轴上占一列：

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - 字段组
     - 形状
     - 说明
   * - 逐接触
       (``found``、``force``、``torque``、``dist``、``pos``、
       ``normal``、``tangent``)
     - ``[B, P * num_slots, ...]``
     - primary 优先排序：索引
       ``[i * num_slots : (i + 1) * num_slots]`` 属于 primary ``i``。
   * - 逐 primary
       (``current_air_time``、``last_air_time``、
       ``current_contact_time``、``last_contact_time``)
     - ``[B, P]``
     - 腾空时间字段按 primary 累积、跨槽位归约（任意槽位有接触即视为
       该 primary 有接触）。

默认 ``num_slots=1`` 时两族形状重合（``N == P``），这就是为什么多数
代码可以把两者都当作 ``[B, P, ...]`` 处理。

模式展开后，用
:attr:`mjlab.sensor.contact_sensor.ContactSensor.primary_names`
恢复索引到名称的映射：

.. code-block:: python

    sensor = env.scene["feet_contact"]
    sensor.primary_names                # ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
    sensor.data.current_air_time[:, 0]  # air time for FR_foot

归约
^^^^

一个 primary 元素可能与 secondary 同时存在大量接触（例如平脚掌踩在
粗糙地形上会有多个接触点）。``reduce`` 模式把这些原始接触坍缩为
``num_slots`` 个代表性接触：

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 模式
     - 行为
   * - ``"none"``
     - 快速、非确定性地选取至多 ``num_slots`` 个接触。
   * - ``"mindist"``
     - 保留最深的 ``num_slots`` 个接触。
   * - ``"maxforce"``
     - 按力大小保留最强的 ``num_slots`` 个接触。
   * - ``"netforce"``
     - 把所有接触求和为合力旋量，作用点在力加权质心。每个 primary
       固定输出一个槽位，与 ``num_slots`` 无关。

什么时候该设 ``num_slots > 1``
""""""""""""""""""""""""""""""

几乎所有配置都保持 ``num_slots`` 默认值 ``1``，因为模式展开已经为每个
感兴趣的元素生成一列（每只脚一列、每根手指一列、每个连杆一列）。只有
当单个 primary 可能存在多个物理上相互独立、且你想分别考察的接触点时
才增大 ``num_slots``，例如：

- 用脚掌四角的接触点计算压力中心。
- 从多个指尖-物体接触点推断抓取质量。
- 通过观察接触面某个角落是否脱开接触来检测倾翻。

这些情况下把 ``num_slots`` 与 ``{"mindist", "maxforce", "none"}``
中的 ``reduce`` 搭配使用。``reduce="netforce"`` 时它没有作用。

.. note::

   ``num_slots`` 是传感器存储量的上限，并不保证 MuJoCo 真的会生成
   那么多接触。碰撞检测器会按 geom 类型组合限制每对 geom 生成的接触点
   数量：例如球-平面对最多 1 个接触，盒-盒对最多 4 个。因此球形
   primary 对平面 secondary 设 ``num_slots=8`` 会有七个槽位永远为零。
   各类型对的限制见
   `MuJoCo 碰撞文档 <https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection>`_。

字段
^^^^

``fields`` 元组选择要提取的接触量。只有请求的字段会被分配，其余字段
在输出 dataclass 上为 ``None``。可用字段为 ``"found"``、``"force"``、
``"torque"``、``"dist"``、``"pos"``、``"normal"`` 和 ``"tangent"``。

.. note::

   ``torque`` 和 ``force`` 的摩擦切向分量只有在接触对启用摩擦时才非零，
   这要求接触对中至少一个 geom 满足 ``condim >= 3``。``condim=1``
   （无摩擦）时接触只产生法向力。这是接触的物理属性，不是传感器限制。

腾空时间跟踪
^^^^^^^^^^^^

运动任务常常需要知道脚何时落地、何时离地，以构造步态奖励。设置
``track_air_time=True`` 即启用按 primary 的计时。传感器会在
``ContactData`` 上维护四个额外张量，形状均为 ``[B, P]``：
``current_air_time``、``last_air_time``、``current_contact_time`` 和
``last_contact_time``。两个辅助方法为状态转换事件提供边沿检测：

.. code-block:: python

    sensor = env.scene["feet_air"]
    first_contact = sensor.compute_first_contact(dt)  # [B, P], True for primaries that just landed
    first_air = sensor.compute_first_air(dt)           # [B, P], True for primaries that just took off

即使 ``num_slots > 1``，腾空时间也按 primary 计：传感器跨槽位归约
``found``，任意槽位有接触即视为该 primary 处于接触状态。

.. _contact-sensor-history:

历史（对 decimation 安全的接触检测）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 decimation（每个策略步多个物理子步）时，一次短暂的碰撞可能完全
发生在子步循环内部。等策略读到传感器时接触已经消失，``found`` 报告
为零。在传感器配置上设置 ``history_length``，传感器就会为 force、
torque 和 distance 字段保留最近 *N* 个子步的滚动缓冲区。策略随后可以
检视完整历史，判断是否真的发生过接触。

把 ``history_length`` 设成与 decimation 相等，缓冲区就恰好覆盖一个
策略步：

.. code-block:: python

    ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        history_length=4,  # matches decimation=4
    )

历史张量与常规字段一起放在 ``ContactData`` 上：

.. code-block:: python

    data = sensor.data
    data.force_history   # [B, N, H, 3]  (H = history_length)
    data.torque_history  # [B, N, H, 3]
    data.dist_history    # [B, N, H]

索引 0 是最近的子步。要检查是否有任何子步的接触力超过阈值：

.. code-block:: python

    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    had_contact = (force_mag > 10.0).any(dim=1).any(dim=-1)  # [B]

.. note::

   ``track_air_time=True`` 本来就会为步态奖励跨子步累积接触状态，因此
   脚-地传感器通常不需要 ``history_length``。历史适用于需要检测原本
   会被漏掉的短暂碰撞的传感器（自碰撞、非法接触终止）。


输出
^^^^

``ContactData`` 是一个 dataclass，其字段与配置中的 ``fields`` 元组
对应。未请求的字段为 ``None``。

.. code-block:: python

    @dataclass
    class ContactData:
        found: Tensor | None     # [B, N] contact count
        force: Tensor | None     # [B, N, 3]
        torque: Tensor | None    # [B, N, 3]
        dist: Tensor | None      # [B, N] penetration depth
        pos: Tensor | None       # [B, N, 3] contact position
        normal: Tensor | None    # [B, N, 3] surface normal
        tangent: Tensor | None   # [B, N, 3]

        # With track_air_time=True.
        current_air_time: Tensor | None
        last_air_time: Tensor | None
        current_contact_time: Tensor | None
        last_contact_time: Tensor | None


RayCastSensor
-------------

``RayCastSensor`` 提供 GPU 加速的光线投射，用于地形扫描与深度感知。
支持网格与针孔相机两种射线模式，对齐方式可配置。完整文档见
:ref:`raycast_sensor`。


RGB-D 相机
----------

``CameraSensor`` 从 MuJoCo 相机渲染 RGB 与深度图像。完整文档见
:ref:`rgbd_camera`。


扩展：自定义传感器
------------------

所有传感器都继承自 ``Sensor[T]``——一个泛型基类，其中 ``T`` 是
``data`` 属性返回的数据类型（``BuiltinSensor`` 为 ``torch.Tensor``，
``ContactSensor`` 为 ``ContactData``）。

基类提供自动的每步缓存。``data`` 属性在每步首次访问时调用
``_compute_data()`` 并缓存结果。缓存会在 ``update()`` 或 ``reset()``
被调用时自动失效，因此同一步内的多次读取（来自不同的观测或奖励项）
只支付一次计算成本。

**生命周期方法：**

- ``edit_spec``：场景构建期间向 MjSpec 添加传感器元素。
- ``initialize``：编译后的初始化。缓存传感器索引、分配缓冲区、解析
  引用。
- ``update``：每个物理步调用。使数据缓存失效。可覆写以维护每步状态
  （如腾空时间计数器）。
- ``reset``：环境重置时调用。使数据缓存失效。可覆写以清除按环境的
  状态。
- ``_compute_data``：计算并返回传感器输出。缓存过期时由 ``data``
  属性惰性调用。

``ContactSensor`` 和 ``RayCastSensor`` 是自定义传感器开发最完整的
参考实现。

.. toctree::
   :maxdepth: 1
   :hidden:

   raycast_sensor
   rgbd_camera
