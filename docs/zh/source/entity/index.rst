.. _entity:

实体（Entity）
==============

``Entity`` 表示仿真中的一个物理对象：机器人、被操作的对象，或桌子这类
固定装置。它是 mjlab 物理层的核心抽象。

单一的 ``Entity`` 类覆盖所有变体（对比 Isaac Lab——它把这一概念拆分为
``Articulation``、``RigidObject`` 以及 ``AssetBase`` 的若干其他子类）。
两个正交的布尔属性对每个实例分类：

**基座类型。**
  *固定基座* 实体被焊接在世界坐标系上，没有自由关节。*浮动基座* 实体
  拥有一个自由关节，提供 6 自由度运动。

**铰接性。**
  *铰接* 实体拥有内部关节（转动关节、滑动关节等）。*非铰接* 实体除了
  可能存在的自由关节外没有其他关节。

.. list-table::
   :header-rows: 1
   :widths: 30 25 15 15 15

   * - 类型
     - 例子
     - ``is_fixed_base``
     - ``is_articulated``
     - ``is_actuated``
   * - 固定基座、非铰接
     - 桌子、墙
     - True
     - False
     - False
   * - 固定基座、铰接
     - 机械臂、门
     - True
     - True
     - True/False
   * - 浮动基座、非铰接
     - 盒子、球、杯子
     - False
     - False
     - False
   * - 浮动基座、铰接
     - 人形机器人、四足机器人
     - False
     - True
     - True/False

.. note::

   mjlab 会自动把每个固定基座实体包装进一个
   `mocap body <https://mujoco.readthedocs.io/en/stable/modeling.html#mocap-bodies>`_
   （运动捕获体），使每个并行环境可以把该实体放到不同位置。若没有这层
   包装，所有固定基座实体都会被焊死在世界原点。包装过程是透明的，但
   **只有在重置事件运行时才会执行定位**。你必须在自己的事件配置中包含
   ``reset_root_state_uniform`` 之类的重置事件；否则每个固定基座实体都会
   停在原点。完整示例见 :ref:`FAQ <faq>`。mocap 实体也可以在运行时通过
   ``entity.write_mocap_pose_to_sim()`` 重新定位。


配置实体
--------

每个实体都由一个 ``EntityCfg`` 描述。实践中只有 ``spec_fn`` 是必需的，
其他字段都有合理的默认值。一个被动浮动物体只需要：

.. code-block:: python

    from mjlab.entity import EntityCfg

    cube_cfg = EntityCfg(spec_fn=get_cube_spec)

带执行器的机器人会用到更多接口：

.. code-block:: python

    from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
    from mjlab.actuator import IdealPDActuatorCfg

    robot_cfg = EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={".*_hip_.*": 0.5, ".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                IdealPDActuatorCfg(
                    target_names_expr=(".*",),
                    stiffness={".*": 50.0},
                    damping={".*": 5.0},
                ),
            ),
        ),
        collisions=(my_collision_cfg,),
    )

以下小节逐一说明各字段。

``spec_fn``
^^^^^^^^^^^

一个返回 ``mujoco.MjSpec`` 的可调用对象。场景在组合阶段调用它，为返回
的 spec 附加名称前缀，并把所有内容编译进共享的 ``MjModel``。

简单场景下一个 lambda 就够用：

.. code-block:: python

    spec_fn = lambda: mujoco.MjSpec.from_file("robot.xml")

稍微复杂的情况请使用普通函数。MuJoCo 会自动从磁盘解析网格资源，因此
``get_spec`` 只需要加载 XML：

.. code-block:: python

    def get_spec() -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(str(ROBOT_XML))

由于 ``spec_fn`` 是任意的可调用对象，你可以在返回前执行任意
`MjSpec 编辑 <https://mujoco.readthedocs.io/en/stable/python.html#spec>`_：
添加 body、修改关节限位、更换材质，甚至完全不使用 XML、以纯编程方式
构建整个模型。

``init_state``
^^^^^^^^^^^^^^

默认的根位姿、根速度和关节位置/速度。这些值以 MuJoCo keyframe 的形式
存储，供重置事件在把实体恢复到初始配置时使用。

``joint_pos`` 和 ``joint_vel`` 是把正则模式映射到取值的字典。模式按
顺序与关节名匹配，因此对同时命中多个模式的关节，靠后的条目会覆盖
靠前的条目：

.. code-block:: python

    init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),       # root position
        rot=(1.0, 0.0, 0.0, 0.0),  # root quaternion (w, x, y, z)
        joint_pos={
            ".*": 0.0,              # all joints to zero
            ".*_hip_.*": 0.5,       # then override hips to 0.5
        },
    )

把 ``joint_pos`` 设为 ``None`` 可以直接使用 MJCF 模型中已有的 keyframe，
而无需在此处定义数值。

``articulation``
^^^^^^^^^^^^^^^^

执行器配置。只有带受控关节的实体才需要。被动对象（盒子、桌子、墙）
可以完全省略此字段。执行器类型的细节见 :ref:`actuators`。

``soft_joint_pos_limit_factor``（默认 1.0）用于收缩软限位惩罚奖励所
使用的关节范围，使策略在到达物理硬限位之前就受到惩罚。它不会修改
MuJoCo 模型中的实际关节限位。

Spec 编辑器
^^^^^^^^^^^

其余字段是可选的 spec 编辑器配置元组，会在编译前修改 ``MjSpec``：

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 字段
     - 用途
   * - ``collisions``
     - 替换实体的碰撞结构：哪些 geom 参与碰撞，以及使用什么接触参数。
   * - ``lights``
     - 向指定 body 添加光源。
   * - ``cameras``
     - 向指定 body 添加相机。
   * - ``textures``
     - 添加程序化纹理（棋盘格、渐变等）。
   * - ``materials``
     - 添加材质，并可通过正则将其指定给 geom。
   * - ``geoms``
     - 修补现有 geom 的属性（可视化分组、碰撞属性）。未设置的属性保持
       原样。

每个编辑器都接受正则模式来定位目标元素。例如，带有
``geom_names_expr=(".*_foot.*",)`` 的 ``CollisionCfg`` 只对脚部 geom
设置接触参数。完整示例见资产库（``mjlab.asset_zoo.robots``）。

``geoms`` 与 ``collisions`` 都会写 geom 属性，但语义不同。``GeomCfg``
是稀疏 *补丁*：每个属性默认 ``None``，只有你显式设置的属性才会被写入。
``CollisionCfg`` 则是一条 *策略*：``contype``、``conaffinity``、
``condim`` 和 ``priority`` 是必填项，且总会写入每个命中的 geom，未命中
的 geom 默认禁用碰撞，因此实体的接触行为完全由配置决定，与来源 XML
无关。碰撞配置在 geom 配置之后应用；如果 ``GeomCfg`` 设置的碰撞属性
随后被 ``CollisionCfg`` 覆盖，mjlab 会给出警告。

异构 world
^^^^^^^^^^

如果场景需要在不同的并行 world 中使用不同的网格资产（例如训练一个
需要泛化到不同物体形状的操作策略），请使用 ``VariantEntityCfg`` 而不是
``EntityCfg``。每个 world 会按可配置的权重被分配一个变体，依赖网格的
编译期常量（碰撞包围盒、body 惯量、子树质量）会以按 world 数组的形式
存储，从而保证域随机化和查看器的一致性。见 :ref:`heterogeneous_worlds`。

子类化 Entity
^^^^^^^^^^^^^

``Entity`` 和 ``EntityCfg`` 可以被子类化以实现特化行为。mjlab 自身
对地形就是这样做的：``TerrainEntity`` 扩展 ``Entity``，加入程序化地形
生成与逐环境原点计算；``TerrainEntityCfg`` 则增加了 ``terrain_type``、
``env_spacing``、``terrain_generator`` 等字段。任何需要超出
``EntityCfg`` 与 spec 编辑器能力的领域特定实体都可以沿用这一模式。

查找元素
^^^^^^^^

Entity 提供 ``find_*`` 方法，接受正则模式并返回匹配元素的索引和名称：

.. code-block:: python

    ids, names = entity.find_joints((".*_hip_.*", ".*_knee_.*"))
    ids, names = entity.find_geoms((".*foot.*",))
    ids, names = entity.find_bodies((".*",))

可用方法：``find_bodies()``、``find_joints()``、``find_geoms()``、
``find_sites()``、``find_tendons()``。它们在场景构建和管理器初始化期间
被内部使用。在奖励与观测项中，请优先使用带名称模式的 ``SceneEntityCfg``，
如下文所述。


读取运行时状态
--------------

实体被加入 ``SceneCfg`` 且环境构建完成后，可以通过三层抽象程度递减的
接口访问其状态。

EntityData
^^^^^^^^^^

``entity.data`` 是奖励、观测和终止函数的主要接口。它以形状为
``(num_envs, ...)`` 的 PyTorch 张量暴露运动学状态（位姿、速度、加速度）、
执行器力、广义力以及投影重力等派生的 body 系物理量。完整的属性参考见
:ref:`entity_data`。

``SceneEntityCfg`` 决定某个项作用于哪个实体以及实体内的哪些元素。
``joint_names``、``body_names``、``site_names`` 等正则模式会在管理器
初始化时一次性解析为整数索引，运行时没有任何正则开销：

.. code-block:: python

    from mjlab.managers.scene_entity_config import SceneEntityCfg

    def flat_orientation_l2(
        env,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Penalize non-flat base orientation using projected gravity."""
        asset = env.scene[asset_cfg.name]
        return torch.sum(
            torch.square(asset.data.projected_gravity_b[:, :2]), dim=1
        )

``SceneEntityCfg`` 还支持通过 ``joint_names``、``body_names``、
``site_names`` 等进行正则元素选择。解析后的整数索引（例如
``asset_cfg.joint_ids``）让运行时读取退化为单次张量切片，没有正则开销。

传感器
^^^^^^

传感器配置在 **场景** 上，而不是在单个实体上。传感器可以引用实体的
某个元素（例如机器人脚上的接触传感器、附着在 body site 上的加速度计），
也可以完全不依赖任何实体。这就是传感器放在 ``SceneCfg`` 而不是
``EntityCfg`` 中的原因。

运行时，传感器通过 ``env.scene`` 按名称访问，方式与实体相同：

.. code-block:: python

    def angular_momentum_penalty(env, sensor_name: str) -> torch.Tensor:
        sensor = env.scene[sensor_name]
        return torch.sum(torch.square(sensor.data), dim=-1)

内置传感器包装了 MuJoCo 的传感器类型（accelerometer、gyro、framepos、
subtreeangmom 等）。``ContactSensor``、``RayCastSensor`` 和
``CameraSensor`` 为接触检测、地形扫描和 RGB-D 渲染提供了更高层级的
抽象。详见 :ref:`sensors`。

原始仿真数据
^^^^^^^^^^^^

对于 ``EntityData`` 和传感器未覆盖的内容，可以通过 ``env.sim.data`` 和
``env.sim.model`` 访问底层 MuJoCo Warp 数组。它们以 PyTorch 张量形式
（零拷贝）暴露完整的 ``mjData`` 和 ``mjModel`` 字段，按全局 MuJoCo ID
而非按实体 ID 索引：

.. code-block:: python

    # Global joint positions across all entities.
    qpos = env.sim.data.qpos          # (num_envs, nq)

    # All body positions.
    xpos = env.sim.data.xpos          # (num_envs, nbody, 3)

    # Model-level constants.
    body_mass = env.sim.model.body_mass  # (nbody,)

这对底层操作或需要跨实体物理量的场景非常有用。

.. note::

   原始仿真数据的主要局限是你必须自行管理全局 MuJoCo 索引。我们计划在
   未来支持 MuJoCo 的
   `bind <https://mujoco.readthedocs.io/en/latest/python.html#relationship-to-pymjcf-and-bind>`_
   功能，届时可以直接把 spec 元素绑定到对应的数据视图，无需手工维护
   索引。

.. toctree::
   :maxdepth: 1

   entity_data
   per_world_mesh
