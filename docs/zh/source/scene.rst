.. _scene:

场景
====

场景把实体、地形和传感器合并进同一次仿真。``SceneCfg`` 描述 world 的
内容，``Scene`` 类负责 MJCF 组合、编译与运行时状态管理。

.. code-block:: python

    from mjlab.scene import SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    # A robot on a flat ground plane with 4096 parallel environments.
    scene_cfg = SceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": robot_cfg},
    )

带程序化地形、传感器和多个实体的场景：

.. code-block:: python

    from mjlab.scene import SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.terrains.config import ROUGH_TERRAINS_CFG
    from mjlab.sensor import RayCastSensorCfg, ContactSensorCfg

    scene_cfg = SceneCfg(
        num_envs=4096,
        terrain=TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            max_init_terrain_level=5,
        ),
        entities={
            "robot": robot_cfg,
            "cube": cube_cfg,
        },
        sensors=(
            RayCastSensorCfg(name="terrain_scan", ...),
            ContactSensorCfg(name="feet_contact", ...),
        ),
    )


组合
----

场景从根 ``MjSpec`` 出发，把每个实体的 spec 以唯一名称前缀
`挂接 <https://mujoco.readthedocs.io/en/stable/python.html#attachment>`_
进来。名为 ``"robot"`` 的机器人实体，其所有内部 MuJoCo 元素（body、
关节、geom、执行器、传感器）都会加上 ``robot/`` 前缀：``base_link``
变成 ``robot/base_link``，``joint0`` 变成 ``robot/joint0``，以此类推。
加前缀可以避免多个实体共享元素名时的命名冲突，也为观测与奖励项提供了
一致的命名空间。

地形（如果存在）不加前缀挂接（其元素位于全局命名空间）。传感器在实体
之后添加，可以通过带前缀的名称引用实体元素。

``scene.compile()`` 把组合后的 ``MjSpec`` 转换为单个 ``MjModel``。
``Simulation`` 类随后经 MuJoCo Warp 把该模型上传到 GPU。仿真创建之后，
``scene.initialize()`` 把每个实体的元素索引解析到编译后的模型中、分配
状态缓冲区，并为所有相机或光线投射传感器搭建 GPU 渲染资源。

``scene.to_zip(path)`` 把编译后的模型导出为 ``.zip`` 文件，供独立的
MuJoCo 查看器离线检视。每个实体的初始状态 keyframe 会合并进导出文件，
因此模型以默认位姿打开。

运行时，实体和传感器都可以按名称访问：

.. code-block:: python

    robot = env.scene["robot"]              # Entity
    scan = env.scene["terrain_scan"]        # Sensor
    contact = env.scene["feet_contact"]     # Sensor

    robot.data.joint_pos                    # [B, num_joints]
    scan.data.distances                     # [B, N]
    contact.data.force                      # [B, N, 3]

实体 XML 中定义的内置传感器会在组合阶段被自动发现，并以实体名前缀
访问：

.. code-block:: python

    imu = env.scene["robot/trunk_imu"]      # Auto-discovered sensor


环境原点
--------

MuJoCo Warp 中的每个环境都是一个拥有独立状态的 world。环境之间不共享
物理空间，也无法互相作用。环境原点有两个用途：一是把实体在 world 中
铺开便于可视化（让查看器里的机器人并排显示，而不是堆在原点）；二是对
带程序化地形的运动任务，把每个环境放置到特定的子地形块上。

**平地地形。** 原点构成以世界原点为中心的规则网格，相邻间隔
``env_spacing`` 米。

**程序化地形。** 地形生成器产生 ``num_rows x num_cols`` 的子地形块
网格，每块有自己的中心点。每个环境被指派到一个块上，地形课程系统会
随表现提升把环境移动到更难的块。细节见 :ref:`terrain`。

.. note::

   目前所有环境共享同一个 ``MjModel``（相同的网格、几何与运动树）。
   允许不同 world 拥有不同网格或几何的异构仿真，
   `MuJoCo Warp 正在开发中 <https://github.com/google-deepmind/mujoco_warp/pull/1009>`_。
   上游落地后 mjlab 将提供支持。

重置事件项读取 ``scene.env_origins`` 来放置实体：

.. code-block:: python

    # Inside a reset event term.
    robot.write_root_pose_to_sim(
        default_root_pose + env_origins[env_ids]
    )

每个原点都标有一个不可见的球形 site（geom group 4），在查看器启用
group 4 时可见，便于开发阶段验证摆放位置。


自定义 spec 编辑
----------------

大多数场景由其实体、地形和传感器完整描述。偶尔会有跨越多个实体的修改。
例如连接天花板龙门吊与机器人的肌腱，无法在任何一个实体的 MJCF 内定义，
因为它同时引用两边的 site。

``SceneCfg`` 上的 ``spec_fn`` 回调负责这种情况。它在所有实体和传感器
都挂接完毕（带前缀名称）之后、编译之前，收到完整组合的 ``MjSpec``：

.. code-block:: python

    import mujoco

    def add_gantry(spec: mujoco.MjSpec):
        spec.worldbody.add_site(name="gantry", pos=(0, 0, 2))
        for side in ["left", "right"]:
            tendon = spec.add_tendon(
                name=f"{side}_rope",
                limited=True,
                range=(0, 1),
            )
            tendon.wrap_site("gantry")
            tendon.wrap_site(f"robot/{side}_hook")

    scene_cfg = SceneCfg(
        entities={"robot": robot_cfg},
        spec_fn=add_gantry,
    )

其他常见用途包括全局等式约束、自定义可视化几何，以及任何需要访问完整
组合场景的修改。
