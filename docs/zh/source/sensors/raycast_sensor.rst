.. _raycast_sensor:

光线投射传感器
==============

``RayCastSensor`` 提供 GPU 加速的光线投射，用于地形扫描、障碍检测和
深度感知。射线从附着于场景中 body、site 或 geom 的坐标系发出，传感器
报告命中距离、世界坐标命中位置和表面法线。

.. raw:: html

   <video controls style="display: block; margin: 0 auto; max-width: 100%; height: auto;">
     <source src="../../_static/raycast_demo.mp4" type="video/mp4">
   </video>


快速上手
--------

.. code-block:: python

    from mjlab.sensor import RayCastSensorCfg, GridPatternCfg, ObjRef

    # Downward-facing grid for terrain height scanning.
    raycast_cfg = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="base", entity="robot"),
        pattern=GridPatternCfg(size=(1.0, 1.0), resolution=0.1),
        max_distance=5.0,
    )

    scene_cfg = SceneCfg(
        entities={"robot": robot_cfg},
        sensors=(raycast_cfg,),
    )

    # Access at runtime.
    data = env.scene["terrain_scan"].data
    data.distances      # [B, N] distance to hit, -1 if miss
    data.hit_pos_w      # [B, N, 3] world-space hit positions
    data.normals_w      # [B, N, 3] surface normals


射线模式
--------

射线模式定义从传感器坐标系发出的射线的空间分布与方向。

.. grid:: 2

   .. grid-item-card:: 网格模式（Grid pattern）

      以固定空间分辨率发出的二维平行射线。因为射线间距以世界单位
      （米）定义，地面覆盖范围不随传感器高度变化。高度图与地形扫描的
      自然选择。

      .. raw:: html

         <video autoplay loop muted playsinline style="width: 100%; height: auto;">
           <source src="../../_static/pattern_grid.mp4" type="video/mp4">
         </video>

   .. grid-item-card:: 针孔相机模式（Pinhole camera pattern）

      从单一原点发出的发散射线，类似深度相机。因为视场以角度单位固定，
      地面覆盖范围随传感器高度增加。

      .. raw:: html

         <video autoplay loop muted playsinline style="width: 100%; height: auto;">
           <source src="../../_static/pattern_pinhole.mp4" type="video/mp4">
         </video>

.. code-block:: python

    from mjlab.sensor import GridPatternCfg, PinholeCameraPatternCfg

    # Parallel grid: fixed footprint, height-invariant.
    grid = GridPatternCfg(
        size=(1.0, 1.0),              # Grid dimensions in meters
        resolution=0.1,               # Spacing between rays
        direction=(0.0, 0.0, -1.0),   # Ray direction (down)
    )

    # Pinhole: perspective projection, diverging rays.
    pinhole = PinholeCameraPatternCfg(
        width=16,
        height=12,
        fovy=45.0,  # Vertical FOV in degrees
    )

    # Pinhole from a MuJoCo camera definition.
    pinhole = PinholeCameraPatternCfg.from_mujoco_camera("robot/depth_cam")

    # Pinhole from an intrinsic matrix.
    pinhole = PinholeCameraPatternCfg.from_intrinsic_matrix(
        intrinsic_matrix=[500, 0, 320, 0, 500, 240, 0, 0, 1],
        width=640,
        height=480,
    )


模式对比
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 维度
     - 网格
     - 针孔
   * - 射线方向
     - 平行
     - 发散
   * - 间距单位
     - 米
     - 度（FOV）
   * - 高度影响覆盖范围
     - 否
     - 是
   * - 投影模型
     - 正交
     - 透视


坐标系附着
----------

射线从经 ``ObjRef`` 指定的场景坐标系发出。该坐标系可以是任何实体上的
body、site 或 geom。

.. code-block:: python

    frame = ObjRef(type="body", name="base", entity="robot")
    frame = ObjRef(type="site", name="scan_site", entity="robot")
    frame = ObjRef(type="geom", name="sensor_mount", entity="robot")

``exclude_parent_body``（默认 ``True``）防止射线击中传感器所附着的
body。


射线对齐
--------

``ray_alignment`` 设置控制 body 旋转时射线相对附着坐标系的取向方式。

.. raw:: html

   <video autoplay loop muted playsinline
          style="display: block; margin: 0 auto; max-width: 100%; height: auto;">
     <source src="../../_static/ray_alignment_comparison.mp4" type="video/mp4">
   </video>

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - 模式
     - 描述
     - 用例
   * - ``"base"``
     - 完整跟踪位置与旋转
     - 随 body 安装的传感器
   * - ``"yaw"``
     - 跟随偏航（yaw），忽略俯仰与横滚
     - 地形高度图
   * - ``"world"``
     - 固定的世界系方向
     - 重力对齐的感知

.. code-block:: python

    RayCastSensorCfg(
        name="height_scan",
        frame=ObjRef(type="body", name="base", entity="robot"),
        pattern=GridPatternCfg(size=(1.0, 1.0), resolution=0.1),
        ray_alignment="yaw",
    )


Geom group 过滤
---------------

MuJoCo 把 geom 分配到 0 至 5 组。用 ``include_geom_groups`` 限制射线
可以击中哪些 geom。这对忽略纯可视化 geom 或隔离地形几何很有用。

.. code-block:: python

    RayCastSensorCfg(
        name="terrain_only",
        frame=ObjRef(type="body", name="base", entity="robot"),
        pattern=GridPatternCfg(),
        include_geom_groups=(0, 1),
    )


输出
----

``RayCastData`` 是一个 dataclass，形状以 ``B``（环境数）和 ``N``
（射线数）标注。

.. code-block:: python

    @dataclass
    class RayCastData:
        distances: Tensor   # [B, N] distance to hit, -1 if miss
        hit_pos_w: Tensor   # [B, N, 3] world-space hit positions
        normals_w: Tensor   # [B, N, 3] surface normals
        pos_w: Tensor       # [B, 3] sensor frame position
        quat_w: Tensor      # [B, 4] sensor frame orientation (w, x, y, z)

.. note::

   在配置上设置 ``debug_vis=True`` 可在运行时可视化射线命中。


示例
----

.. code-block:: python

    from mjlab.sensor import (
        RayCastSensorCfg, GridPatternCfg, PinholeCameraPatternCfg, ObjRef,
    )

    # Dense height map for terrain-aware locomotion.
    height_scan = RayCastSensorCfg(
        name="height_scan",
        frame=ObjRef(type="body", name="base", entity="robot"),
        pattern=GridPatternCfg(
            size=(1.6, 1.0),
            resolution=0.1,
            direction=(0.0, 0.0, -1.0),
        ),
        ray_alignment="yaw",
        max_distance=2.0,
    )

    # Simulated depth camera using pinhole projection.
    depth_cam = RayCastSensorCfg(
        name="depth",
        frame=ObjRef(type="site", name="camera_site", entity="robot"),
        pattern=PinholeCameraPatternCfg.from_mujoco_camera("robot/depth_cam"),
        max_distance=10.0,
    )

    # Forward-facing obstacle scan.
    obstacle_scan = RayCastSensorCfg(
        name="obstacle",
        frame=ObjRef(type="body", name="head", entity="robot"),
        pattern=GridPatternCfg(
            size=(0.5, 0.3),
            resolution=0.1,
            direction=(-1.0, 0.0, 0.0),
        ),
        max_distance=3.0,
        include_geom_groups=(0,),
    )


TerrainHeightSensor
-------------------

``TerrainHeightSensor`` 是 ``RayCastSensor`` 的轻量子类，在传感器数据
上为每个坐标系增加垂直净空。它对每条射线计算 ``frame_z - hit_z``，
把未命中替换为 ``max_distance``，并按坐标系跨射线归约。

.. code-block:: python

    from mjlab.sensor import TerrainHeightSensorCfg, RingPatternCfg, ObjRef

    cfg = TerrainHeightSensorCfg(
        name="foot_height",
        frame=(
            ObjRef(type="site", name="left_foot", entity="robot"),
            ObjRef(type="site", name="right_foot", entity="robot"),
        ),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=4),
        max_distance=1.0,
        include_geom_groups=(0,),
    )

    # At runtime:
    sensor = env.scene["foot_height"]
    sensor.data.heights    # [B, F] vertical clearance per foot
    sensor.data.distances  # [B, N] raw ray distances (inherited)

``reduction`` 配置字段控制每个坐标系内射线的聚合方式：``"min"``
（默认）、``"max"`` 或 ``"mean"``。
