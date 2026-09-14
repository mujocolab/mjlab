.. _rgbd_camera:

RGB-D 相机
==========

``CameraSensor`` 使用 MuJoCo Warp 的光线追踪渲染管线在 GPU 上渲染 RGB
与深度图像。它既可以包装你在 XML 中定义的现有 MuJoCo 相机，也可以以
编程方式创建新相机。


快速上手
--------

.. code-block:: python

    from mjlab.sensor import CameraSensorCfg

    # Wrap an existing MuJoCo camera from the robot's XML.
    cam = CameraSensorCfg(
        name="wrist_cam",
        camera_name="robot/wrist_camera",
        data_types=("rgb", "depth"),
        width=160,
        height=120,
    )

    scene_cfg = SceneCfg(
        entities={"robot": robot_cfg},
        sensors=(cam,),
    )

    # Access at runtime.
    data = env.scene["wrist_cam"].data
    data.rgb      # [B, 120, 160, 3] uint8
    data.depth    # [B, 120, 160, 1] float32


创建 vs 包装相机
----------------

搭建相机传感器有两种方式。

**包装现有相机。** 如果你的 MJCF 模型已经定义了相机，通过
``camera_name`` 传入其名称。传感器使用模型中该相机的位置、姿态和视场。
你还可以选择覆盖 ``fovy`` 或切换为正交投影。

.. code-block:: python

    # Wrap the camera named "front_cam" in the robot's XML.
    CameraSensorCfg(
        name="front",
        camera_name="robot/front_cam",
        data_types=("rgb",),
    )

**创建新相机。** 当 ``camera_name`` 为 ``None``（默认值）时，传感器会在
场景构建阶段向 MjSpec 添加一个新相机。通过指定 ``pos``、``quat`` 和
可选的 ``fovy`` 来摆放它。

.. code-block:: python

    # Fixed overhead camera on the worldbody.
    CameraSensorCfg(
        name="overhead",
        pos=(0.0, 0.0, 2.0),
        quat=(0.0, 0.707, 0.707, 0.0),
        fovy=60.0,
        width=320,
        height=240,
        data_types=("rgb", "depth"),
    )


相机参数化
----------

MuJoCo 支持两种定义相机投影的方式，``CameraSensor`` 对两者都兼容。
完整细节见 `MuJoCo 相机文档
<https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera>`_。

**基于 FOV。** 更简单的方式。单个 ``fovy``（垂直视场角，单位度）加上
图像分辨率即可确定投影。这是通过 ``CameraSensorCfg`` 以编程方式创建
相机时的默认方式。

**基于内参。** 为了匹配真实相机硬件，MuJoCo 相机可以用 ``sensorsize``、
``focal``（或 ``focalpixel``）和 ``principal``（或 ``principalpixel``）
参数化。这些字段在 MJCF XML 中设置，可直接控制内参矩阵。存在内参参数时
MuJoCo 会忽略 ``fovy``。

包装现有相机时，传感器继承 XML 定义的参数化方式；创建新相机时使用
``fovy``。想为新相机使用内参参数，请在 XML 中定义它并用
``camera_name`` 包装。

.. note::

   如果你计划用域随机化随机化视场角：基于 FOV 的相机使用
   ``dr.cam_fovy``，基于内参的相机使用 ``dr.cam_intrinsic``。对使用
   内参参数的相机随机化 ``cam_fovy`` 没有任何效果。


挂在 body 上的相机
------------------

设置 ``parent_body`` 可以把新相机挂到特定 body 上而不是 worldbody。
此时 ``pos`` 和 ``quat`` 相对父 body 坐标系，相机随 body 移动。

.. code-block:: python

    # Camera mounted on the robot's end-effector.
    CameraSensorCfg(
        name="ee_cam",
        parent_body="robot/link_6",
        pos=(0.0, 0.0, 0.05),
        quat=(1.0, 0.0, 0.0, 0.0),
        fovy=45.0,
        width=160,
        height=120,
        data_types=("rgb", "depth"),
    )


数据类型
--------

``data_types`` 元组选择要渲染的图像模态。只有请求的类型会被分配，
``CameraSensorData`` 上的其他字段为 ``None``。

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - 类型
     - 形状
     - 描述
   * - ``"rgb"``
     - ``[B, H, W, 3]`` uint8
     - 彩色图像。MuJoCo Warp 以打包的 ABGR uint32 渲染，再在 GPU 上
       解包为 RGB 通道。
   * - ``"depth"``
     - ``[B, H, W, 1]`` float32
     - 深度图像。数值为到相机平面的距离。
   * - ``"segmentation"``
     - ``[B, H, W, 2]`` int32
     - 类型化分割。通道 0 存对象 ID，通道 1 存 MuJoCo 对象类型。背景
       像素为 ``(-1, -1)``。


渲染设置
--------

场景中所有相机传感器的 ``use_textures``、``use_shadows`` 和
``enabled_geom_groups`` 必须取值相同。这是底层 MuJoCo Warp 渲染系统的
约束——它为所有相机使用单个 ``RenderContext``。设置不一致会在场景构建
时抛出 ``ValueError``。

.. code-block:: python

    # These two cameras must agree on render settings.
    cam_a = CameraSensorCfg(
        name="cam_a",
        camera_name="robot/front_cam",
        use_textures=True,
        use_shadows=False,
        enabled_geom_groups=(0, 1, 2),
        data_types=("rgb",),
    )
    cam_b = CameraSensorCfg(
        name="cam_b",
        camera_name="robot/wrist_cam",
        use_textures=True,       # Must match cam_a
        use_shadows=False,       # Must match cam_a
        enabled_geom_groups=(0, 1, 2),  # Must match cam_a
        data_types=("depth",),
    )


输出
----

``CameraSensorData`` 是一个 dataclass，每种数据类型一个字段。

.. code-block:: python

    @dataclass
    class CameraSensorData:
        rgb: Tensor | None      # [B, H, W, 3] uint8
        depth: Tensor | None    # [B, H, W, 1] float32
        segmentation: Tensor | None  # [B, H, W, 2] int32

默认情况下，返回的张量是渲染缓冲区的零拷贝视图。如果你要就地修改它们，
请在配置上设置 ``clone_data=True``，避免破坏共享缓冲区。


Viser 中的可视化
----------------

Viser 查看器会自动发现场景中的所有 ``CameraSensor`` 实例，并在 GUI
侧边栏以实时图像面板显示其 RGB 与深度输出。3D 视口中会渲染相机视锥体，
标示相机的位置、姿态和视场。深度图像附带一个交互式滑杆，用于调整可视化
范围。

.. image:: ../_static/viser_camera_pane.png
   :align: center
   :alt: Viser viewer showing camera image panels and frustum visualization
