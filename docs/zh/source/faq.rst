.. _faq:

常见问题与故障排查
==================

本页收集关于 **平台支持**、**性能**、**训练稳定性** 和 **可视化** 的
常见问题，并附上实用的调试技巧与延伸资源链接。

平台支持
--------

macOS 上能用吗？
~~~~~~~~~~~~~~~~

可以，但性能有限。mjlab 在 macOS 上通过 MuJoCo Warp 以 **纯 CPU**
方式运行。

- **不建议在 macOS 上训练**，因为没有 GPU 加速。
- **评估可以运行**，但比带 CUDA 的 Linux 明显更慢。

严肃的训练负载我们强烈建议 **Linux + NVIDIA GPU**。

Windows 上能用吗？
~~~~~~~~~~~~~~~~~~

我们在 **Windows** 和 **WSL** 上做过初步测试，但部分工作流不保证稳定。

- Windows 支持 **进度可能落后** 于 Linux。
- Windows 的 **测试频率更低**，因为 Linux 是主要的开发与部署平台。
- 非常欢迎社区贡献来改进 Windows 支持。

CUDA 兼容性
~~~~~~~~~~~

MuJoCo Warp 并不支持所有 CUDA 版本。

- CUDA 兼容性详情见
  `mujoco_warp#101 <https://github.com/google-deepmind/mujoco_warp/issues/101>`_。
- **推荐**：CUDA **12.4+**（支持 CUDA graph 的条件执行）。

如何在不碰 GPU 的情况下跑 CPU？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

传 ``device="cpu"`` 会把所有 mjlab 计算放到 CPU 上，但 **不能** 阻止
Warp 初始化 GPU。Warp 运行时第一次启动时会急切地枚举并为 **每一块**
可见设备创建 CUDA 上下文，不管你请求的是哪个设备。所以在带可见 GPU
的机器上，``device="cpu"`` 的运行仍会占用显存。

这发生在 Warp 内部，包一旦导入就无法从 Python 阻止。要让进程完全
不碰 GPU，请在启动前对 CUDA 隐藏设备：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="" uv run train.py ...

没有可见 CUDA 设备时，Warp 以纯 CPU 初始化，从不在 GPU 上分配。背景
见 `issue #949
<https://github.com/mujocolab/mjlab/issues/949>`_。

性能
----

比 Isaac Lab 快吗？
~~~~~~~~~~~~~~~~~~~

根据我们近几个月的经验，mjlab 与 Isaac Lab **相当或更快**。

推荐什么 GPU？
~~~~~~~~~~~~~~

- **RTX 40 系**（或更新）
- **L40s、H100**

支持多 GPU 训练吗？
~~~~~~~~~~~~~~~~~~~

支持。mjlab 用 `torchrunx <https://github.com/apoorvkh/torchrunx>`_
支持 **多 GPU 分布式训练**。

- 运行 ``train`` 命令时传 ``--gpu-ids "[0, 1]"``（或 ``--gpu-ids all``）。
- 配置细节与示例见 :doc:`training/distributed_training`。

训练与调试
----------

训练因 NaN 错误崩溃
~~~~~~~~~~~~~~~~~~~

使用 ``rsl_rl`` 时的典型报错：

.. code-block:: bash

   RuntimeError: normal expects all elements of std >= 0.0

这是 **物理状态** 中的 NaN/Inf 传播到策略网络，使其输出标准差变为负数
或 NaN 导致的。

可能的原因很多，包括 **MuJoCo Warp**（仍在 beta）的潜在缺陷。mjlab
提供两个互补机制：

1. **为了训练稳定** - NaN 终止

添加 ``nan_detection`` 终止项来重置出现 NaN 的环境：

.. code-block:: python

   from mjlab.envs.mdp import terminations as mdp_term
   from mjlab.managers.termination_manager import TerminationTermCfg

   # In your ManagerBasedRlEnvCfg subclass:
   terminations = {
      # Your other terminations...
      "nan_term": TerminationTermCfg(func=mdp_term.nan_detection),
   }

这会把 NaN 环境标记为终止，使其在训练继续的同时被重置。终止以
``Episode_Termination/nan_term`` 记录在指标中。

.. warning::

   这是 **权宜之计**。如果 NaN 与任务目标相关（例如智能体一尝试抓取
   物体就出现 NaN），策略永远学不会完成任务的这一部分。除了这个终止，
   一定要用 ``nan_guard`` 排查 **根本原因**。

2. **为了调试** - NaN 守卫

启用 ``nan_guard`` 在 NaN 出现时捕获仿真状态：

.. code-block:: bash

   uv run train.py --enable-nan-guard True

细节见 :doc:`NaN 守卫文档 <debugging/nan_guard>`。

``nan_guard`` 工具让你可以：

- 检查 NaN 出现那一刻的仿真状态。
- 构造最小可复现示例（MRE）。
- 向 `MuJoCo Warp 团队 <https://github.com/google-deepmind/mujoco_warp/issues>`_
  报告潜在的框架缺陷。

报告隔离良好的问题能帮助改进框架，惠及所有人。

如何检视生成的场景 XML？
~~~~~~~~~~~~~~~~~~~~~~~~

用 ``export-scene`` 脚本把完整场景（XML 与网格资产）写到目录：

.. code-block:: bash

    uv run export-scene g1 --output-dir /tmp/g1

导出的 ``scene.xml`` 可以直接加载进 MuJoCo 做可视化检查或 diff。
这对验证任务配置与物理设置是否正确很有用，也便于制作最小可复现示例
分享给 mjlab 或 MuJoCo Warp 开发者。脚本接受任务 ID、实体别名
（``g1``、``go1``、``yam``）或任意导入路径。完整细节见
:doc:`debugging/export_scene`。

使用 decimation 时接触传感器漏检碰撞
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``decimation > 1`` 时，物理在每个策略步内运行多个子步。一次短暂的接触
（如自碰撞或非法触地）可能在子步循环内出现又消失，等读到传感器时
``found`` 已经是零，奖励和终止都看不到这次事件。

把 ``ContactSensorCfg`` 的 ``history_length`` 设成与 decimation 相等。
传感器随后会保存最近 *N* 个子步的力、力矩和距离。你的奖励或终止函数
可以检视这段历史，检测原本会被漏掉的接触：

.. code-block:: python

    ContactSensorCfg(
        name="self_collision",
        ...,
        fields=("found", "force"),
        history_length=4,  # matches decimation=4
    )

    # In the reward/termination function:
    force_mag = torch.norm(sensor.data.force_history, dim=-1)  # [B, N, H]
    had_contact = (force_mag > 10.0).any(dim=1).any(dim=-1)    # [B]

完整细节见 :ref:`contact-sensor-history`。

.. note::

   带 ``track_air_time=True`` 的脚-地传感器本来就跨子步累积接触状态，
   不需要 history。

.. _faq-sim-forward:

什么时候需要调用 ``sim.forward()``？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

简短回答：几乎肯定不需要。

``sim.forward()`` 包装了 MuJoCo 的 ``mj_forward``——运行完整的正向
动力学管线（运动学、接触、力、约束求解、传感器），但跳过积分，
``qpos``/``qvel`` 保持不变。它把 ``mjData`` 中所有派生量（``xpos``、
``xquat``、``site_xpos``、``cvel``、``sensordata`` 等）带到与当前
``qpos``/``qvel`` 一致的状态。
环境的 ``step()`` 每步调用它一次，恰在观测计算之前，因此观测和指令
总能看到新鲜的派生量。终止、奖励与 step/interval 事件在该调用 *之前*
运行，看到的派生量落后一个物理子步——这是刻意权衡：避免二次
``forward()`` 调用，同时保持 MDP 定义良好（过期程度在所有环境、所有
步上保持一致）。因为事件先于该调用运行，事件写入的任何状态（例如
速度推动）都会被它刷新，对同一步的观测可见。

唯一需要关心的情形是：你在同一条事件或指令里既写状态又读派生量。
例如事件 A 先调用 ``entity.write_root_velocity_to_sim()``（修改
``qvel``），紧接着读取 ``entity.data.root_link_vel_w``（来自
``cvel``），读到的将是写入之前的过期值。

.. warning::

   写方法（``write_root_state_to_sim``、``write_joint_state_to_sim``
   等）直接修改 ``qpos``/``qvel``。读属性（``root_link_pose_w``、
   ``body_link_vel_w`` 等）返回派生量，其新鲜程度只到上一次
   ``sim.forward()``。如果在同一个函数里先写后读，请在两者之间调用
   ``env.sim.forward()``。

更深入的解析见 `Discussion #289
<https://github.com/mujocolab/mjlab/discussions/289>`_。

固定了种子为什么训练还是不可复现？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MuJoCo Warp 尚不保证确定性，相同输入跑同一仿真可能产生略微不同的
输出。这一已知限制记录在
`mujoco_warp#562 <https://github.com/google-deepmind/mujoco_warp/issues/562>`_。

在上游实现确定性之前，即使设置了种子，mjlab 的训练也无法完全复现。

XML 里的 ``<option>`` 标志不生效
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果你在实体 XML 中设置了 ``<flag contact="disable"/>`` 之类的仿真
选项，它们会被悄悄忽略。原因在于 mjlab 组合场景时用 ``MjSpec.attach()``
把实体 spec 挂接到父场景 spec，该操作不会把子 spec 的 ``<option>``
传递给父 spec。这是 MuJoCo 的设计决策：没有办法在多个被挂接的模型间
合理合并引擎选项（时间步长、重力、求解器设置等）。

配置仿真选项请使用任务 Python 配置中的
:class:`~mjlab.sim.sim.MujocoCfg`：

.. code-block:: python

   from mjlab.sim.sim import MujocoCfg, SimulationCfg

   sim=SimulationCfg(
       mujoco=MujocoCfg(
           disableflags=("contact",),
           # timestep=0.01, gravity=(0, 0, -9.81), etc.
       ),
   )

``MujocoCfg`` 把选项直接应用到编译后的模型上，所以始终生效。mjlab
检测到被挂接实体 spec 上存在非默认 ``<option>`` 字段时会发出警告。

渲染与可视化
------------

有哪些可视化选项？
~~~~~~~~~~~~~~~~~~

mjlab 目前支持两个用于策略评估与调试的可视化器：

- **MuJoCo 原生查看器** - MuJoCo 自带的查看器。
- **Viser** - `Viser <https://github.com/nerfstudio-project/viser>`_，
  基于 Web 的 3D 可视化工具。

我们正在探索 **训练时可视化**（如实时采样查看器），尚未提供。

替代方案：mjlab 支持 **视频记录到 Weights & Biases (W&B)**，可以直接
在实验仪表盘中查看回合采样视频。

一次能可视化多少环境？
~~~~~~~~~~~~~~~~~~~~~~

出于性能考虑，查看器只渲染少量环境。

- **离屏渲染器**（录制视频用）：渲染被跟踪环境及其最近的邻居，数量由
  ``ViewerConfig.max_extra_envs`` 控制（默认 2）。
- **原生/Viser 查看器**：受 MuJoCo 几何缓冲区限制（默认 10000 个
  geom）。查看器显示几何预算内放得下的环境。

固定基座机器人为什么全堆在原点而不排成网格？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

固定基座机器人需要一个 **显式重置事件** 才会被放到各自的
``env_origins``。如果你的机器人堆在 (0, 0, 0)：

**常见原因：**

1. **缺少重置事件** - 最常见的问题。
2. **env_spacing 为 0 或极小** - 检查 ``SceneCfg(env_spacing=...)``。
   即使有正确的重置事件，``env_spacing=0.0`` 时所有机器人也在同一
   位置。``env_spacing`` 很小（如 0.01）时，它们会挤在一个远看像一条线
   的小区域里。

**解决方案**：添加调用 ``reset_root_state_uniform`` 的重置事件：

.. code-block:: python

   # In your ManagerBasedRlEnvCfg
   events = {
     # For positioning the base of the robot at env_origins.
     "reset_base": EventTermCfg(
       func=mdp.reset_root_state_uniform,
       mode="reset",
       params={
         "pose_range": {},  # Empty = use default pose + env_origins
         "velocity_range": {},
       },
     ),
     # ... other events
   }

示例操作任务使用了这一模式（见 ``lift_cube_env_cfg.py:85-94``）。

**为什么需要它**：固定基座机器人会被 ``auto_wrap_fixed_base_mocap()``
自动包装进 mocap body，但 mocap 定位只有在你显式调用重置事件时才会
执行。``env_origins`` 偏移在 ``envs/mdp/events.py`` 第 131 行的
``reset_root_state_uniform()`` 内部应用。

示例见 `issue #560 <https://github.com/mujocolab/mjlab/issues/560>`_。

env_origins 如何决定机器人布局？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

机器人间距取决于地形配置：

**平面地形**（``terrain_type="plane"``）：
  - 自动生成近似正方形的网格
  - 网格尺寸：``ceil(sqrt(num_envs))`` 行 x 列
  - 间距由 ``env_spacing`` 参数控制（默认 2.0m）
  - ``env_spacing=2.0`` 的例子：
    - 32 个环境 → 7x5 网格，跨 12m x 8m
    - 4096 个环境 → 64x64 网格，跨 126m x 126m
  - **重要**：``env_spacing=0`` 时所有机器人都在 (0, 0, 0)
  - 实现：``terrain_importer.py:_compute_env_origins_grid()``

**程序化地形**（``terrain_type="generator"``）：
  - 原点从预生成的子地形块加载
  - 网格尺寸：``TerrainGeneratorCfg.num_rows x num_cols``
  - 行索引 = 难度等级（课程模式）
  - 列索引 = 地形类型变体
  - **重要分配行为**：列（地形类型）在环境间均匀分配，而行（难度
    等级）随机采样。这意味着即使 ``num_envs > num_patches``，多个环境
    也可能出生在同一 (行, 列) 块上，而其他块空置。
  - 示例：5x5 网格（25 块）、100 个环境 → 每列恰好 20 个环境，但这
    20 个在 5 行间随机分布，所以有些块仍是空的。
  - 支持 ``randomize_env_origins()`` 在训练期间打乱位置。

如何保证每种地形类型独占一列？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 ``TerrainGeneratorCfg`` 中设置 ``curriculum=True``。这使列分配变为
确定性的，每种地形按归一化比例获得一列。

两种地形类型的例子：

.. code-block:: python

   TerrainGeneratorCfg(
     num_rows=3,
     num_cols=2,
     curriculum=True,  # Required for deterministic column allocation!
     sub_terrains={
       "flat": BoxFlatTerrainCfg(proportion=0.5),  # Gets column 0
       "pillars": HfDiscreteObstaclesTerrainCfg(
         proportion=0.5,  # Gets column 1
       ),
     },
   )

不设 ``curriculum=True`` 时，每块都随机采样，两种地形类型会混杂散布在
所有块上。

**注意**：当 ``num_cols`` 等于地形类型数时，无论比例取值多少（会归一化），
每种地形恰好一列。``num_cols > num_terrain_types`` 时，比例决定每种
地形类型占几列。

什么是平地块采样，它如何影响机器人生成？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

平地块采样在高度场地形上检测机器人可以安全出生的平坦区域。它对高度场
做形态学滤波，找出高度变化在容差以内的圆形区域。

在任意子地形上通过 ``flat_patch_sampling`` 配置：

.. code-block:: python

   from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg

   "obstacles": HfDiscreteObstaclesTerrainCfg(
     ...,
     flat_patch_sampling={
       "spawn": FlatPatchSamplingCfg(
         num_patches=10,      # patches to sample per sub-terrain
         patch_radius=0.5,    # flatness check radius (meters)
         max_height_diff=0.05,  # max height variation within radius
       ),
     },
   )

然后把 ``reset_root_state_from_flat_patches`` 设为重置事件，让机器人
出生在检测到的地块上而不是子地形中心。

**关键细节：**

- 只有高度场（``Hf*``）地形支持真正的平地块检测。Box 地形
  （``Box*``）没有高度场数据可供分析。
- 只要网格配置中有任何子地形配置了 ``flat_patch_sampling``，平地块
  数组就会为 **所有** 格子分配。不产生地块的子地形，其槽位以该子地形
  的出生原点填充，因此 ``reset_root_state_from_flat_patches`` 总能拿到
  有效位置。
- 没有 ``flat_patch_sampling`` 时，用 ``reset_root_state_uniform``，
  机器人在子地形原点（``env_origins``）出生，可带随机偏移。

开发与扩展
----------

可以在自己的仓库里开发自定义任务吗？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

可以。mjlab 有 **插件系统**，让你在独立的仓库中开发任务，同时与核心
无缝集成：

- 你的任务会作为常规条目出现在 ``train`` 和 ``play`` 命令中。
- 你可以独立地对任务仓库做版本管理和维护。

完整指南将在未来版本提供。

资产与兼容性
------------

内置了哪些机器人？
~~~~~~~~~~~~~~~~~~

mjlab 包含两个 **参考机器人**：

- **Unitree Go1**（四足）。
- **Unitree G1**（人形）。

这些机器人用于：

- **机器人接入** 的最小示例。
- **基准任务** 的稳定、经过充分测试的基线。

为保持核心库精简，我们 **不** 计划激进扩充内置机器人库。更多机器人
可能在独立仓库或社区维护的包中提供。

能用 USD 或 URDF 模型吗？
~~~~~~~~~~~~~~~~~~~~~~~~~

不能。mjlab 只接受 **MJCF（MuJoCo XML）** 模型。

- 你需要把 USD 或 URDF 资产 **转换** 为 MJCF。
- 对许多常见机器人，可以直接使用
  `MuJoCo Menagerie <https://github.com/google-deepmind/mujoco_menagerie>`_，
  它提供高质量的 MJCF 模型与资产。

寻求帮助
--------

GitHub Issues
~~~~~~~~~~~~~

GitHub issues 用于：

- **缺陷报告**
- **性能回退**
- **文档缺口**

提 bug 时请附上：

- CUDA 驱动与运行时版本
- GPU 型号
- 最小复现脚本
- 完整的错误日志与堆栈跟踪
- 适当的标签（例如 ``bug``、``performance``、``docs``）

`提交 issue <https://github.com/mujocolab/mjlab/issues>`_

Discussions
~~~~~~~~~~~

GitHub Discussions 用于：

- 使用问题（配置、调试、最佳实践）
- 性能调优技巧
- 资产转换与建模问题
- 设计讨论与路线图想法

`发起讨论 <https://github.com/mujocolab/mjlab/discussions>`_

已知限制
--------

稳定版缺失的功能我们在
https://github.com/mujocolab/mjlab/issues/100 跟踪。查看
`开放 issue <https://github.com/mujocolab/mjlab/issues>`_ 了解当前
正在进行的工作。

如果有东西不工作，或我们遗漏了什么，请
`提交缺陷报告 <https://github.com/mujocolab/mjlab/issues/new>`_。
