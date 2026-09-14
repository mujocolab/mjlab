.. _migration_isaac_lab:

从 Isaac Lab 迁移
=================

.. warning::

   本指南仍在完善中。随着更多用户迁移，我们会持续补充更多模式与边界
   情况。如有未覆盖的内容，请在 GitHub 上提交 issue 或发起讨论：

   - Issues: https://github.com/mujocolab/mjlab/issues
   - Discussions: https://github.com/mujocolab/mjlab/discussions

速览
----

大多数 Isaac Lab *基于管理器* 的任务配置只需少量改动即可移植到
``mjlab``：

- 整体 **MDP 结构相同** （奖励、观测、动作、指令、终止、事件、课程的
  各管理器）。
- **环境基类相似**，只是命名略有差异。
- 最大的变化是 **配置风格**：Isaac Lab 使用嵌套的 ``@configclass``
  定义；``mjlab`` 使用配置对象字典。

如果你熟悉 Isaac Lab 的基于管理器的 API，迁移基本上是机械性工作。

关键差异
--------

1. 导入路径
~~~~~~~~~~

Isaac Lab：

.. code-block:: python

   from isaaclab.envs import ManagerBasedRLEnv

mjlab：

.. code-block:: python

   from mjlab.envs import ManagerBasedRlEnvCfg

.. note::

   ``mjlab`` 使用一致的 ``CamelCase`` 命名约定（例如用 ``RlEnv``
   而不是 ``RLEnv``）。

2. 配置结构
~~~~~~~~~~~

Isaac Lab 为管理器项使用嵌套的 ``@configclass`` 块。``mjlab`` 改用
**普通字典** 把名称映射到配置对象，便于构造变体、合并配置或以编程方式
生成。这一设计决策的完整背景见
`PR #292 <https://github.com/mujocolab/mjlab/pull/292>`_。

**Isaac Lab:**

.. code-block:: python

   @configclass
   class RewardsCfg:
       """Reward terms for the MDP."""

       motion_global_anchor_pos = RewTerm(
           func=mdp.motion_global_anchor_position_error_exp,
           weight=0.5,
           params={"command_name": "motion", "std": 0.3},
       )
       motion_global_anchor_ori = RewTerm(
           func=mdp.motion_global_anchor_orientation_error_exp,
           weight=0.5,
           params={"command_name": "motion", "std": 0.4},
       )

**mjlab:**

.. code-block:: python

   rewards = {
       "motion_global_anchor_pos": RewardTermCfg(
           func=mdp.motion_global_anchor_position_error_exp,
           weight=0.5,
           params={"command_name": "motion", "std": 0.3},
       ),
       "motion_global_anchor_ori": RewardTermCfg(
           func=mdp.motion_global_anchor_orientation_error_exp,
           weight=0.5,
           params={"command_name": "motion", "std": 0.4},
       ),
   }

   cfg = ManagerBasedRlEnvCfg(
       scene=scene,
       rewards=rewards,
       # ... other manager dictionaries:
       # observations=..., actions=..., commands=..., terminations=...,
       # events=..., curriculum=...
   )

该模式适用于所有管理器：

- ``rewards``
- ``observations``
- ``actions``
- ``commands``
- ``terminations``
- ``events``
- ``curriculum``

3. 场景配置
~~~~~~~~~~~

``mjlab`` 的场景搭建 **更简单**：

- 没有 Omniverse / USD 场景图，没有 ``prim_path`` 管理。
- 资产是纯 MuJoCo（MJCF），配合施加在 ``mujoco.MjSpec`` 上的修饰
  dataclass。
- 光源、材质、纹理和传感器作为 ``SceneCfg`` 与机器人配置的一部分
  配置。

**Isaac Lab:**

.. code-block:: python

   from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
   from isaaclab.scene import InteractiveSceneCfg
   from isaaclab.sensors import ContactSensorCfg
   from isaaclab.terrains import TerrainImporterCfg
   import isaaclab.sim as sim_utils
   from isaaclab.assets import ArticulationCfg, AssetBaseCfg

   @configclass
   class MySceneCfg(InteractiveSceneCfg):
       """Configuration for the terrain scene with a legged robot."""

       # ground terrain
       terrain = TerrainEntityCfg(
           prim_path="/World/ground",
           terrain_type="plane",
           collision_group=-1,
           physics_material=sim_utils.RigidBodyMaterialCfg(
               friction_combine_mode="multiply",
               restitution_combine_mode="multiply",
               static_friction=1.0,
               dynamic_friction=1.0,
           ),
           visual_material=sim_utils.MdlFileCfg(
               mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
               project_uvw=True,
           ),
       )
       # lights
       light = AssetBaseCfg(
           prim_path="/World/light",
           spawn=sim_utils.DistantLightCfg(
               color=(0.75, 0.75, 0.75), intensity=3000.0
           ),
       )
       sky_light = AssetBaseCfg(
           prim_path="/World/skyLight",
           spawn=sim_utils.DomeLightCfg(
               color=(0.13, 0.13, 0.13), intensity=1000.0
           ),
       )
       robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

**mjlab:**

.. code-block:: python

   from dataclasses import replace

   from mjlab.scene import SceneCfg
   from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
   from mjlab.utils.spec_config import ContactSensorCfg
   from mjlab.terrains import TerrainEntityCfg

   # Configure contact sensor
   self_collision_sensor = ContactSensorCfg(
       name="self_collision",
       subtree1="pelvis",
       subtree2="pelvis",
       data=("found",),
       reduce="netforce",
       num=10,  # report up to 10 contacts
   )

   # Add sensor to robot config
   g1_cfg = replace(get_g1_robot_cfg(), sensors=(self_collision_sensor,))

   # Create scene
   SCENE_CFG = SceneCfg(
       terrain=TerrainEntityCfg(terrain_type="plane"),
       entities={"robot": g1_cfg},
   )

关键变化：

- 没有 USD ``prim_path`` 或克隆；场景直接用 MuJoCo 描述。
- 材质、光源和可视化属性经 ``MjSpec`` 修饰 dataclass 应用。
- 替你完成这些修改的辅助工具见仓库中的 ``mjlab.utils.spec_config``。
- 所有配置中的 ``asset_name`` 已统一为 ``entity_name``。

完整示例对照
------------

对照已经完成移植的具体任务是学习这一模式的好办法：

- Isaac Lab 实现（Beyond Mimic）：

  - https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py

- mjlab 实现：

  - https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py

你会发现：

- ``mjlab`` 的管理器字典与 Isaac Lab 的配置类一一对应，
- 奖励、观测、指令和终止逻辑几乎完全相同，
- 场景与资产搭建简化为纯 MuJoCo。

迁移清单
--------

移植任务时把下面的清单当速查表用：

1. **基类与导入**

   - 把 Isaac Lab 导入（例如
     ``from isaaclab.envs import ManagerBasedRLEnv``）替换为对应的
     ``mjlab`` 导入（例如
     ``from mjlab.envs import ManagerBasedRlEnvCfg``）。

2. **管理器配置**

   - 把每个 Isaac Lab ``@configclass`` 管理器（``RewardsCfg``、
     ``ObservationsCfg`` 等）转换为配置对象字典。
   - 把这些字典传给 ``ManagerBasedRlEnvCfg``。

3. **场景与资产**

   - 把 ``InteractiveSceneCfg`` 替换为 ``SceneCfg`` 实例。
   - 把 USD / ``prim_path`` 逻辑替换为 MuJoCo 资产配置与场景实体
     （例如来自 ``asset_zoo`` 的机器人）。

4. **传感器与接触处理**

   - 把 Isaac Lab 的 ``ContactSensorCfg`` 转换为
     ``mjlab.utils.spec_config.ContactSensorCfg`` 并挂到机器人配置上。

5. **RL 入口**

   - 确保训练脚本或入口使用正确的任务 ID 与环境配置（视项目结构而定，
     通过 Gymnasium 注册或直接构造）。

技巧与支持
----------

1. 查看仓库中的示例：

   - ``src/mjlab/tasks/``

2. 遇到困难时：

   - 提交 issue：https://github.com/mujocolab/mjlab/issues
   - 发起讨论：https://github.com/mujocolab/mjlab/discussions

3. 记住 MuJoCo 与 Isaac Sim 的差异：

   - 一些 Omniverse / USD 渲染特性没有直接对应物。
   - 先对齐 **物理与观测**，再按需打磨视觉。
