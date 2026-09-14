.. _entity_data:

实体数据（Entity Data）
=======================

本页面是 ``EntityData`` 的属性参考。关于 ``entity.data`` 在整个数据访问
体系中的定位概览，见 :ref:`entity`。

所有属性都是 PyTorch 张量，直接建立在 MuJoCo Warp 的 GPU 缓冲区之上，
没有任何拷贝开销。第一维始终是 ``num_envs``，即并行仿真世界的数量。

.. warning::

   读取属性反映的是 ``sim.forward()`` 调用之后的状态。如果在同一个事件
   项里先写仿真状态再读取派生属性，需要在写入和读取之间调用
   ``sim.forward()``。环境步进序列已经这样做了；该警告只针对在同一条
   自定义事件项里混合读写的情况。详细解释见 :ref:`FAQ <faq-sim-forward>`。


参考：根状态
------------

根属性描述实体根 body 的位置、姿态和速度。以 ``_w`` 结尾的属性在世界
坐标系中表示；以 ``_b`` 结尾的属性在实体的基座坐标系中表示。详见
:ref:`frame-conventions`。

每个实体有两个根参考点：**连杆原点**（link origin，MJCF 中定义的 body
坐标系原点）和 **质心**（center of mass, COM）。哪个更相关取决于任务。

.. admonition:: MuJoCo 的混合坐标系 ``qvel``

   对浮动基座实体，自由关节在 ``qvel`` 中存储 6 个自由度。MuJoCo 把
   **线速度** 分量表示在 **世界坐标系** 中，却把 **角速度** 分量表示在
   **本地 body 坐标系** 中。EntityData 规避了这一陷阱：所有 ``_w``
   速度属性都从 ``cvel`` 计算（见下方 :ref:`cvel-section`），完全是
   世界坐标系。如果你直接读取 ``env.sim.data.qvel``，请留意这种混合
   约定。

.. rubric:: 根连杆属性

.. list-table::
   :header-rows: 1
   :widths: 35 20 15 30

   * - 属性
     - 形状
     - 坐标系
     - 描述
   * - ``root_link_pose_w``
     - ``[num_envs, 7]``
     - world
     - 根连杆位置 (3) 与四元数 (4) 拼接
   * - ``root_link_pos_w``
     - ``[num_envs, 3]``
     - world
     - 根连杆位置
   * - ``root_link_quat_w``
     - ``[num_envs, 4]``
     - world
     - 根连杆姿态四元数 (w, x, y, z)
   * - ``root_link_vel_w``
     - ``[num_envs, 6]``
     - world
     - 根连杆线速度 (3) 与角速度 (3) 拼接
   * - ``root_link_lin_vel_w``
     - ``[num_envs, 3]``
     - world
     - 根连杆线速度
   * - ``root_link_ang_vel_w``
     - ``[num_envs, 3]``
     - world
     - 根连杆角速度
   * - ``root_link_lin_vel_b``
     - ``[num_envs, 3]``
     - body
     - 基座坐标系下的根连杆线速度
   * - ``root_link_ang_vel_b``
     - ``[num_envs, 3]``
     - body
     - 基座坐标系下的根连杆角速度

.. rubric:: 根质心属性

.. list-table::
   :header-rows: 1
   :widths: 35 20 15 30

   * - 属性
     - 形状
     - 坐标系
     - 描述
   * - ``root_com_pose_w``
     - ``[num_envs, 7]``
     - world
     - 根质心位置 (3) 与四元数 (4) 拼接
   * - ``root_com_pos_w``
     - ``[num_envs, 3]``
     - world
     - 根质心位置
   * - ``root_com_quat_w``
     - ``[num_envs, 4]``
     - world
     - 根质心姿态四元数 (w, x, y, z)
   * - ``root_com_vel_w``
     - ``[num_envs, 6]``
     - world
     - 根质心线速度 (3) 与角速度 (3) 拼接
   * - ``root_com_lin_vel_w``
     - ``[num_envs, 3]``
     - world
     - 根质心线速度
   * - ``root_com_ang_vel_w``
     - ``[num_envs, 3]``
     - world
     - 根质心角速度
   * - ``root_com_lin_vel_b``
     - ``[num_envs, 3]``
     - body
     - 基座坐标系下的根质心线速度
   * - ``root_com_ang_vel_b``
     - ``[num_envs, 3]``
     - body
     - 基座坐标系下的根质心角速度

.. rubric:: 派生根属性

.. list-table::
   :header-rows: 1
   :widths: 35 20 15 30

   * - 属性
     - 形状
     - 坐标系
     - 描述
   * - ``projected_gravity_b``
     - ``[num_envs, 3]``
     - body
     - 重力向量 (0, 0, -1) 旋转到基座坐标系后的结果。用于度量倾斜：
       完全直立的机器人读数为 ``[0, 0, -1]``。
   * - ``heading_w``
     - ``[num_envs]``
     - world
     - 根 body 前向轴投影到 XY 平面后的朝向角（弧度）。


参考：body 状态
---------------

body 属性给出实体所有 body 的运动学状态。第二维是 ``num_bodies``，即
实体运动树中所有非世界 body 的数量。

.. list-table::
   :header-rows: 1
   :widths: 35 25 15 25

   * - 属性
     - 形状
     - 坐标系
     - 描述
   * - ``body_link_pose_w``
     - ``[num_envs, num_bodies, 7]``
     - world
     - 每个 body 的连杆位置 (3) 与四元数 (4)
   * - ``body_link_pos_w``
     - ``[num_envs, num_bodies, 3]``
     - world
     - 每个 body 的连杆位置
   * - ``body_link_quat_w``
     - ``[num_envs, num_bodies, 4]``
     - world
     - 每个 body 的连杆姿态
   * - ``body_link_vel_w``
     - ``[num_envs, num_bodies, 6]``
     - world
     - 每个 body 的连杆线速度 (3) 与角速度 (3)
   * - ``body_link_lin_vel_w``
     - ``[num_envs, num_bodies, 3]``
     - world
     - 每个 body 的连杆线速度
   * - ``body_link_ang_vel_w``
     - ``[num_envs, num_bodies, 3]``
     - world
     - 每个 body 的连杆角速度
   * - ``body_com_pose_w``
     - ``[num_envs, num_bodies, 7]``
     - world
     - 每个 body 的质心位置 (3) 与四元数 (4)
   * - ``body_com_pos_w``
     - ``[num_envs, num_bodies, 3]``
     - world
     - 每个 body 的质心位置
   * - ``body_com_quat_w``
     - ``[num_envs, num_bodies, 4]``
     - world
     - 每个 body 的质心姿态
   * - ``body_com_vel_w``
     - ``[num_envs, num_bodies, 6]``
     - world
     - 每个 body 的质心线速度 (3) 与角速度 (3)
   * - ``body_com_lin_vel_w``
     - ``[num_envs, num_bodies, 3]``
     - world
     - 每个 body 的质心线速度
   * - ``body_com_ang_vel_w``
     - ``[num_envs, num_bodies, 3]``
     - world
     - 每个 body 的质心角速度
   * - ``body_external_wrench``
     - ``[num_envs, num_bodies, 6]``
     - world
     - 施加在每个 body 上的外力 (3) 与外力矩 (3)
   * - ``body_external_force``
     - ``[num_envs, num_bodies, 3]``
     - world
     - 施加在每个 body 上的外力
   * - ``body_external_torque``
     - ``[num_envs, num_bodies, 3]``
     - world
     - 施加在每个 body 上的外力矩


参考：关节状态
--------------

关节属性覆盖 1 自由度的转动关节和滑动关节。自由关节（根浮动基座自由度）
不包含在内；请使用根状态属性。

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - 属性
     - 形状
     - 描述
   * - ``joint_pos``
     - ``[num_envs, num_joints]``
     - 关节位置，单位为弧度（转动关节）或米（滑动关节）
   * - ``joint_pos_biased``
     - ``[num_envs, num_joints]``
     - 带编码器偏置的关节位置。用于通过域随机化模拟编码器标定误差。
   * - ``joint_vel``
     - ``[num_envs, num_joints]``
     - 关节速度，单位 rad/s 或 m/s
   * - ``joint_acc``
     - ``[num_envs, num_joints]``
     - 关节加速度，单位 rad/s² 或 m/s²
   * - ``actuator_force``
     - ``[num_envs, num_actuators]``
     - 执行空间中的标量执行器输出（按执行器计）。这是经传动雅可比投影
       之前的力。关节空间中的执行器力请改用 ``qfrc_actuator``。


.. _generalized-forces:

参考：广义力
------------

这些属性暴露 MuJoCo 广义力分解中被选中的分量，切片到本实体的铰接关节
自由度。自由关节自由度不包含在内。所有形状都是 ``[num_envs, nv]``，
其中 ``nv`` 是本实体拥有的铰接自由度数量。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 属性
     - 描述
   * - ``qfrc_actuator``
     - 所有执行器产生并映射到关节空间的力。对电机而言是指令力矩乘以
       齿轮比；对位置和速度执行器而言是内部 PD 律计算出的力。当关节
       启用 ``actuatorgravcomp`` 时，重力补偿力也包含在此项中。
   * - ``qfrc_external``
     - 通过 ``xfrc_applied`` 施加到 body 上的笛卡尔力旋量（wrench）对
       关节产生的力，即 :math:`J^\top F` 映射。MuJoCo 并不单独存储该
       项；此属性在 ``forward()`` 之后从其他力分量中恢复出来。

参考：geom 与 site 状态
------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - 属性
     - 形状
     - 描述
   * - ``geom_pose_w``
     - ``[num_envs, num_geoms, 7]``
     - 每个 geom 在世界坐标系中的位置 (3) 与四元数 (4)
   * - ``geom_pos_w``
     - ``[num_envs, num_geoms, 3]``
     - 每个 geom 在世界坐标系中的位置
   * - ``geom_quat_w``
     - ``[num_envs, num_geoms, 4]``
     - 每个 geom 在世界坐标系中的姿态
   * - ``geom_vel_w``
     - ``[num_envs, num_geoms, 6]``
     - 每个 geom 在世界坐标系中的线速度 (3) 与角速度 (3)
   * - ``geom_lin_vel_w``
     - ``[num_envs, num_geoms, 3]``
     - 每个 geom 在世界坐标系中的线速度
   * - ``geom_ang_vel_w``
     - ``[num_envs, num_geoms, 3]``
     - 每个 geom 在世界坐标系中的角速度
   * - ``site_pose_w``
     - ``[num_envs, num_sites, 7]``
     - 每个 site 在世界坐标系中的位置 (3) 与四元数 (4)
   * - ``site_pos_w``
     - ``[num_envs, num_sites, 3]``
     - 每个 site 在世界坐标系中的位置
   * - ``site_quat_w``
     - ``[num_envs, num_sites, 4]``
     - 每个 site 在世界坐标系中的姿态
   * - ``site_vel_w``
     - ``[num_envs, num_sites, 6]``
     - 每个 site 在世界坐标系中的线速度 (3) 与角速度 (3)
   * - ``site_lin_vel_w``
     - ``[num_envs, num_sites, 3]``
     - 每个 site 在世界坐标系中的线速度
   * - ``site_ang_vel_w``
     - ``[num_envs, num_sites, 3]``
     - 每个 site 在世界坐标系中的角速度


参考：肌腱状态
--------------

肌腱属性只为带肌腱驱动执行器的实体填充。

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - 属性
     - 形状
     - 描述
   * - ``tendon_len``
     - ``[num_envs, num_tendons]``
     - 肌腱长度
   * - ``tendon_vel``
     - ``[num_envs, num_tendons]``
     - 肌腱速度


.. _frame-conventions:

坐标系约定
----------

属性名通过后缀标明其参考坐标系。

``_w``（世界坐标系）
    固定的全局坐标系。原点通常在场景原点，坐标轴在整个回合内保持不变。
    当你需要绝对位置时（例如检测机器人是否低于某个高度阈值），世界系
    物理量非常有用。

``_b``（body 坐标系 / 基座坐标系）
    实体根 body 的坐标系，随机器人平移和旋转。大多数观测项使用 body 系
    物理量，因为它们对机器人的朝向不变。用 body 系表达的速度无论机器人
    朝北还是朝南读数都相同，这让策略更容易泛化。

``projected_gravity_b`` 很好地说明了坐标系后缀为什么重要。它把世界系
重力向量 ``[0, 0, -1]`` 旋转到基座坐标系。机器人直立时结果为
``[0, 0, -1]``；机器人倾斜时 x 和 y 分量开始增大，为策略提供直接的
姿态修正信号。

四元数约定
^^^^^^^^^^

所有四元数都采用 ``(w, x, y, z)`` 约定，与 MuJoCo 一致。

缩减状态与派生量
^^^^^^^^^^^^^^^^

EntityData 属性按照对 ``sim.forward()`` 的行为差异分为两类：

**缩减状态。** ``joint_pos`` 和 ``joint_vel`` 直接读取 MuJoCo 的
``qpos`` 和 ``qvel`` 数组。``write_joint_state_to_sim()`` 等写方法直接
修改这些数组，因此读取总是最新的。

**派生量。** 所有位姿与速度属性（``*_pose_w``、``*_vel_w``、
``*_vel_b``）都从 MuJoCo 的内部数组（``xpos``、``xquat``、``cvel``、
``subtree_com`` 等）计算得来，而这些数组只在 ``sim.forward()`` 运行时
更新。如果你写入了 ``qpos``/``qvel``，然后在没有插入 ``forward()`` 的
情况下读取派生属性，读到的将是过期值。

环境步进序列会在正确的时机调用 ``forward()``，因此只有当你在同一条
自定义事件项里既写又读时才需要关心这个问题。详见
:ref:`FAQ <faq-sim-forward>`。

.. _cvel-section:

速度属性如何由 ``cvel`` 计算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MuJoCo 不直接存储世界坐标系下的线速度。它为每个 body 存储一个称为
``cvel`` 的 6 维空间速度（基于质心的速度），布局为
``(angular[3], linear[3])``。该向量表达在 **c 系** 中：一个以
``subtree_com``（body 运动子树的质心）为中心、方向与世界坐标系一致的
坐标系。MuJoCo 用这种表示来提高远离世界原点的机构的数值精度。背景知识
参见
`c-frame variables <https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#c-frame-variables>`_
与 Featherstone 的
`Spatial Algebra <http://royfeatherstone.org/spatial/>`_。

要恢复刚体上任意一点 :math:`\mathbf{p}` 的世界系线速度，我们套用标准的
刚体速度转移公式。令 :math:`\boldsymbol{\omega}` 和
:math:`\mathbf{v}_c` 表示 ``cvel`` 的角速度与线速度分量，
:math:`\mathbf{c}` 表示 ``subtree_com``。由于 c 系与世界系对齐，
:math:`\boldsymbol{\omega}` 已经在世界系中。点 :math:`\mathbf{p}` 处的
线速度为：

.. math::

   \mathbf{v}_p
     = \mathbf{v}_c
       - \boldsymbol{\omega} \times (\mathbf{c} - \mathbf{p})

EntityData 在 ``compute_velocity_from_cvel()`` 中应用该公式：

.. code-block:: python

   def compute_velocity_from_cvel(pos, subtree_com, cvel):
       lin_vel_c = cvel[..., 3:6]
       ang_vel_c = cvel[..., 0:3]
       offset = subtree_com - pos
       lin_vel_w = lin_vel_c - torch.cross(ang_vel_c, offset, dim=-1)
       ang_vel_w = ang_vel_c
       return torch.cat([lin_vel_w, ang_vel_w], dim=-1)

EntityData 中的每个速度属性（``root_link_vel_w``、``body_link_vel_w``、
``geom_vel_w``、``site_vel_w`` 及其质心变体）都使用这一函数，只需代入
相应的点：

- **连杆速度** 使用 ``xpos``（body 坐标系原点）。
- **质心速度** 使用 ``xipos``（body 质心）。
- **geom/site 速度** 使用 ``geom_xpos``/``site_xpos``，``cvel`` 从其
  父 body 查询。


默认位姿与相对量
----------------

``entity.data.default_joint_pos`` 保存实体初始状态配置（``EntityCfg``
的 ``init_state.joint_pos`` 字段）中的关节位置。它的形状是
``[num_envs, num_joints]``，在初始化时复制到所有环境。

相对关节位置是当前关节位置相对该默认值的偏差：

.. code-block:: python

    joint_pos_rel = joint_pos - default_joint_pos

这正是 ``joint_pos_rel`` 观测函数所计算的内容：

.. code-block:: python

    def joint_pos_rel(env, asset_cfg):
        asset = env.scene[asset_cfg.name]
        jnt_ids = asset_cfg.joint_ids
        return (
            asset.data.joint_pos[:, jnt_ids]
            - asset.data.default_joint_pos[:, jnt_ids]
        )

相对关节位置为策略提供了姿态偏差的紧凑表示。机器人处于默认位姿时，
每个元素都是零。

类似地，``default_joint_vel`` 供 ``joint_vel_rel`` 观测函数使用。对
大多数配置而言默认速度为零，此时 ``joint_vel_rel`` 与 ``joint_vel``
完全相同。这一层间接存在意义在于：允许运动模仿等任务使用非零的参考
速度。

关节位置动作配置中的 ``use_default_offset=True`` 选项把
``default_joint_pos`` 作为动作空间的零点，因此网络输出为零即命令机器人
回到默认位姿。这是运动（locomotion）任务的标准配置。
