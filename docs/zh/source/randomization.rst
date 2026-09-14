.. _domain_randomization:

域随机化
========

域随机化在训练期间扰动物理参数，使策略对建模误差和现实世界的变化保持
鲁棒。本指南介绍如何用 ``EventTermCfg`` 和 ``dr`` 模块把随机化项挂到
环境上。

快速上手
--------

使用一个 ``EventTermCfg``，调用 ``dr`` 中带类型签名的函数，并给出
**取值范围** 和描述施加方式的 **操作**。

.. code-block:: python

    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    foot_friction: EventTermCfg = EventTermCfg(
        mode="reset",  # randomize each episode
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=[".*_foot.*"]),
            "ranges": (0.3, 1.2),
            "operation": "abs",
        },
    )

每个 ``dr`` 函数都带有 ``@requires_model_fields`` 装饰器，自动跟踪哪些
字段需要展开为按 world 存储，**以及** 保持 :ref:`派生量
<dr-recomputation>` 一致所需的 ``RecomputeLevel``。

``mode`` 参数控制事件何时触发：

* ``"startup"`` 在初始化时随机化一次
* ``"reset"`` 在每次回合重置时随机化
* ``"interval"`` 按固定时间间隔随机化


可用函数
--------

模型字段函数
^^^^^^^^^^^^

每个函数写入 ``sim.model``（MuJoCo Warp 模型）上的单个字段。例如
``dr.geom_friction`` 写入 ``sim.model.geom_friction``，
``dr.body_mass`` 写入 ``sim.model.body_mass``，以此类推。大多数函数
共享签名 ``(env, env_ids, ranges, ...)``，由 ``distribution`` 和
``operation`` 控制采样与施加（见 :ref:`dr-parameters`）。有些函数的
名字比底层 MuJoCo 字段更易读，此时原始字段名以别名形式提供
（``dr.body_com_offset`` 与 ``dr.body_ipos`` 是同一个函数）。

.. rubric:: Geom 字段

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - 函数
     - MuJoCo 字段
     - 描述
     - 备注
   * - ``dr.geom_friction``
     - ``geom_friction``
     - 滑动、扭转与滚动摩擦系数
     - 默认轴：0（仅切向）
   * - ``dr.geom_pos``
     - ``geom_pos``
     - geom 在父 body 系中的位置
     -
   * - ``dr.geom_quat``
     - ``geom_quat``
     - geom 坐标系的姿态
     - 接受 roll/pitch/yaw 范围（弧度）；与默认值复合
   * - ``dr.geom_rgba``
     - ``geom_rgba``
     - 颜色与透明度（RGBA）
     -
   * - ``dr.geom_size``
     - ``geom_size``
     - geom 特定的尺寸参数（半径、半长等）
     - 自动重算 ``geom_rbound`` 与 ``geom_aabb``
   * - ``dr.geom_matid``
     - ``geom_matid``
     - geom 渲染使用的烘焙材质
     - 从 ``asset_cfg.material_names`` 均匀采样

.. rubric:: Body 字段

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - 函数
     - MuJoCo 字段
     - 描述
     - 备注
   * - ``dr.body_mass``
     - ``body_mass``
     - body 质量
     - 触发 ``set_const`` 重算
   * - ``dr.body_com_offset``（别名 ``body_ipos``）
     - ``body_ipos``
     - 质心相对 body 坐标系的位置
     - 触发 ``set_const``
   * - ``dr.body_pos``
     - ``body_pos``
     - body 坐标系在父坐标系中的位置
     - 触发 ``set_const_0``
   * - ``dr.body_quat``
     - ``body_quat``
     - body 坐标系的姿态
     - 接受 roll/pitch/yaw 范围（弧度）；与默认值复合；触发
       ``set_const_0``

.. rubric:: 关节字段

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - 函数
     - MuJoCo 字段
     - 描述
     - 备注
   * - ``dr.joint_damping``（别名 ``dof_damping``）
     - ``dof_damping``
     - 与速度成正比的被动阻尼力
     -
   * - ``dr.joint_armature``（别名 ``dof_armature``）
     - ``dof_armature``
     - 附加转子惯量（建模带减速箱的传动）
     - 触发 ``set_const_0``
   * - ``dr.joint_friction``（别名 ``dof_frictionloss``）
     - ``dof_frictionloss``
     - 关节中的干摩擦损耗
     -
   * - ``dr.joint_stiffness``（别名 ``jnt_stiffness``）
     - ``jnt_stiffness``
     - 拉向参考位置的弹簧刚度
     -
   * - ``dr.joint_limits``（别名 ``jnt_range``）
     - ``jnt_range``
     - 关节位置上下限
     -
   * - ``dr.joint_default_pos``（别名 ``qpos0``）
     - ``qpos0``
     - 参考关节位置（零弹簧平衡点）
     - 触发 ``set_const_0``

.. rubric:: Site 字段

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - 函数
     - MuJoCo 字段
     - 描述
     - 备注
   * - ``dr.site_pos``
     - ``site_pos``
     - site 坐标系在父 body 系中的位置
     -
   * - ``dr.site_quat``
     - ``site_quat``
     - site 坐标系的姿态
     - 接受 roll/pitch/yaw 范围（弧度）；与默认值复合

.. rubric:: 相机字段

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - 函数
     - MuJoCo 字段
     - 描述
     - 备注
   * - ``dr.cam_fovy``
     - ``cam_fovy``
     - 垂直视场角（度）
     -
   * - ``dr.cam_pos``
     - ``cam_pos``
     - 相机在父 body 系中的位置
     -
   * - ``dr.cam_quat``
     - ``cam_quat``
     - 相机姿态
     - 接受 roll/pitch/yaw 范围（弧度）；与默认值复合
   * - ``dr.cam_intrinsic``
     - ``cam_intrinsic``
     - 焦距与主点 ``[fx, fy, cx, cy]``
     -

.. rubric:: 光源字段

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - 函数
     - MuJoCo 字段
     - 描述
     - 备注
   * - ``dr.light_pos``
     - ``light_pos``
     - 光源在父 body 系中的位置
     -
   * - ``dr.light_dir``
     - ``light_dir``
     - 光源方向向量
     -
   * - ``dr.light_diffuse``
     - ``light_diffuse``
     - 漫反射 RGB 颜色
     -
   * - ``dr.light_specular``
     - ``light_specular``
     - 镜面反射 RGB 颜色
     -
   * - ``dr.light_ambient``
     - ``light_ambient``
     - 环境光 RGB 颜色
     -
   * - ``dr.light_attenuation``
     - ``light_attenuation``
     - 常数、一次与二次衰减系数
     -
   * - ``dr.light_cutoff``
     - ``light_cutoff``
     - 聚光灯半锥角（度）
     - 对方向光无效
   * - ``dr.light_exponent``
     - ``light_exponent``
     - 聚光灯角度衰减指数
     - 对方向光无效

.. rubric:: 材质字段

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - 函数
     - MuJoCo 字段
     - 描述
     - 备注
   * - ``dr.mat_rgba``
     - ``mat_rgba``
     - 材质 RGBA 颜色（为纹理着色）
     -
   * - ``dr.mat_emission``
     - ``mat_emission``
     - 自发光倍率
     - 用于 MuJoCo Warp RGB 渲染
   * - ``dr.mat_specular``
     - ``mat_specular``
     - ``[0, 1]`` 内的镜面反射强度
     - 缩放 MuJoCo Warp RGB 的镜面分量
   * - ``dr.mat_shininess``
     - ``mat_shininess``
     - ``[0, 1]`` 内的表面光泽度
     - 用于 MuJoCo Warp RGB 渲染
   * - ``dr.mat_texrepeat``
     - ``mat_texrepeat``
     - S/T 方向的纹理重复次数
     - 只影响带纹理的材质；取值应保持为正
   * - ``dr.mat_texid``
     - ``mat_texid``
     - 指派给材质 ``mjtTextureRole`` 槽位的纹理（默认 RGB）
     - 从 ``asset_cfg.texture_names`` 均匀采样。用 ``role`` 指定其他
       纹理角色。

.. rubric:: 接触对字段

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - 函数
     - MuJoCo 字段
     - 描述
     - 备注
   * - ``dr.pair_friction``
     - ``pair_friction``
     - 逐对摩擦覆盖 ``[tangent1, tangent2, spin, roll1, roll2]``
     - 默认轴：0（tangent1，需要 ``condim >= 3``）。用 ``isotropic=True``
       使 tangent2 = tangent1（及 roll2 = roll1）。覆盖显式定义的
       `<contact><pair> <https://mujoco.readthedocs.io/en/stable/XMLreference.html#contact-pair>`_
       元素的逐 geom 摩擦。见 :ref:`dr-pair-friction`。

.. rubric:: 肌腱字段

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - 函数
     - MuJoCo 字段
     - 描述
     - 备注
   * - ``dr.tendon_damping``
     - ``tendon_damping``
     - 沿肌腱的与速度成正比的阻尼
     -
   * - ``dr.tendon_stiffness``
     - ``tendon_stiffness``
     - 沿肌腱的弹簧刚度
     -
   * - ``dr.tendon_friction``（别名 ``tendon_frictionloss``）
     - ``tendon_frictionloss``
     - 沿肌腱的干摩擦损耗
     -
   * - ``dr.tendon_armature``
     - ``tendon_armature``
     - 与肌腱速度相关的惯量
     - 触发 ``set_const_0``

实体级函数
^^^^^^^^^^

上面的函数都写入单个 ``sim.model`` 字段。下面的函数则工作在 mjlab 实体
层级，因为它们一次触碰多个模型字段，或修改不驻留在 MuJoCo 模型上的实体
状态。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 函数
     - 作用
   * - ``dr.pseudo_inertia``
     - 通过伪惯量矩阵参数化（Rucker & Wensing 2022）对 ``body_mass``、
       ``body_ipos``、``body_inertia`` 和 ``body_iquat`` 做物理一致的
       关节级联合随机化。保证正定惯量张量与正质量。细节见
       :ref:`dr-pseudo-inertia`。
   * - ``dr.pd_gains``
     - 联合随机化刚度（kp）与阻尼（kd）。对 ``BuiltinPositionActuator``
       和 ``XmlActuator`` 写入 ``actuator_gainprm`` 与
       ``actuator_biasprm``；对 ``IdealPdActuator`` 直接设置实体上的
       增益。三者都支持内联延迟字段。
   * - ``dr.effort_limits``
     - 随机化执行器力范围（``actuator_forcerange``）。对
       ``IdealPdActuator`` 同时更新实体内部的力限制。支持
       ``BuiltinPositionActuator``、``XmlActuator`` 和
       ``IdealPdActuator``。
   * - ``dr.encoder_bias``
     - 给位置读数加固定逐关节偏置，模拟编码器标定误差。写入
       ``entity.data.encoder_bias``，而不是 MuJoCo 模型。


.. _dr-pseudo-inertia:

伪惯量随机化
^^^^^^^^^^^^

``dr.pseudo_inertia`` 联合随机化 ``body_mass``、``body_ipos``、
``body_inertia`` 和 ``body_iquat``，同时保证任意扰动幅度下的物理一致性。
独立随机化这些字段可能产生负质量、虚数主惯量矩或三角不等式违背，
而 ``pseudo_inertia`` 通过 *伪惯量矩阵* :math:`J \succ 0` 参数化惯量，
确保结果始终物理有效。

伪惯量矩阵 :math:`J` 是 :math:`4 \times 4` 对称正定矩阵，把质量、质心和
完整旋转惯量张量编码在同一个对象中：

.. math::

   J = \begin{bmatrix}
     \Sigma & h \\
     h^\top & m
   \end{bmatrix}, \qquad
   \Sigma = \tfrac{1}{2}\operatorname{tr}(I)\,I_3 - I, \qquad
   h = m\,c

其中 :math:`m` 是质量（``body_mass``），:math:`c` 是质心
（``body_ipos``），:math:`d` 是主惯量矩向量（``body_inertia``），而
:math:`I` 是 body 系原点处的 :math:`3 \times 3` 惯量张量。惯量张量的
构造方式是：把对角主惯量矩旋转到 body 系，再应用平行轴定理：

.. math::

   I_{\text{com}} &= V \operatorname{diag}(d)\, V^\top,
   \qquad V = R(q)^\top \\
   I &= I_{\text{com}} + m\bigl(\lVert c \rVert^2 I_3 - c\,c^\top\bigr)

其中 :math:`q` 是 body 到主轴系的四元数（``body_iquat``），:math:`R(q)`
是其旋转矩阵。

数学方法遵循 `Rucker & Wensing, "Smooth Parameterization of Rigid-Body
Inertia," IEEE RA-L 2022 <https://par.nsf.gov/servlets/purl/10347458>`_。
:math:`J` 经 Cholesky 分解为 :math:`J = LL^\top`。扰动通过上三角矩阵
:math:`U` 施加：

.. math::

   J' = (UL)(UL)^\top

对任意 :math:`U` 该矩阵都保证正定。受扰动的惯量张量随后被分解回 MuJoCo
字段：逆平行轴定理把 :math:`I` 移回质心，特征分解提取主惯量矩
（``body_inertia``）与主轴系旋转（``body_iquat``）。这对任意扰动幅度都
是精确的。

:math:`U` 的 10 个参数控制不同的物理效应：

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - 参数
     - 物理效应
   * - ``alpha``
     - 全局质量密度对数缩放。质量与所有主惯量矩按 :math:`e^{2\alpha}`
       缩放。质心不变。
   * - ``d1, d2, d3``
     - 惯量系中沿 x、y、z 的轴向拉伸/压缩。用 ``d_range`` 把三者设为
       同一范围（各向同性）。
   * - ``s12, s13, s23``
     - xy、xz、yz 平面内的剪切扰动。非对称地重新分布质量；产生非对角
       惯量贡献。
   * - ``t1, t2, t3``
     - 沿 x、y、z（body 系）平移质心。纯 ``t1`` 平移时质量不变，
       ``body_ipos[0]`` 恰好平移 ``t1``。用 ``t_range`` 把三者设为同一
       范围。

.. rubric:: 示例

.. code-block:: python

    events = {
        # Isotropic mass scaling + small COM variation.
        "body_inertia_dr": EventTermCfg(
            mode="reset",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["torso"]),
                "alpha_range": (-0.1, 0.1),   # ±10% mass/inertia scaling
                "t_range": (-0.02, 0.02),     # ±2 cm COM shift
            },
        ),
        # Anisotropic stretching (x stiffer than y/z).
        "body_inertia_aniso_dr": EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
                "alpha_range": (-0.2, 0.2),
                "d1_range": (-0.1, 0.1),
                "d2_range": (-0.3, 0.3),
                "d3_range": (-0.3, 0.3),
            },
        ),
    }


.. _dr-safety:

运行时修改模型的安全性
----------------------

在 C MuJoCo 中，运行时修改 ``mjModel`` 字段可能不安全：有些改动会使
内部加速结构（BVH）失效，或让派生量过期。MuJoCo Warp 的碰撞管线不同，
因此这些顾虑大多不成立。

两个架构差异最为关键：

1. **没有碰撞 BVH。** C MuJoCo 为中相碰撞剔除构建静态包围盒层级树
   （BVH）。修改静态 body 的 ``body_pos``/``body_quat`` 会使该树失效。
   MuJoCo Warp 使用 NxN 或 sweep-and-prune 宽相，不存在会失效的静态
   树。

2. **局部包围盒。** MuJoCo Warp 的 ``geom_aabb`` 是 *局部* 包围盒
   （geom 系中的中心 + 半尺寸）。宽相每步用正向运动学得到的
   ``geom_xpos``/``geom_xmat`` 把它变换到世界系。C MuJoCo 的 BVH 缓存
   世界系包围盒，因此任何 ``geom_pos``/``geom_quat`` 的改动都会使其
   过期。

下表列出哪些字段在 mjlab 中可以安全随机化，以及与 C MuJoCo 的对比。

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 38

   * - 字段
     - C MuJoCo
     - mjlab / MuJoCo Warp
     - 差异原因
   * - ``body_pos``、``body_quat``
     - 配合 ``mj_setConst`` 安全，但 **对静态 body 不安全**
       （使中相 BVH 失效）
     - 配合 ``set_const_0`` **安全**
     - 没有会失效的碰撞 BVH（见上）。无论 ``body_treeid`` 如何，所有
       body 都会执行 FK。见下方 :ref:`静态 body 注意事项
       <dr-static-body-caveat>`。
   * - ``body_mass``、``body_inertia``、``body_ipos``、``body_iquat``
     - 配合 ``mj_setConst`` 安全
     - 配合 ``set_const`` **安全**
     - 方法相同。``dr.pseudo_inertia`` 联合随机化全部四项并保证物理
       一致性。
   * - ``geom_pos``、``geom_quat``
     - **不安全** （不支持 ``mj_setConst`` ）
     - 对动态 body 上的 geom **安全**
     - FK 每步从 ``geom_pos``/``geom_quat`` 重算
       ``geom_xpos``/``geom_xmat``，局部 ``geom_aabb`` 保持有效
       （见上）。见下方 :ref:`静态 body 注意事项
       <dr-static-body-caveat>`。
   * - ``geom_size``
     - **不安全**
     - **安全** （自动重算包围盒）
     - ``dr.geom_size`` 在写入新尺寸后内联重算 ``geom_rbound`` 与
       ``geom_aabb``。仅支持图元类型（球、胶囊、椭球、圆柱、盒）。
   * - ``geom_rbound``、``geom_aabb``
     - **不安全** （内部派生量）
     - **不随机化** （派生量）
     - 宽相加速数据。只在模型加载时设置一次。``geom_size`` 改变时
       需要重算。
   * - ``geom_friction``、``geom_rgba``
     - 安全
     - **安全**
     - 无派生量。接触摩擦每步直接读取。
   * - ``dof_armature``
     - 配合 ``mj_setConst`` 安全
     - 配合 ``set_const_0`` **安全**
     - 方法相同。
   * - ``dof_damping``、``dof_frictionloss``、``jnt_stiffness``、
       ``jnt_range``
     - 安全
     - **安全**
     - 无派生量。
   * - ``qpos0``
     - 配合 ``mj_setConst`` 安全
     - 配合 ``set_const_0`` **安全**
     - 方法相同。
   * - ``tendon_stiffness``、``tendon_damping``、``tendon_frictionloss``
     - 大多安全（从零切换或切换到零时需 ``mj_setConst``）
     - **安全**
     - MuJoCo Warp 不使用让零/非零切换变得特殊的休眠机制。
   * - ``tendon_armature``
     - 配合 ``mj_setConst`` 安全
     - 配合 ``set_const_0`` **安全**
     - 经 ``smooth.tendon_armature()`` 进入质量矩阵。方法与
       ``dof_armature`` 相同。
   * - ``actuator_gainprm``、``actuator_biasprm``
     - 大多安全（dampratio 执行器需 ``mj_setConst``）
     - **安全**
     - mjlab 的 ``dr.pd_gains`` 在内部处理 dampratio。
   * - ``site_pos``、``site_quat``
     - 大多安全（跟踪相机/光源需 ``mj_setConst``）
     - **安全**
     - site 经 FK 重算。典型 RL 使用中不存在跟踪相机问题。
   * - ``bvh_aabb``、``oct_aabb``、``oct_coeff``
     - **不安全**
     - 不适用 / 不随机化
     - MuJoCo Warp 只在渲染中使用 BVH，不用于碰撞。八叉树数据
       （``oct_*``）用于 SDF 碰撞，不应修改。

.. _dr-static-body-caveat:

.. admonition:: ``geom_pos``/``geom_quat`` 与 ``body_pos``/``body_quat``
   的静态 body 注意事项

   MuJoCo Warp 的正向运动学会跳过既 **焊接在世界** （``body_weldid == 0``）
   又 **不是 mocap body 后代** （``body_mocapid[root] == -1``）的 geom。
   对这类 geom，``geom_xpos``/``geom_xmat`` 只在 ``make_data``
   时计算一次，之后不再更新。修改 ``geom_pos`` 或父 ``body_pos`` 会让
   世界系碰撞位置过期。

   实践中这只会影响直接放在 XML ``<worldbody>`` 上的裸 ``<geom>``
   元素（如地面），且它们不属于任何 mjlab 实体。所有 mjlab 实体
   （包括固定基座实体）都会被 ``auto_wrap_fixed_base_mocap`` 自动
   包装进 mocap body，因此豁免于 FK 跳过。内置 ``dr`` 函数以实体上的
   命名 body/geom 为目标，所以始终安全。

.. _dr-geom-size:

``geom_size`` 重算的工作方式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

运行时修改 ``geom_size`` 需要更新宽相每步读取的两个派生字段：

- ``geom_rbound``：包围球半径（用于球体过滤剔除）
- ``geom_aabb``：局部轴对齐包围盒（用于 AABB/OBB 剔除）

两者都在模型加载时从 ``MjModel`` 拷贝而来，之后 MuJoCo Warp 从不重算。
``dr.geom_size`` 的处理方式是：写入新尺寸后内联重算这两个字段
（纯 PyTorch，无需 Warp kernel）。

公式依类型而定：

.. list-table::
   :header-rows: 1
   :widths: 18 30 30

   * - geom 类型
     - ``geom_rbound``
     - ``geom_aabb`` 半尺寸
   * - 球
     - ``s[0]``
     - ``(s[0], s[0], s[0])``
   * - 胶囊
     - ``s[0] + s[1]``
     - ``(s[0], s[0], s[0] + s[1])``
   * - 圆柱
     - ``sqrt(s[0]² + s[1]²)``
     - ``(s[0], s[0], s[1])``
   * - 椭球
     - ``max(s[0], s[1], s[2])``
     - ``(s[0], s[1], s[2])``
   * - 盒
     - ``sqrt(s[0]² + s[1]² + s[2]²)``
     - ``(s[0], s[1], s[2])``

平面、高度场、网格和 SDF geom 不受支持，因为它们的包围盒来自顶点数据
或为无穷大，无法从 ``geom_size`` 推导。


.. _dr-pair-friction:

``pair_friction`` 与各向同性摩擦
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pair_friction`` 为每个接触对存储五个摩擦系数：

.. list-table::
   :header-rows: 1
   :widths: 10 20 20 50

   * - 索引
     - 名称
     - 何时生效
     - 含义
   * - 0
     - tangent1
     - ``condim >= 3``
     - 沿接触系第一切向轴的滑动摩擦
   * - 1
     - tangent2
     - ``condim >= 3``
     - 沿第二切向轴的滑动摩擦
   * - 2
     - spin
     - ``condim >= 4``
     - 绕接触法线的扭转摩擦
   * - 3
     - roll1
     - ``condim = 6``
     - 绕第一切向轴的滚动摩擦
   * - 4
     - roll2
     - ``condim = 6``
     - 绕第二切向轴的滚动摩擦

MuJoCo 的标准 geom 摩擦是 3 向量 ``[tangential, torsional, rolling]``，
会自动扩展为 5 向量：把切向系数复制到 tangent1 和 tangent2，把滚动系数
复制到 roll1 和 roll2。对接触对覆盖，五个值独立存储，因此必须显式维护
对称性。传 ``isotropic=True`` 即可强制：采样之后，只要目标轴包含 3 或
4，tangent2 就会被覆写为 tangent1，roll2 被覆写为 roll1。

**按 condim 给出建议。**

**condim = 3。** 两个切向轴都生效。采样 tangent1 并用
``isotropic=True`` 令 tangent2 与之相等。传 ``shared_random=True`` 让
``asset_cfg`` 选中的所有接触对在每个环境内共享同一采样值（环境之间
仍然相互独立）：

.. code-block:: python

   dr.pair_friction(
       env,
       env_ids=None,
       ranges=(0.4, 1.0),
       operation="abs",
       asset_cfg=SceneEntityCfg("robot", pair_names=("foot1_floor", "foot2_floor")),
       axes=[0],           # sample tangent1
       shared_random=True, # all pairs selected by asset_cfg share one value per env
       isotropic=True,     # tangent2 = tangent1
   )

**condim = 4。** 自旋（轴 2）额外生效。滑动摩擦按上述方式随机化。若
自旋也要随机化，请用独立的范围单独调用：

.. code-block:: python

   dr.pair_friction(
       env,
       env_ids=None,
       ranges=(0.4, 1.0),
       operation="abs",
       asset_cfg=SceneEntityCfg("robot", pair_names=("foot1_floor", "foot2_floor")),
       axes=[0],
       shared_random=True,
       isotropic=True,
   )
   dr.pair_friction(
       env,
       env_ids=None,
       ranges=(0.003, 0.01),
       operation="abs",
       asset_cfg=SceneEntityCfg("robot", pair_names=("foot1_floor", "foot2_floor")),
       axes=[2],
       shared_random=True,
   )

**condim = 6。** 全部五个轴生效。滑动摩擦按上述方式随机化。要对称地
随机化滚动摩擦，以 ``isotropic=True`` 指定轴 3，使 roll2 等于 roll1：

.. code-block:: python

   dr.pair_friction(
       env,
       env_ids=None,
       ranges=(0.4, 1.0),
       operation="abs",
       asset_cfg=SceneEntityCfg("robot", pair_names=("foot1_floor", "foot2_floor")),
       axes=[0],
       shared_random=True,
       isotropic=True,
   )
   dr.pair_friction(
       env,
       env_ids=None,
       ranges=(0.0001, 0.001),
       operation="abs",
       asset_cfg=SceneEntityCfg("robot", pair_names=("foot1_floor", "foot2_floor")),
       axes=[3],
       shared_random=True,
       isotropic=True,  # roll2 = roll1
   )


没有 ``dr`` 函数的字段
^^^^^^^^^^^^^^^^^^^^^^

按需添加
""""""""

这些连续模型字段本可以用标准 ``dr.*`` 函数，但目前还没有。会随需求
出现而添加。

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 类别
     - 字段
     - 备注
   * - Body
     - ``body_gravcomp``
     - 重力补偿权重。需要 ``set_const_fixed``。
   * - 关节 / DOF
     - ``jnt_margin``
     - 关节限位检测的距离阈值。
   * - 肌腱
     - ``tendon_range``
     - 肌腱长度限制。
   * -
     - ``tendon_margin``
     - 肌腱限位检测的距离阈值。
   * -
     - ``tendon_lengthspring``
     - 弹簧静息长度范围。
   * - 执行器
     - ``actuator_dynprm``、``actuator_gear``、
       ``actuator_ctrlrange``、``actuator_actrange``
     - ``pd_gains`` 与 ``effort_limits`` 已覆盖常见情况。

更适合自定义代码
""""""""""""""""

这些字段的语义相互耦合，通用 ``dr.*`` 函数反而容易误导。例如
``solref`` 的解释取决于求解器类型（椭圆 vs 直接），``solimp`` 有顺序
约束（dmin < dmax、width > 0），``qpos_spring`` 与 ``qpos0`` 耦合。
合适的范围取决于具体建模选择。请改为编写自定义事件项（见
:ref:`基于类的自定义事件项 <dr-custom-event-terms>`，或用
``@requires_model_fields`` 自动处理字段展开）。

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 类别
     - 字段
     - 备注
   * - 求解器参数
     - ``geom_solref``、``geom_solimp``、``geom_solmix``、
       ``jnt_solref``、``jnt_solimp``、``dof_solref``、
       ``dof_solimp``、``pair_solref``、``pair_solimp``、
       ``eq_solref``、``eq_solimp``
     - 语义依赖求解器类型与时间步长。
   * - 接触阈值
     - ``geom_margin``、``geom_gap``、``pair_margin``、``pair_gap``
     - 与上述求解器参数交互。
   * - 接触对覆盖
     - ``eq_data``
     - 约束锚点覆盖。
   * - 弹簧参考
     - ``qpos_spring``
     - 与 ``qpos0`` 耦合；独立随机化容易出错。

需要专门 API
""""""""""""

这些字段需要专门处理，因为它们是整数/类别型、涉及顶点数据，或者缺少
按环境独立取值所需的按 world 维度。

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 类别
     - 字段
     - 备注
   * - 网格
     - ``mesh_vert``、``mesh_normal``、``mesh_face`` 等
     - 操作物体的形状变化。这些字段缺少按 world 维度，当前的展开
       基础设施无法实现按 world 变化。异构 world 支持
       `正在开发中 <https://github.com/google-deepmind/mujoco_warp/pull/1009>`_。
   * - 可变形体
     - ``flex_*``
     - 软体操作的可变形体参数。与网格字段一样，大多数 ``flex_*``
       字段缺少按 world 维度。

不是 DR 目标
""""""""""""

- ``light_active``、``light_castshadow``、``light_type``、
  ``cam_projection``：布尔/整数开关，不是连续参数。
- ``jnt_pos``、``jnt_axis``：结构性关节几何；运行时修改很脆弱，也不是
  常见用例。
- ``hfield_data``、``hfield_size``：地形数据；请改用地形系统。


.. _dr-parameters:

参数
----

模型字段函数共享三个控制随机化的参数：``distribution`` 控制如何从
``ranges`` 采样，``operation`` 控制采样值如何施加到模型字段。
（``dr.pseudo_inertia`` 和 ``dr.pd_gains`` 等实体级函数有自己的签名；
细节见它们的 docstring。）

分布
^^^^

``distribution`` 参数控制如何从给定 ``ranges`` 采样随机值。它接受内置
字符串或自定义采样逻辑的 ``dr.Distribution`` 实例。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 取值
     - 行为
   * - ``"uniform"`` （默认）
     - 在 ``ranges[0]`` 与 ``ranges[1]`` 之间均匀采样
   * - ``"log_uniform"``
     - 在对数空间采样，适用于跨数量级的参数（如扭转摩擦）。两个范围
       端点都必须 > 0。
   * - ``"gaussian"``
     - ``ranges`` 被解释为 ``(mean, std)``

定义自定义分布请创建 ``dr.Distribution`` 实例。``sample`` 可调用对象
接收 ``(lower, upper, shape, device)`` 并返回张量。例如截断正态分布，
把采样截断到给定边界：

.. code-block:: python

    import torch
    from mjlab.envs.mdp import dr

    truncated_normal = dr.Distribution(
        name="truncated_normal",
        sample=lambda lo, hi, shape, device: torch.clamp(
            (lo + hi) / 2
            + (hi - lo) / 4  # 95% of samples within bounds
            * torch.randn(shape, device=device),
            min=lo,
            max=hi,
        ),
    )

    params={"distribution": truncated_normal, "ranges": (0.3, 1.2)}

.. warning::

   ``sample`` 接收的 ``lower``/``upper`` 是张量，因此采样张量必须从
   ``shape`` 构建（例如 ``torch.randn(shape, device=device)``）。把张量
   边界直接传给 ``torch.normal`` 之类的采样器会得到形状如同边界的
   张量而非 ``shape``，从而悄悄地让每个环境拿到相同的值。

操作
^^^^

``operation`` 参数控制采样值如何施加到模型字段。它接受内置字符串或
自定义逻辑的 ``dr.Operation`` 实例。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 取值
     - 行为
   * - ``"abs"`` （默认）
     - 把字段直接设为采样值
   * - ``"scale"``
     - 把原始默认值乘以采样值
   * - ``"add"``
     - 把采样值加到原始默认值上

对 ``"scale"`` 和 ``"add"``，DR 引擎始终把随机抽取施加到从编译的
``MjModel`` 在 CPU 上捕获的 **原始默认值** 上，而不是当前值。这避免了
累积：连续三次把摩擦乘 2，得到的是原值的 2 倍，而不是 8 倍。

定义自定义操作请创建 ``dr.Operation`` 实例。四个字段为：

- ``name``：用于错误消息的可读标签。
- ``initialize``：创建结果张量，随后逐轴填充采样值。例如 ``scale``
  从全 1 开始，未采样的轴乘 1（不变）；``add`` 从全 0 开始。
- ``combine``：接收 ``(base_values, random_values)`` 并返回写入模型
  字段的最终张量。例如 ``scale`` 返回 ``base * random``，``add``
  返回 ``base + random``。
- ``uses_defaults``：为 ``True`` 时，基准值是编译期默认值（避免重复
  调用间的累积）；为 ``False`` 时，基准值是当前模型值。

例如，内置 ``add`` 始终加到 *默认值* 上，重复调用是重置而不是漂移。
一个加到 *当前值* 上的自定义 ``drift`` 操作适用于 ``mode="interval"``
事件，让参数随时间缓慢游走：

.. code-block:: python

    import torch
    from mjlab.envs.mdp import dr

    drift = dr.Operation(
        name="drift",
        initialize=torch.zeros_like,
        combine=torch.add,
        uses_defaults=False,  # read current values, not defaults
    )

    # Friction slowly wanders each interval step.
    friction_drift: EventTermCfg = EventTermCfg(
        mode="interval",
        interval_range_s=(0.5, 1.0),
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=[".*_foot.*"]),
            "ranges": (-0.01, 0.01),
            "operation": drift,
        },
    )

轴选择
^^^^^^

许多模型字段是多维的。例如 ``geom_friction`` 有三个分量
``[tangential, torsional, rolling]``，``body_pos`` 有三个空间轴
``[x, y, z]``。可以用 ``axes`` 参数或传字典形式的 ``ranges`` 来指定
具体轴。

对 ``condim=3``（标准摩擦接触）的 ``geom_friction``，只有
**轴 0（切向）** 影响接触行为。condim 与摩擦系数的细节见 `MuJoCo 接触
文档
<https://mujoco.readthedocs.io/en/stable/computation/index.html#contact>`_。

.. code-block:: python

    # Tangential friction only (this is the default for geom_friction)
    params={"ranges": {0: (0.3, 1.2)}}

    # Tangential + torsional (torsional matters for condim >= 4)
    params={"ranges": {0: (0.5, 1.0), 1: (0.001, 0.01)}}

    # X and Y position with the same range
    params={"axes": [0, 1], "ranges": (-0.1, 0.1)}

按分量字符串键控的范围
^^^^^^^^^^^^^^^^^^^^^^

在单次调用中用正则模式作为字典键，可以对不同实体施加不同范围：

.. code-block:: python

    dr.joint_damping(
        env, env_ids,
        ranges={".*knee.*": (0.5, 1.5), ".*hip.*": (0.8, 1.2)},
        operation="scale",
        asset_cfg=SceneEntityCfg("robot", joint_names=[".*"]),
    )

每个模式与实体名称（关节名、geom 名等）匹配，对应的范围施加到匹配的
分量上。本例中膝关节的阻尼被缩放 0.5x-1.5x，而髋关节用更紧的 0.8x-1.2x
范围，全部在一个事件项内完成。


示例
----

摩擦（reset）
^^^^^^^^^^^^^

.. code-block:: python

    foot_friction: EventTermCfg = EventTermCfg(
        mode="reset",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=[".*_foot.*"]),
            "ranges": (0.3, 1.2),
            "operation": "abs",
        },
    )

.. note::

     把机器人的碰撞 geom 的 **priority** 设得比地形高（geom priority
     默认为 0）。这样只有机器人一侧的摩擦生效。MuJoCo 会在
     （机器人, 地形）接触中使用优先级更高的 geom 的摩擦。

.. code-block:: python

    from mjlab.utils.spec_config import CollisionCfg

    robot_collision = CollisionCfg(
        geom_names_expr=[".*_foot.*"],
        contype=1,
        conaffinity=1,
        condim=3,
        priority=1,
        friction=(0.6,),
    )


关节偏置（startup）
^^^^^^^^^^^^^^^^^^^

随机化默认关节位置，模拟关节偏置标定误差：

.. code-block:: python

    joint_offset: EventTermCfg = EventTermCfg(
        mode="startup",
        func=dr.joint_default_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "ranges": (-0.01, 0.01),
            "operation": "add",
        },
    )


质心（startup）
^^^^^^^^^^^^^^^

.. code-block:: python

    com: EventTermCfg = EventTermCfg(
        mode="startup",
        func=dr.body_com_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso"]),
            "ranges": {0: (-0.02, 0.02), 1: (-0.02, 0.02)},
            "operation": "add",
        },
    )


常见陷阱
--------

``dr.body_mass`` 不缩放惯量
^^^^^^^^^^^^^^^^^^^^^^^^^^^

只缩放质量而不缩放惯量，仅在建模加在质心上的点质量（转动惯量贡献为零）
时物理正确。常见的 DR 用例——模拟制造误差或连杆密度的不确定性——应该
让质量与惯量一起缩放。请改用带 ``alpha_range`` 的
:func:`dr.pseudo_inertia`：

.. code-block:: python

    # Wrong: mass changes, inertia stays fixed (physically inconsistent).
    EventTermCfg(func=dr.body_mass, params={"ranges": (0.8, 1.2)})

    # Correct: mass and inertia both scale by e^{2alpha} (uniform density change).
    EventTermCfg(func=dr.pseudo_inertia, params={"alpha_range": (-0.1, 0.1)})

``dr.body_mass`` 在运行时发出 ``UserWarning`` 提示这一点。

``*_quat`` 范围以弧度为单位，不是度
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

所有四元数随机化函数（:func:`dr.geom_quat`、:func:`dr.body_quat`、
:func:`dr.site_quat`、:func:`dr.cam_quat`）接受的 roll/pitch/yaw 范围
以 **弧度** 为单位。传入角度值会悄悄产生约 57 倍于预期的旋转。没有
运行时检查。

``*_quat`` 扰动相对默认值，而不是当前值
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

采样的 RPY 扰动与 **默认** 四元数复合，而不是当前四元数。这与所有其他
``dr`` 函数的无累积保证一致，但也意味着重复调用不会叠加旋转。每次调用
都独立地从默认姿态采样新的扰动。

``dr.geom_friction`` 默认只随机化切向摩擦
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

默认轴是 0（切向摩擦）。轴 1（扭转）和轴 2（滚动）只影响
``condim >= 4`` 的接触，因此对标准 ``condim=3`` 接触，默认值是正确的。
如果模型使用高维接触，请显式传 ``axes=[0, 1, 2]``。

``dr.geom_size`` 对非图元 geom 抛异常
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`dr.geom_size` 只支持图元 geom 类型（球、胶囊、椭球、圆柱、盒），
因为网格、平面和高度场 geom 的宽相包围盒无法解析重算。选中非图元
geom 会抛出 ``ValueError``。请把 ``SceneEntityCfg`` 过滤到仅含图元
geom：

.. code-block:: python

    # Select only primitive geoms by name pattern.
    geom_cfg = SceneEntityCfg("robot", geom_names=(".*sphere.*", ".*box.*"))

底层实现原理
------------

理解内部机制有助于编写自定义 DR 项或调试意外行为。

按 world 存储
^^^^^^^^^^^^^

MuJoCo Warp 把数千个 world 批处理进单次仿真。为节省内存，
``geom_friction`` 之类的模型数组只存储一份，形状 ``(1, ngeom, 3)``，
沿第一维（world 维）的 **stride 为 0**。GPU kernel 用
``worldid % arr.shape[0]`` 索引这些数组。当 ``shape[0]`` 为 1 时，每个
world 读到同一行，因此它们共享相同的模型参数。

在 PyTorch 侧，mjlab 用 ``torch.expand`` 包装这些 stride-0 数组，使其
看起来形状为 ``(num_envs, ngeom, 3)``，而底层仍只有一行内存。
``tensor[env_id]`` 这样的索引让人以为每个 world 有自己的数据，但对任何
world 的写入会影响所有 world，因为它们指向同一块底层内存。

要让每个 world 拥有独立的值，底层 Warp 数组需要从形状 ``(1, N)``
**展开** 为 ``(num_worlds, N)``，配备真实的按 world 内存与正常 stride。
``sim.expand_model_fields()`` 分配新数组、把共享数据拷贝进每个 world
的行、并把旧数组替换掉。展开之后，对一个 world 的写入不再影响其他
world，每个 world 可以有自己的摩擦、质量或阻尼。

每个 ``dr`` 函数通过 ``@requires_model_fields`` 装饰器声明自己需要哪些
字段，``EventManager`` 在启动时收集这些声明，从而自动完成展开。直接
修改模型数组的自定义 DR 项必须确保相关数组已经展开：要么用
``@requires_model_fields`` 装饰函数，要么手动调用
``sim.expand_model_fields()``。

.. note::

   展开字段会分配新的 GPU 显存，并使任何已捕获的 CUDA graph 失效，
   因为 graph 持有指向旧数组的指针。mjlab 会在展开后自动重建 graph。
   这是环境启动时的一次性开销，不是每回合的。

为什么不直接重新编译模型？
^^^^^^^^^^^^^^^^^^^^^^^^^^

获得按 world 差异最干净的方式是：为每个 world 修改 ``MjSpec``、分别
编译成各自的 ``MjModel``，再全部传输到 GPU。这样派生量天然完全一致，
因为 ``mj_setConst`` 在编译期间运行。

mjlab 没有这么做，原因有二：

1. **开销。** 编译 ``MjSpec`` 是 CPU 操作。数千环境下每次回合重置都对
   每个 world 编译一次，慢得无法接受。

2. **架构。** MuJoCo Warp 期望所有 world 共享单个 ``Model``。没有把
   N 个独立模型加载进同一次仿真的机制。

mjlab 的替代方案是：在 GPU 上就地修改展开后的数组，并只选择性地重算
依赖于已改字段的派生量。这正是 ``RecomputeLevel`` 系统负责的部分。

.. _dr-recomputation:

派生字段的重算
^^^^^^^^^^^^^^

一些模型字段是 **派生** 自其他字段的。例如 ``body_subtreemass``
（某 body 及其全部后代的总质量）依赖 ``body_mass``。改了
``body_mass`` 却不更新 ``body_subtreemass``，约束求解器就会用过期的
阻抗值，仿真会出现细微错误。

在 C MuJoCo 中，解决方案是运行时修改模型参数后调用
`mj_setConst
<https://mujoco.readthedocs.io/en/latest/programming/simulation.html#mjmodel-changes>`_。
MuJoCo Warp 提供了一组等价函数（``set_const``、``set_const_0``、
``set_const_fixed``），在 GPU 上运行并对所有 world 并行。mjlab 自动
调用它们。

三个重算等级，从最便宜到最贵：

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 等级
     - 重算内容
     - 使用时机
   * - ``set_const_fixed``
     - ``body_subtreemass``
     - 修改 ``body_gravcomp`` 之后
   * - ``set_const_0``
     - ``dof_invweight0``、``body_invweight0``、``tendon_length0``、
       ``tendon_invweight0``、``actuator_acc0``，外加相机与光源参考
     - 修改 ``dof_armature``、``tendon_armature``、``body_inertia``、
       ``body_pos``、``body_quat`` 或 ``qpos0`` 之后
   * - ``set_const``
     - 以上全部
     - 修改 ``body_mass`` 或 ``body_ipos`` （质心）之后

内置 ``dr`` 函数已经声明了正确的等级。当 ``EventManager`` 在一次
``apply()`` 调用中触发多个 DR 项时，它会跟踪其中最强的等级，并在末尾
调用一次 ``sim.recompute_constants()``。除非编写自定义 DR 逻辑，无需
手动调用。

只直接影响接触或关节行为的字段（``geom_friction``、``dof_damping``、
``dof_frictionloss`` 等）没有派生量，无需任何重算。

.. note::

   mjlab 把 ``sim.step()``、``sim.forward()``、``sim.reset()`` 和
   ``sim.sense()`` 分别捕获为独立的 CUDA graph 以提升性能。事件管理器的
   全部逻辑（包括 ``recompute_constants``）在这些 graph 重放之间以普通
   Python 运行，因此不会破坏 graph 捕获。

   不过 ``set_const`` 相当昂贵：它要对所有 world 运行正向运动学、
   复合刚体算法和质量矩阵分解。用 ``interval`` 事件每步调用会带来显著
   开销。实践中，需要重算的字段（``body_mass``、``body_com_offset``、
   ``joint_armature`` 等）最好用 ``startup`` 或 ``reset`` 模式随机化；
   无需重算的字段（``geom_friction``、``dof_damping`` 等）以任意频率
   随机化都很便宜。


.. _dr-custom-event-terms:

基于类的自定义事件项
--------------------

自定义事件项也可以用类而不是函数。这适用于需要维护状态或执行初始化
逻辑的事件项：

.. code-block:: python

    class RandomizeTerrainFriction:
        """Custom event term that randomizes terrain friction."""

        def __init__(self, cfg, env):
            # Find the terrain geom index during initialization
            self._terrain_idx = None
            for idx, geom in enumerate(env.scene.spec.geoms):
                if geom.name == "terrain":
                    self._terrain_idx = idx

            if self._terrain_idx is None:
                raise ValueError("Terrain geom not found in the model.")

        def __call__(self, env, env_ids, ranges):
            """Called each time the event is triggered."""
            from mjlab.utils.math import sample_uniform
            env.sim.model.geom_friction[env_ids, self._terrain_idx, 0] = (
                sample_uniform(ranges[0], ranges[1], len(env_ids), env.device)
            )


    # Register in the environment config.
    terrain_friction: EventTermCfg = EventTermCfg(
        mode="reset",
        func=RandomizeTerrainFriction,
        params={"ranges": (0.3, 1.2)},
    )


可视化 DR 变化
--------------

两个查看器都能反映 DR 变化，但覆盖面不同。

**原生查看器**

原生查看器在每次渲染前把按 world 的模型字段从 GPU 同步到本地
``MjModel``。MuJoCo 内置的所有可视化开关随后都能在随机化后的模型上
正确工作：

- geom 外观（``geom_rgba``、``geom_size``、``geom_pos``、
  ``geom_quat``、``geom_matid``）
- 材质外观（``mat_rgba``、``mat_emission``、``mat_specular``、
  ``mat_shininess``、``mat_texrepeat``、``mat_texid``）
- body 与 site 位姿（``body_pos``、``body_quat``、``body_ipos``、
  ``site_pos``、``site_quat``）
- 惯量（``body_inertia``、``body_iquat``、``body_mass``）：按 ``I``
  切换惯量盒显示
- 相机参数（``cam_pos``、``cam_quat``、``cam_fovy``、
  ``cam_intrinsic``）：按 ``Q`` 切换相机视锥显示
- 光源（``light_pos``、``light_dir``、``light_diffuse``、
  ``light_specular``、``light_ambient``、``light_attenuation``、
  ``light_cutoff``、``light_exponent``）

.. grid:: 2

   .. grid-item-card::

      .. image:: ../../source/_static/dr_combined_rand.gif
         :alt: Cube color, size, and link orientations randomized each reset

      立方体颜色（``dr.geom_rgba``）、立方体尺寸（``dr.geom_size``）
      与连杆 2/3 姿态（``dr.body_quat``）在每次回合重置时随机化。尺寸
      变化后宽相包围盒自动重算。

   .. grid-item-card::

      .. image:: ../../source/_static/dr_pseudo_inertia.gif
         :alt: Inertia ellipsoids resizing each episode reset

      对连杆 2 和 3 施加 ``alpha_range=(-0.5, 0.5)`` 的
      ``dr.pseudo_inertia``。惯量椭球在每次回合重置时缩放，其他连杆
      保持不变。

.. note::

   ``cam_fovy`` 对使用内参参数（XML 中设置 ``sensorsize`` /
   ``focal``）的相机没有效果，渲染图像和视锥可视化都如此。MuJoCo 改为
   从 ``cam_intrinsic`` 和 ``cam_sensorsize`` 计算投影。对这类相机请用
   ``dr.cam_intrinsic`` 随机化视场。内参参数与 ``fovy`` 的交互细节见
   `MuJoCo 相机文档
   <https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera>`_。

**Viser**

相机视锥与 body 位姿始终是最新的，因为 viser 每帧直接从 GPU 仿真数据
（``cam_xpos``、``body_xpos``）读取世界坐标。


.. note::

   ``geom_rgba``、``geom_size`` 和 ``mat_texid`` 的 DR **不会** 反映到
   viser。geom 颜色、尺寸和纹理在构建时烘焙进场景的 GLB 网格。底层
   viser API（``add_batched_meshes_simple``）通过 ``batched_colors``
   支持逐实例颜色更新，但这要求把仅改颜色的 geom 走与当前
   ``add_batched_meshes_trimesh`` 不同的句柄路径。已推迟到未来更新。


从 Isaac Lab 迁移
-----------------

Isaac Lab 暴露显式的摩擦组合模式（``multiply``、``average``、``min``、
``max``）。MuJoCo 改用 **基于优先级的选择**：若其中一个接触 geom 的
``priority`` 更高，则使用它的摩擦；否则取 **逐元素最大值**。细节见
`MuJoCo 接触文档 <https://mujoco.readthedocs.io/en/stable/computation/index.html#contact>`_。
