.. _heterogeneous_worlds:

异构 world
==========

mjlab 可以在单个批处理仿真中让不同的并行 world 为同一个逻辑实体使用
不同的网格资产：world 0 仿真一个立方体，world 1 一个球体，world 2 一个
碗。所有 world 共享同一份编译后的场景和相同的 body、关节结构；只有
网格以及随网格携带的逐 geom 属性（摩擦、接触位、质量、密度等少数几项）
在不同 world 之间不同。铰接道具同样支持（变体的根之下可以有铰链或滑动
关节），只要关节拓扑在变体间保持一致。该功能通过 ``VariantEntityCfg``
暴露。哪些内容可以在变体间变化、哪些不能，下一节给出完整清单。


快速上手
--------

假设你希望一部分并行 world 放球体、另一部分放圆锥，且同一个场景同时
运行两者。把每个变体定义为返回 ``MjSpec`` 的函数，然后在同一个
``VariantEntityCfg`` 下分组：

.. code-block:: python

    import mujoco

    from mjlab.entity import EntityCfg, VariantEntityCfg


    def make_sphere_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        mesh = spec.add_mesh(name="visual")
        mesh.make_sphere(subdivision=3)
        mesh.scale[:] = (0.05,) * 3
        body = spec.worldbody.add_body(name="prop")
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="visual")
        return spec


    def make_cone_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        mesh = spec.add_mesh(name="visual")
        mesh.make_cone(nedge=16, radius=0.04)
        body = spec.worldbody.add_body(name="prop")
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="visual")
        return spec


    object_cfg = VariantEntityCfg(
        variants={
            "sphere": make_sphere_spec,
            "cone":   make_cone_spec,
        },
        assignment={"cone": 2.0},  # twice as many cones as spheres
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
    )

把变体实体接入 :ref:`scene` 的方式与普通 ``EntityCfg`` 完全一样：

.. code-block:: python

    from mjlab.scene import SceneCfg

    scene_cfg = SceneCfg(
        num_envs=4096,
        entities={"object": object_cfg},
    )

放圆锥的 world 数量将是放球体的两倍。未出现在 ``assignment`` 字典中的
变体默认权重为 1.0；完全省略 ``assignment`` 则在所有变体间均匀分配。


变体之间可以差异化的内容
------------------------

**可以自由变化：** 分配给每个槽位的网格资产；变体 body 上每个
``(body, role)`` 桶的网格 geom 数量（一个变体可以比另一个拥有更多碰撞
网格）；随网格携带的逐网格 geom 属性（摩擦、接触位、质量、密度、
``condim`` 等少数几项）；以及在各变体按 body 达成一致的单一惯量模式下
显式设置的 body 惯量值。

**必须在变体间一致：** body 树、关节拓扑、图元（非网格）geom，以及
所有执行器 / 传感器 / 肌腱 / 等式约束。各变体还必须对每个 body 采用
相同的惯量表示（网格推导、对角或 fullinertia），并且不得在任何元素上
使用保留的 ``mjlab/pad/`` 名称前缀。变体实体还必须是浮动基座：根 body
声明 freejoint。

校验器在实体构建时运行，抛出 ``ValueError`` 并指明出问题的变体与具体
不一致之处。


变体如何组装
------------

mjlab 把所有变体的网格资产合并进单个 ``MjSpec``，并为变体 body 分配
足够多的网格 geom *槽位*，以覆盖任何变体在每个 ``(body, role)`` 桶上
用到的最大网格数量。槽位由 ``(body_path, role, ordinal)`` 标识。
``role`` 是 "visual" 或 "collision"，从 ``contype``/``conaffinity``
推导而来；mujoco_warp 的 ``geom_contype``/``geom_conaffinity`` 是一维
共享数组（非按 world），因此槽位的 role 在构造上就跨 world 固定。

一个具体例子
~~~~~~~~~~~~

设变体 ``sphere`` 在 prop body 上有 1 个视觉网格 geom 和 2 个碰撞网格
geom，变体 ``cone`` 在同一 body 上有 1 个视觉网格 geom 和 4 个碰撞网格
geom。

.. code-block:: text

    sphere variant body              cone variant body
    -------------------              -------------------
    prop body                        prop body
      [visual] sphere_vis              [visual] cone_vis
      [coll]   sphere_col_0            [coll]   cone_col_0
      [coll]   sphere_col_1            [coll]   cone_col_1
                                       [coll]   cone_col_2
                                       [coll]   cone_col_3

mjlab 遍历每个变体的 body 树，按 ``(body_path, role)`` 对网格 geom
分桶，并把并集铺成槽位：

.. list-table::
   :header-rows: 1
   :widths: 8 18 8 12 27 27

   * - 槽位
     - body_path
     - role
     - ordinal
     - sphere 填充
     - cone 填充
   * - 0
     - /prop
     - visual
     - 0
     - sphere_vis
     - cone_vis
   * - 1
     - /prop
     - collision
     - 0
     - sphere_col_0
     - cone_col_0
   * - 2
     - /prop
     - collision
     - 1
     - sphere_col_1
     - cone_col_1
   * - 3
     - /prop
     - collision
     - 2
     - *(未填充)*
     - cone_col_2
   * - 4
     - /prop
     - collision
     - 3
     - *(未填充)*
     - cone_col_3

共 5 个槽位。合并后场景的 prop body 拥有 5 个网格 geom：槽位 0 加上
4 个碰撞槽位（sphere 的 2 个与 cone 的 4 个的并集）。合并时，每个
变体的网格资产以唯一名称加入合并后的 spec（如 ``sphere/sphere_vis``、
``cone/cone_col_2``）。

合并场景一次性编译为单一规范 ``MjModel``，批内所有 world 在布局上完全
一致：相同的 nbody、ngeom、相同的 body 与 geom ID。mjlab 在这一份模型
之上施加按 world 的覆盖，从而让 world 之间异构。

每个 world 在运行时看到什么
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sphere`` 生效的 world 只看到它的 3 个网格；多出的 2 个碰撞槽位通过
按 world 的 ``geom_dataid = -1`` 禁用，mujoco_warp 会跳过它们。
``cone`` 生效的 world 则看到全部 5 个网格都被接线。

.. list-table::
   :header-rows: 1
   :widths: 14 14 12 12 12 12 12

   * - World
     - 变体
     - slot 0
     - slot 1
     - slot 2
     - slot 3
     - slot 4
   * - 0
     - sphere
     - sphere_vis
     - sphere_col_0
     - sphere_col_1
     - **关闭 (-1)**
     - **关闭 (-1)**
   * - 1
     - cone
     - cone_vis
     - cone_col_0
     - cone_col_1
     - cone_col_2
     - cone_col_3

差异由三类按 world 的覆盖承载：

* **geom_dataid** 是一张 ``(num_envs, ngeom)`` 表。world W 对应的行
  决定每个槽位指向哪个已编译网格。``-1`` 是 mujoco_warp 原生理解的
  "跳过我" 哨兵值。
* **网格派生字段** （``geom_size``、``geom_rbound``、``geom_aabb``、
  ``geom_pos``、``geom_quat``、``body_mass``、``body_subtreemass``、
  ``body_inertia``、``body_invweight0``、``body_ipos``、
  ``body_iquat`` ）以 ``(num_envs, ...)`` 数组存储。sphere world 的取值
  反映球形的惯量张量和球形的 AABB；cone world 的取值反映圆锥。完整
  清单见 ``mjlab.entity.variants.VARIANT_DEPENDENT_FIELDS``。
* **逐网格 geom 属性** （接触位、摩擦、质量、密度、condim、group、
  priority、rgba、solref、solimp、margin、gap）在合并时按变体捕获到
  ``VariantGeomSpec`` 中，并在逐变体参考编译期间原样恢复到槽位 geom
  上。因此如果 sphere 的碰撞 geom 是 ``friction=0.5``、cone 的是
  ``friction=1.2``，world W 每步使用的摩擦就是其所分配变体的源值。
  唯一的例外是 ``material``——它不在变体间传播；如果需要按 world 的
  外观差异，请对 ``geom_rgba`` / ``mat_rgba`` 做 DR。

如果 ``sphere`` 拥有 ``cone`` 没有的 body（或反之），校验器会在任何
合并逻辑运行前拒绝该配置。槽位机制只在结构匹配的 body 内部弹性伸缩
网格 geom 数量；geom 层级之上的结构性内容必须完全一致。

.. note::

   **合并场景的编译不会毁掉 prop body 的惯量吗？**

   不会，但值得理解为什么——朴素的直觉认为会。如果把每个变体的网格
   geom 都挂到 prop body 上再调用 ``spec.compile()``，MuJoCo 会把每个
   geom 的惯量贡献求和，得到一个质量与惯量张量混杂了所有变体形状的
   body，毫无意义。

   mjlab 用两层机制避免了这一点：

   * **合并场景不会把每个变体的 geom 都挂到 body 上。** 合并 spec 中的
     prop body 只携带变体 0 的网格 geom（连同其原始质量和密度），外加
     为变体 0 未填充的槽位合成的填充 geom，后者 ``mass = 0``、
     ``density = 0``，对 body 惯量没有任何贡献。其他变体的网格在合并
     spec 中只以 **网格资产** 形式存在（在 assets 一节，而不是作为任何
     body 上的 geom）。它们在运行时经按 world 的 ``geom_dataid``
     接线，从不影响宿主编译的惯量求和。
   * **按 world 的覆盖来自逐变体的源编译。** 即便有上述机制，合并场景
     编译出的 prop body 惯量也只对变体 0 正确。对其余每个变体，mjlab
     会单独编译该变体的原始源 spec（一个 body、该变体的网格），读取
     结果中的 ``body_mass``、``body_inertia``、``body_ipos``、
     ``body_iquat``、``body_invweight0`` 和 ``body_subtreemass``，并
     写入按 world 数组中 prop body 的索引位置。

   最终结果：world W 的 prop body 惯量与单独编译变体 W 的源 spec 得到的
   结果逐字节相等。仓库里有一个回归测试
   （``tests/test_variants.py`` 中的
   ``test_visual_collision_split_inertia_matches_independent_compile``）
   专门对照独立的逐变体编译来断言这一点。


world 分配
----------

world 如何映射到变体由 ``VariantEntityCfg`` 的 ``assignment`` 字段控制，
它接受三种形式：

* ``None`` （默认）：在变体间均匀分配。
* ``dict[str, float]``：逐变体权重。未列出的变体默认权重 1.0。
* ``Callable[[int], Sequence[int]]``：显式分配函数，在仿真初始化时以
  ``num_envs`` 调用。

``None`` 与字典两种情况都使用
`最大余数法
<https://en.wikipedia.org/wiki/Largest_remainder_method>`_。每个变体的
配额为 ``q_i = (w_i / sum(w)) * num_envs``；每个变体先获得
``floor(q_i)`` 个 world，剩余的 ``num_envs - sum(floors)`` 个 world 按
小数余数从大到小分配，平局时按声明顺序。对 ``num_envs = 10``、权重
``(1.0, 2.0, 1.0)``，各变体分得 ``(3, 5, 2)`` 个 world。权重会在内部
归一化，因此 ``{"a": 1, "b": 2, "c": 1}`` 与
``{"a": 0.25, "b": 0.5, "c": 0.25}`` 产生完全相同的分配。允许权重为零，
该变体将分得 0 个 world；但至少要有一个变体的权重为正。

在给定 ``(assignment, num_envs)`` 的情况下，默认路径与字典路径都是纯
确定性的。取 ``assignment={"a": 1, "b": 1}`` 和 ``num_envs = 8``，你
永远得到 ``[0, 0, 0, 0, 1, 1, 1, 1]``。没有任何随机种子参与；重跑同一
配置每次都得到相同的划分。注意划分的 *边界* 依赖 ``num_envs``，因此
改变 ``num_envs`` 后 world W 的变体未必保持不变。如果你需要跨批大小
的显式按 world 稳定性（例如"不管我启动多少环境，world 0 永远是变体
0、world 1 永远是变体 1"），请使用下面的可调用分配。

变体分配在 ``Simulation`` 初始化时固定，回合重置时不会重新采样。其
设计用途是批内的异构训练，而不是逐回合的网格随机化。

在用户代码中通过 ``env.sim.world_to_variant`` 读取解析后的分配：

.. code-block:: python

    >>> env.sim.world_to_variant["object"]
    tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

该映射以实体名（不带末尾斜杠）为键，返回按 ``VariantEntityCfg.variants``
中变体声明顺序索引的 ``(num_envs,)`` 张量。非变体场景中该字典为空。


用可调用对象自定义分配
~~~~~~~~~~~~~~~~~~~~~~

当加权默认值不满足需求时，可以给 ``assignment`` 传入可调用对象。它
接收 ``num_envs``，必须返回长度为 ``num_envs``、取值在
``[0, len(variants))`` 的变体索引序列。返回序列的长度与取值范围会在
仿真初始化时校验，不一致时抛出 ``ValueError`` 并指明出问题的实体。

几种常见模式：

**轮转（round-robin）** - 按 world 索引在变体间循环。

.. code-block:: python

    cfg = VariantEntityCfg(
        variants={"a": make_a, "b": make_b, "c": make_c},
        assignment=lambda n: [w % 3 for w in range(n)],
    )

**分层对半** - 前一半是变体 0，后一半是变体 1。

.. code-block:: python

    cfg = VariantEntityCfg(
        variants={"easy": make_easy, "hard": make_hard},
        assignment=lambda n: [0] * (n // 2) + [1] * (n - n // 2),
    )

域随机化
--------

变体场景上的域随机化会自动保留各变体的基线。仿真初始化时，mjlab 把
依赖变体的字段快照为 ``(num_envs, ...)`` 张量并注册到
``sim.per_world_default_fields``。读取默认值的 DR 操作（缩放、加性
偏移）会检测到这一注册并按环境索引按 world 的默认值数组，因此在一个
同时包含 100 g 球体变体和 1 kg 立方体变体的批次上施加 10% 的质量缩放，
得到的是 *围绕每个变体自身质量* 的 10% 扰动，而不是共享模板质量的
10%。

不依赖变体的字段（``geom_friction``、``dof_armature``、
``dof_damping`` 等）在变体与非变体场景上的行为完全相同。

对惯量随机化，推荐使用 ``dr.pseudo_inertia``：它通过 `Rucker 和
Wensing (2022)
<https://par.nsf.gov/servlets/purl/10347458>`_ 的伪惯量矩阵分解，联合
随机化质量、质心偏移、主惯量矩和主轴方向。它对任意扰动幅度都是精确的，
并且在尺寸悬殊的不同变体之间保持物理一致。``dr.body_mass`` 只修改
``body_mass`` 而不动惯量张量，调用时会发出 ``UserWarning``；它只适合
建模加在质心上的点质量，不适合密度类随机化。这一区分在变体场景上比
单一资产场景更重要，因为变体之间的质量常常相差一个数量级。


查看器
------

原生查看器、离屏渲染器和 Viser 查看器都会在渲染前把所选环境的按 world
字段同步到宿主 ``MjModel``，因此渲染出的几何体与被查看环境分配到的
变体一致。在原生查看器中切换环境（``,`` 和 ``.`` 键）会相应更新显示的
网格。

Viser 把网格数据烘焙进批量句柄，无法依赖对 ``geom_dataid`` 的实时
视图。它按视觉指纹（网格选择、局部 geom 坐标系、烘焙后的外观）对
world 分组，为每组构建一个批量句柄，并把每个环境指派到所属句柄。含
N 个变体的场景通常每个 body 最多产生 N 个句柄。凸包可视化按变体从该
变体的网格顶点计算。


性能
----

**每步开销与变体数量无关。** 依赖变体的字段以按 world 数组的形式存储，
在既有内核中按 world 索引访问，没有任何按变体的分支或分发。

**构建开销与变体总数线性相关。** mjlab 先把合并场景编译一次得到规范
``MjModel``，然后单独编译每个变体的原始（未合并）源 spec，以恢复该
变体逐 body、逐 geom 的网格派生字段。每次逐变体编译只看到该变体自己的
单个 body 和网格，因此其开销与场景中变体总数无关。

对声明了 k 个变体的单个变体实体，构建需要 ``1 + k`` 次编译。多个变体
实体时，编译在各实体间解耦：两个各含 5 个变体的实体开销是
``1 + 5 + 5 = 11`` 次编译，而不是 ``1 + 5 * 5 = 26`` 次。作为数量级
参考，在 CPU 上编译典型的程序化网格时，每次逐变体编译约需 1-2 ms，
因此 100 个变体的场景启动要多付几百毫秒，1000 个变体大约两秒。

合并 spec 同时包含所有变体的网格资产，因此场景构建期的内存随所有变体
的网格顶点/面总数增长。这是一次性的启动开销，不影响训练吞吐。


限制
----

**仅支持浮动基座。** 每个变体的根 body 必须声明自由关节。固定基座
变体会被拒绝；适用于非变体实体的 mocap 自动包装在这里不生效。

**材质资产不传播。** 每个变体的 ``contype``、``conaffinity``、
``condim``、``friction``、``mass``、``density``、``group``、
``priority``、``rgba``、``solref``、``solimp``、``margin`` 和 ``gap``
会在编译期间按 world 恢复，但槽位 geom 上的 ``material`` 引用继承的
是模板变体设置的材质。按 world 的外观差异请对 ``geom_rgba`` /
``mat_rgba`` 做 DR。

**分配在仿真初始化时固定。** 没有提供在回合重置时把某个 world 换到
另一个变体的 API。world W 的网格资产在整个仿真生命周期内保持初始化时
的指派。目前不支持逐回合网格随机化；DR 可以在固定变体上随机化标量
属性（质量、摩擦、颜色、缩放），但无法把一个网格换成另一个。

**不支持各 world 运动学拓扑不同。** 变体必须共享相同的 body 树、关节
以及执行器/传感器数量，因此无法配置诸如：

* 每个 world 的物体数量不同（world 0 的桌上有两个道具，world 1 有
  三个）；
* 每个 world 的铰接方式不同（world 0 的道具是带滑动关节的铰接抽屉，
  world 1 的道具是刚性方块）。

真正的异构拓扑需要 mujoco_warp 的上游支持，目前尚不存在。
