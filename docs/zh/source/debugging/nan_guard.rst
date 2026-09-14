.. _nan-guard:

NaN 守卫
========

NaN 守卫在检测到 NaN/Inf 时捕获仿真状态，帮助调试数值不稳定问题。


快速上手
--------

用一个 CLI 标志启用 NaN 守卫：

.. code-block:: bash

    uv run train <task-name> --enable-nan-guard True

检测到 NaN/Inf 时会自动捕获并保存仿真状态。也可以以编程方式启用：

.. code-block:: python

    from mjlab.sim.sim import SimulationCfg
    from mjlab.utils.nan_guard import NanGuardCfg

    cfg = SimulationCfg(
        nan_guard=NanGuardCfg(
            enabled=True,
            buffer_size=100,
            output_dir="/tmp/mjlab/nan_dumps",
            max_envs_to_dump=5,
        ),
    )


配置
----

``enabled`` *（默认 False）*
    启用/禁用 NaN 检测与转储。

``buffer_size`` *（默认 100）*
    滚动缓冲区中保留的最近仿真状态数量。

``output_dir`` *（默认 "/tmp/mjlab/nan_dumps"）*
    NaN 转储文件的保存目录。

``max_envs_to_dump`` *（默认 5）*
    转储到磁盘的 NaN 环境数量上限。所有环境都会进入缓冲区，但只有前
    N 个被保存，以控制转储体积。


行为
----

- **捕获** 每步之前的仿真状态（``qpos``、``qvel``，模型带执行器激活时
  还有 ``act``，带 mocap body 时还有 ``mocap_pos``/``mocap_quat``）
- **检测** 每步之后 ``qpos``、``qvel``、``qacc``、``qacc_warmstart``
  与 ``sensordata`` 中的 NaN/Inf
- **转储** 首次检测时把滚动缓冲区和模型写入磁盘
- **停止** 首次转储后停止，避免刷屏

禁用时所有操作均为空操作，开销可忽略。


输出格式
--------

每次 NaN 检测会生成带时间戳的文件以及指向最新文件的符号链接：

- ``nan_dump_TIMESTAMP.npz``：压缩的状态缓冲区

  - ``states_step_NNNNNN``：每步捕获的状态
    （形状 ``[num_envs_dumped, state_size]``）
  - ``_metadata``：包含 ``num_envs_total``、``nan_env_ids``、
    ``dumped_env_ids`` 等的字典

- ``model_TIMESTAMP.mjb``：二进制格式的 MuJoCo 模型
- ``nan_dump_latest.npz``：指向最新转储的符号链接
- ``model_latest.mjb``：指向最新模型的符号链接


可视化转储
----------

用交互式查看器逐帧浏览捕获的状态：

.. code-block:: bash

    # View latest dump.
    uv run viz-nan /tmp/mjlab/nan_dumps/nan_dump_latest.npz

    # View a specific dump.
    uv run viz-nan /tmp/mjlab/nan_dumps/nan_dump_20251014_123456.npz


.. figure:: ../../../source/_static/content/nan_debug.gif
   :alt: NaN Debug Viewer

   NaN 调试查看器。

查看器提供：

- 步进滑杆，逐帧浏览缓冲区
- 环境滑杆，对比不同环境
- 信息面板，显示哪些环境出现 NaN/Inf
- 每个状态下机器人与地形的 3D 可视化


NaN 检测终止
------------

NaN 守卫通过捕获状态帮助 **调试** NaN 问题；你还可以用 ``nan_detection``
终止项 **防止** 训练崩溃。它把出现 NaN 的环境标记为终止，使其重置而
训练继续：

.. code-block:: python

    from mjlab.envs.mdp.terminations import nan_detection
    from mjlab.managers.termination_manager import TerminationTermCfg

    nan_term: TerminationTermCfg = field(
        default_factory=lambda: TerminationTermCfg(
            func=nan_detection,
            time_out=False,
        )
    )

终止会以 ``Episode_Termination/nan_term`` 记录在指标中。

.. important::

   ``nan_detection`` 是创可贴，不是药方。如果 NaN 出现在任务目标本身
   （例如抓取时出现 NaN），策略永远学不会完成任务——它会在拿到奖励
   之前就被重置。请密切监控 ``Episode_Termination/nan_term`` 指标。

**如何选择：**

- ``nan_guard``：调试并弄清 NaN 为何出现（永远先做这件事）
- ``nan_detection``：在彻底修复完成之前保持训练稳定
