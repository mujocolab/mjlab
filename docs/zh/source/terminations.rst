.. _terminations:

终止
====

终止项定义回合何时结束。每项都是一个每步返回按环境布尔张量的函数。
终止管理器聚合所有项，并把结果以"终止失败"或"截断"之一报告给训练
框架。

每项通过 ``TerminationTermCfg`` 按名称注册。设置 ``time_out=True``
会把该条件标记为截断而非终止失败。截断映射到 Gym 接口的 ``truncated``
信号；失败映射到 ``terminated``。这一区分对价值自举很重要：智能体应当
在截断之外估计未来价值，而在失败之外不做估计。

.. code-block:: python

    from mjlab.envs.mdp import terminations
    from mjlab.managers.termination_manager import TerminationTermCfg

    terminations_cfg = {
        "time_out": TerminationTermCfg(
            func=terminations.time_out, time_out=True,
        ),
        "fallen": TerminationTermCfg(
            func=terminations.bad_orientation,
            params={"limit_angle": 1.0},
        ),
    }


内置终止函数
------------

下列函数位于 ``mjlab.envs.mdp.terminations``，在所有任务间共享。各任务
可以针对自身目标定义额外的终止函数。所有终止函数都返回形状为
``[num_envs]`` 的布尔张量。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 函数
     - 描述
   * - ``time_out``
     - 回合长度达到 ``env.max_episode_length`` 时返回 ``True``。注册时
       带 ``time_out=True``，让管理器把它当作截断。
   * - ``bad_orientation``
     - 资产上轴与世界竖直方向的夹角超过 ``limit_angle``（弧度）时返回
       ``True``。
   * - ``root_height_below_minimum``
     - 资产根连杆高度低于 ``minimum_height``（米）时返回 ``True``。
   * - ``nan_detection``
     - 物理状态中任何位置出现 NaN 或 Inf 时返回 ``True``。作为安全网，
       让发散的仿真被干净地终止。


编写自定义终止函数
------------------

自定义终止函数遵循与奖励函数相同的模式。普通函数接收 ``env`` 并返回
布尔 ``[num_envs]`` 张量。通用模式见 :ref:`env-config-term-pattern`。
