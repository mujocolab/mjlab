.. _metrics:

指标
====

指标管理器把每步标量值以回合平均值的形式记录。与奖励不同，指标没有
权重，也不按步时长缩放。它们纯粹用于诊断：在不影响优化的前提下，把
跟踪误差、接触力或能耗等数量与奖励曲线放在一起观察。

指标在每个环境步计算、按环境累积，并在环境重置时对回合长度求平均。
得到的平均值写入训练日志器（TensorBoard 或 Weights & Biases），前缀为
``Episode_Metrics/``。

如果 ``ManagerBasedRlEnvCfg`` 上的 ``metrics`` 字典为空，环境会替换为
一个零开销的轻量空操作管理器。


注册
----

每个指标项在 ``ManagerBasedRlEnvCfg`` 的 ``metrics`` 字典中按名称注册。
配置非常精简：一个可调用对象和一个可选的 ``params`` 字典。

.. code-block:: python

    from mjlab.managers.metrics_manager import MetricsTermCfg

    metrics = {
        "base_height": MetricsTermCfg(
            func=base_height,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    }

可调用对象的第一个参数是 ``env``，``params`` 中的条目作为关键字参数
传入。它必须返回形状 ``[num_envs]`` 的张量，即每步每环境一个标量。


指标如何计算
------------

管理器为每个环境维护一个累计和与一个步数计数器。每次调用 ``compute()``：

1. 所有环境的步数计数器递增。
2. 以当前环境状态调用每个指标函数。
3. 返回的按环境数值累加进累计和。

环境重置时，管理器把每项的累计值归约为标量，跨所有被重置环境求平均，
并以 ``Episode_Metrics/<term_name>`` 为键返回。随后这些和与计数器对被
重置的环境清零。

归约方式由 ``MetricsTermCfg`` 的 ``reduce`` 字段控制：

- ``"mean"``（默认）：把累计和除以各环境的步数。除法按环境进行，
  因此提前终止的环境不会被运行更久的环境稀释。
- ``"last"``：报告回合最后一步的值。适用于不该随时间平均的布尔型成功
  指标（例如机器人是否站立）。
- ``"max"``：报告回合内出现的最大值，适用于峰值量，如最大功率或最大
  接触力。
- ``"sum"``：报告回合内的累计总量，不除以步数。适用于天然累积的量，
  如回合奖励或总移动距离。注意该值随回合长度增长，不同时长的回合之间
  不可比。

这些标量经 ``env.extras["log"]`` 流入训练运行器，由后者写入配置的
日志器。典型训练运行中它们显示为：

.. code-block:: text

    Episode_Metrics/base_height
    Episode_Metrics/contact_force

与奖励管理器产生的 ``Episode_Reward/`` 条目并列。


编写自定义指标函数
------------------

指标函数遵循与奖励和观测函数相同的模式。它以环境为第一个参数，读取
所需的状态，返回 ``[num_envs]`` 张量。

.. code-block:: python

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    def base_height(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        robot = env.scene[asset_cfg.name]
        return robot.data.root_link_pos_w[:, 2]

需要缓存初始化或回合内状态的指标，请把该项实现为带有
``__init__(self, cfg, env)`` 和 ``__call__`` 方法的类。若该类定义了
``reset(env_ids)`` 方法，管理器会在回合重置时自动调用。
