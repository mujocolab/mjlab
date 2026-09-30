.. _rewards:

奖励
====

奖励是塑造策略行为的训练信号。每个奖励项都是一个每步返回按环境标量的
函数。奖励管理器计算所有奖励项的加权和并返回给训练框架。

每个奖励项通过 ``RewardTermCfg`` 按名称注册，配置携带可调用对象和
``weight``。负权重产生惩罚。额外的关键字参数通过 ``params`` 提供。

.. code-block:: python

    from mjlab.envs.mdp import rewards
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    rewards_cfg = {
        "alive": RewardTermCfg(func=rewards.is_alive, weight=1.0),
        "joint_torques": RewardTermCfg(
            func=rewards.joint_torques_l2,
            weight=-1e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    }


内置奖励函数
------------

下列函数位于 ``mjlab.envs.mdp.rewards``，在所有任务间共享。各个任务还会
定义针对任务目标定制的奖励函数（如运动任务的速度跟踪）。所有奖励
函数都返回形状为 ``[num_envs]`` 的张量。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 函数
     - 描述
   * - ``is_alive``
     - 本步未终止的环境返回 ``1.0``。配正权重作生存奖励。
   * - ``is_terminated``
     - 因非超时条件终止的环境返回 ``1.0``。配负权重惩罚失败。
   * - ``joint_torques_l2``
     - 执行器力的平方和。惩罚高能耗动作。
   * - ``joint_vel_l2``
     - 关节速度的平方和。
   * - ``joint_acc_l2``
     - 关节加速度的平方和。
   * - ``action_rate_l2``
     - 当前动作与上一动作之差的平方和。惩罚策略输出的快速变化。
   * - ``action_acc_l2``
     - 动作二阶差的平方和。惩罚动作信号中的高频抖动。
   * - ``joint_pos_limits``
     - 关节位置超出软限位的惩罚。所有关节都在限位内时为零。
   * - ``posture`` *（类）*
     - 以指数核度量相对默认关节位置的偏差：
       ``exp(-mean(error^2 / std^2))``。
   * - ``electrical_power_cost`` *（类）*
     - 执行器消耗的正机械功率之和。再生功率不惩罚。
   * - ``flat_orientation_l2``
     - 基座系投影重力向量 x、y 分量的平方和。完全直立时为零。


奖励按 dt 缩放
--------------

``ManagerBasedRlEnvCfg.scale_rewards_by_dt`` 默认为 ``True``。启用时，
奖励管理器在累加前把每项乘以环境步时长。这使回合奖励总量对仿真频率
不变：50 Hz 下运行的任务与 200 Hz 下的同一任务产生相同的期望回合回报，
因为每步对总量的贡献按比例变小。

每项的回合累计值以 ``Episode_Reward/<term_name>`` 记录，并始终除以回合
时长，得到可跨不同回合长度的运行比较的奖励速率。


编写自定义奖励函数
------------------

奖励函数第一个参数接收 ``env``，返回 ``[num_envs]`` 张量。额外参数
声明为函数参数，通过 ``RewardTermCfg(params={...})`` 提供。当奖励项
需要缓存初始化工作或维护回合内状态时，把它实现为类。通用模式见
:ref:`env-config-term-pattern`。
