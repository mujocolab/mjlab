.. _events:

事件
====

事件管理器在环境生命周期的特定节点执行钩子。任何需要在启动时、回合
重置时或训练期间按间隔运行的逻辑，都注册为事件项。常见例子包括把实体
重置到初始状态、对模型参数施加域随机化、以随机速度扰动推动机器人，以及
从参考运动片段初始化机器人状态。它们都通过同一个 ``EventTermCfg``
接口配置，差别只在控制各项何时触发的 ``mode`` 字段。

作为事件最常见用途之一的域随机化拥有独立的参考页。完整的 ``dr`` 模块、
可用函数与内部机制见 :ref:`domain_randomization`。

.. code-block:: python

    from mjlab.envs.mdp import events as event_fns, dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    events = {
        # Reset all entities to their default state each episode.
        "reset_scene": EventTermCfg(
            func=event_fns.reset_scene_to_default,
            mode="reset",
        ),
        # Randomize foot friction once at startup.
        "foot_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=[".*foot.*"]),
                "ranges": (0.3, 1.2),
                "operation": "abs",
            },
        ),
        # Push the robot at random intervals during the episode.
        "push_robot": EventTermCfg(
            func=event_fns.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            },
        ),
        # Transient random impulses with duration and cooldown.
        "impulse": EventTermCfg(
            func=event_fns.apply_body_impulse,
            mode="step",
            params={
                "force_range": (-50.0, 50.0),
                "torque_range": (0.0, 0.0),
                "duration_s": (0.1, 0.2),
                "cooldown_s": (1.0, 3.0),
                "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
            },
        ),
    }


生命周期模式
------------

``EventTermCfg`` 的 ``mode`` 字段决定各项何时触发。四种模式对应 RL
训练运行的四个时间尺度：进程启动时一次、每回合一次、回合内周期性、
每个环境步。

``"startup"``
    在环境初始化期间触发一次，此时所有管理器都已构建。每个环境同时
    收到该事件。该模式适用于希望跨环境不同、但在整个训练运行期间保持
    不变的参数，例如经 ``dr`` 模块随机化的连杆质量或关节 armature。

``"reset"``
    在每次回合重置时对每个被重置的环境触发。这是最常见的模式。状态
    初始化（把机器人写回默认位姿）和回合级域随机化都属于这里。

    可选的 ``min_step_count_between_reset`` 字段防止该项在回合很短时
    触发过频。自上次触发以来步数未达到该值的环境会跳过本项。首次调用
    无条件触发。

``"interval"``
    在训练期间按固定时间间隔触发，与回合边界无关。触发频率由
    ``interval_range_s``（秒为单位的 ``(min, max)`` 区间）控制。每次
    触发后管理器从该区间均匀采样新的等待时间。默认每个环境有独立的
    计时器；设置 ``is_global_time=True`` 可把所有环境同步到单一共享
    计时器。间隔事件是回合中段扰动（外部推动、模型参数漂移）的天然
    归宿。

``"step"``
    在每个环境步对所有环境触发。该模式适用于必须每步评估的连续效应，
    例如自带内部时长与冷却计时器的 ``apply_body_impulse``。step 事件
    每步都运行，应保持轻量，或在内部自行管理激活逻辑以避免不必要的
    计算。

与所有管理器项一样，``func`` 指向可调用对象，``params`` 保存随
``env`` 和 ``env_ids`` 一起转发给它的关键字参数。``params`` 中的
``SceneEntityCfg`` 值在管理器构建时解析一次（正则模式在那时匹配到模型
索引，而不是每次调用时匹配）。各项可以是普通函数或类；通用模式见
:ref:`env-config-term-pattern`。


内置事件函数
------------

下列函数位于 ``mjlab.envs.mdp.events``。

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 函数
     - 描述
   * - ``reset_scene_to_default``
     - 把所有实体重置到默认状态：浮动基座实体的根位姿与速度、固定基座
       实体的 mocap 位姿、铰接实体的关节位置与速度。环境原点自动生效。
       这是 ``ManagerBasedRlEnvCfg`` 的默认事件；大多数环境保留它并在
       旁边追加其他项。
   * - ``reset_root_state_uniform``
     - 以默认值加均匀随机偏移重置单个实体的根位姿与速度。接受键为
       ``"x"``、``"y"``、``"z"``、``"roll"``、``"pitch"``、``"yaw"`` 的
       ``pose_range`` 和 ``velocity_range`` 字典。姿态扰动与默认四元数
       复合。对固定基座机器人，这是把它们放到各自环境原点的唯一途径；
       否则它们会堆在世界原点。
   * - ``reset_root_state_from_flat_patches``
     - 依据环境被指派的地形等级和类型，把实体放到随机选择的平坦地形块
       上。没有可用平地块时回退到 ``reset_root_state_uniform``。适用于
       机器人应在所分配子地形的水平地面上出生的运动任务。
   * - ``reset_joints_by_offset``
     - 通过给实体默认值加均匀随机偏移重置关节位置与速度，并截断到软
       关节限位内。
   * - ``push_by_setting_velocity``
     - 给实体当前根速度叠加一个随机速度增量，模拟外部推动。通常配
       ``mode="interval"`` 测试抗扰性。
   * - ``apply_external_force_torque``
     - 经 MuJoCo 外部力旋量机制对一个或多个 body 施加随机力和力矩。
   * - ``apply_body_impulse``
     - 对 body 施加时长与冷却均可配置的瞬态外部力旋量。每个环境独立
       采样随机力方向并保持采样时长，随后经过冷却期再次触发。支持可选
       的 ``body_point_offset`` 把施加点移离质心。内置调试可视化，可在
       查看器中绘制力箭头。配 ``mode="step"`` 使用。
   * - ``randomize_terrain``
     - 把每个环境指派到随机的子地形行列，忽略课程。适用于评估或试玩
       模式。


编写自定义事件项
----------------

事件函数的前两个参数是 ``env`` 和 ``env_ids``，其余参数来自
``EventTermCfg.params``。它就地修改仿真状态，无返回值。对需要昂贵
一次性初始化（例如从磁盘加载数据）的项，请用类，让初始化只在构造时
运行一次而不是每次调用都执行。例如，下面这个自定义事件项把机器人重置
为从预录数据集中采样的随机位姿：

.. code-block:: python

    import torch
    from mjlab.managers.manager_base import ManagerTermBase
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    class ResetFromDataset(ManagerTermBase):
        """Reset the robot to a random pose from a dataset."""

        def __init__(self, cfg, env):
            super().__init__(env)
            self._robot = env.scene["robot"]
            self._poses = torch.load(
                cfg.params["dataset_path"],
                map_location=env.device,
            )

        def __call__(self, env, env_ids, **kwargs):
            # Sample with replacement: each env gets an independent pose.
            indices = torch.randint(
                len(self._poses), (len(env_ids),), device=env.device,
            )
            self._robot.write_joint_position_to_sim(
                self._poses[indices], env_ids=env_ids,
            )

当项需要维护状态或执行昂贵初始化时，把它实现为类。通用模式见
:ref:`env-config-term-pattern`。写入模型字段的自定义 DR 项见
:ref:`domain_randomization`。
