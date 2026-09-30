.. _recorders:

记录器
======

记录器管理器提供在回合采样期间记录数据的生命周期钩子。与奖励不同，
记录器对优化循环没有任何影响。它们的存在纯粹是为了让你在不修改 mjlab
内部的前提下捕获观测、动作或任何其他环境状态。

每个记录器项都是你实现的一个类。mjlab 在恰当的时机调用其方法，把所有
I/O 决策留给你。如果 ``ManagerBasedRlEnvCfg`` 的 ``recorders`` 字典为空，
环境会替换为一个零开销的轻量空操作管理器。


生命周期钩子
------------

管理器为每项暴露三个钩子：

``record_pre_reset(env_ids)``
    在 ``env.step()`` 内部、被终止环境重置之前调用。``obs_buf`` 保存的
    是 *上一步* 末尾的观测（智能体用来选择终止动作的输入，而不是动作
    之后的终止状态）。``action_manager.action`` 保存终止动作，此刻仍然
    有效；紧接着 ``_reset_idx`` 会把这些环境的动作清零。``reward_buf``
    保存终止奖励。这里是记录终止转移 ``(obs_t, action_t, reward_t,
    done=True)`` 的正确位置。

``record_post_reset(env_ids)``
    在重置完成、新的观测可用之后调用。它在 ``env.reset()``（全部环境）
    末尾触发，也在 ``env.step()`` 内每批被终止环境重置之后触发。
    ``obs_buf[env_ids]`` 保存新回合的初始观测；
    ``action_manager.action[env_ids]`` 为零。用它来初始化回合内状态或
    记录第一条观测。

``record_post_step()``
    在每个 ``env.step()`` 末尾、新观测就绪时调用。对本步中发生重置的
    环境，``action_manager.action`` 已被清零，``obs_buf`` 保存的是新
    回合的初始状态而不是动作后的终止观测。这些环境的终止转移请用
    ``record_pre_reset`` 记录，用 ``self._env.reset_buf`` 识别哪些环境
    发生了重置。

``close()``
    在环境关闭时调用。在这里释放文件句柄并刷新缓冲区。


编写记录器项
------------

继承 :class:`~mjlab.managers.RecorderTerm` 并覆写所需的钩子。环境以
``self._env`` 的形式可用，可以访问 ``self._env.obs_buf``、
``self._env.action_manager.action`` 以及所有其他管理器。

.. code-block:: python

    import csv
    from mjlab.managers import RecorderTerm, RecorderTermCfg

    class CsvRecorder(RecorderTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._file = open(cfg.params["path"], "w", newline="")
            self._writer = csv.writer(self._file)

        def record_pre_reset(self, env_ids):
            # Terminal transition: action is still intact here.
            # It will be zeroed by _reset_idx immediately after this returns.
            obs = self._env.obs_buf["actor"][env_ids].cpu().numpy()
            act = self._env.action_manager.action[env_ids].cpu().numpy()
            for o, a in zip(obs, act):
                self._writer.writerow(o.tolist() + a.tolist())

        def record_post_step(self):
            # Skip envs that just reset: their terminal pair was written
            # in record_pre_reset and their action is now zeroed.
            mask = ~self._env.reset_buf
            obs = self._env.obs_buf["actor"][mask].cpu().numpy()
            act = self._env.action_manager.action[mask].cpu().numpy()
            for o, a in zip(obs, act):
                self._writer.writerow(o.tolist() + a.tolist())

        def close(self):
            self._file.close()

记录器项会收到完整的 ``cfg`` 对象，因此可以读取你放进 ``cfg.params``
的任何值。


注册
----

把记录器项加入环境配置的 ``recorders`` 字典：

.. code-block:: python

    from dataclasses import dataclass, field
    from mjlab.managers import RecorderTermCfg

    @dataclass
    class MyEnvCfg(SomeTaskEnvCfg):
        recorders: dict = field(default_factory=lambda: {
            "csv": RecorderTermCfg(
                func=CsvRecorder,
                params={"path": "rollout.csv"},
            )
        })

多个记录器项可以用不同的键注册并同时运行。

.. note::

    ``func`` 必须是 :class:`~mjlab.managers.RecorderTerm` 的子类。
    不支持基于函数的记录器项，因为记录器项是有状态的。
