.. _rsl_rl:

使用 RSL-RL 训练
================

mjlab 使用 `RSL-RL <https://github.com/leggedrobotics/rsl_rl>`_ 做 on-policy
强化学习。集成由三部分组成：把环境配置和训练配置捆绑在单一名称下的
**任务注册表**、把 mjlab 环境适配到 RSL-RL 所需接口的 **VecEnv 包装器**，
以及控制训练运行的一组 **配置 dataclass**。


任务注册表
----------

mjlab 中的每个任务都是一个配对：环境配置（``ManagerBasedRlEnvCfg``）加
训练配置（``RslRlOnPolicyRunnerCfg``）。任务注册表把字符串名称映射到
这一配对，从而可以从 CLI 按名称启动训练。

在任务的 ``__init__.py`` 中调用 ``register_mjlab_task`` 注册任务：

.. code-block:: python

    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

    from .env_cfgs import unitree_g1_rough_env_cfg, unitree_g1_flat_env_cfg
    from .rl_cfg import unitree_g1_ppo_runner_cfg

    register_mjlab_task(
      task_id="Mjlab-Velocity-Rough-Unitree-G1",
      env_cfg=unitree_g1_rough_env_cfg(),
      play_env_cfg=unitree_g1_rough_env_cfg(play=True),
      rl_cfg=unitree_g1_ppo_runner_cfg(),
      runner_cls=VelocityOnPolicyRunner,
    )

每次注册接收：

- ``task_id``：唯一名称，遵循约定
  ``Mjlab-{Category}-{Terrain}-{Robot}``
- ``env_cfg``：训练使用的 ``ManagerBasedRlEnvCfg``
- ``play_env_cfg``：关闭随机化、回合长度设为无穷的变体，用于评估
- ``rl_cfg``：包含 PPO 超参数与网络结构的 ``RslRlOnPolicyRunnerCfg``
- ``runner_cls``：可选的自定义 runner 类（默认
  ``MjlabOnPolicyRunner``）

``src/mjlab/tasks/`` 下的所有任务包都会在导入时被自动发现，因此添加
新任务只需创建配置包并调用 ``register_mjlab_task``。


训练与回放
----------

**启动训练：**

.. code-block:: bash

    uv run train Mjlab-Velocity-Flat-Unitree-G1 --num-envs 4096

任务名是第一个位置参数。整个配置层级（环境、场景、奖励、PPO 超参数等）
通过 `tyro <https://brentyi.github.io/tyro/>`_ 暴露为 CLI 标志。
``ManagerBasedRlEnvCfg`` 和 ``RslRlOnPolicyRunnerCfg`` 中的每个字段都
可以用点分路径从命令行覆盖：

.. code-block:: bash

    uv run train Mjlab-Velocity-Flat-Unitree-G1 \
        --num-envs 4096 \
        --agent.max-iterations 10000 \
        --agent.algorithm.learning-rate 3e-4 \
        --env.decimation 2

.. important::

   - **用连字符，不用下划线**：Python 字段名用下划线（``num_envs``），
     CLI 标志用 POSIX 风格连字符（``--num-envs``）。
   - **布尔值显式写**：布尔标志必须显式给出 ``True`` 或 ``False``
     （例如 ``--agent.resume True``，而不是 ``--agent.resume``）。这是
     为兼容 W&B sweep 配置而有意为之。

用 ``--help`` 并配合 ``grep`` 发现可用标志：

.. code-block:: bash

    # See all flags.
    uv run train Mjlab-Velocity-Flat-Unitree-G1 --help

    # Search for a specific field.
    uv run train Mjlab-Velocity-Flat-Unitree-G1 --help | grep learning-rate

几个常用的顶层标志：

``--num-envs``
    并行仿真环境数量。

``--gpu-ids``
    使用的 GPU 索引。多 GPU 训练传多个索引（见
    :ref:`distributed-training`），CPU 模式传 ``None``。

``--video``
    把训练回合采样视频录到 ``{log_dir}/videos/train/``。

``--enable-nan-guard``
    启用 NaN 检测与状态捕获（见 :ref:`nan-guard`）。


**回放训练好的策略：**

.. code-block:: bash

    # From W&B.
    uv run play Mjlab-Velocity-Flat-Unitree-G1 \
        --wandb-run-path your-entity/mjlab/run-id

    # From a local checkpoint.
    uv run play Mjlab-Velocity-Flat-Unitree-G1 \
        --checkpoint-file logs/rsl_rl/g1_velocity/2025-01-27_14-30-00/model_1000.pt

``play`` 的关键参数：

``--agent``
    策略模式：``"trained"``（默认）、``"zero"``（零动作）或
    ``"random"``（均匀随机）。

``--viewer``
    查看器后端：``"native"``（MuJoCo 查看器）或 ``"viser"``（浏览器）。

``--no-terminations``
    禁用终止条件，让策略无限运行。


VecEnv 包装器
-------------

``RslRlVecEnvWrapper`` 把 ``ManagerBasedRlEnv`` 适配到 RSL-RL 的
``VecEnv`` 接口。它处理三件事：

1. **观测格式**：把观测字典转换为 RSL-RL 期望的 ``TensorDict`` 格式。
2. **完成信号**：把 ``terminated`` 和 ``truncated`` 合并为单个
   ``dones`` 张量，并经 ``extras`` 传递 ``time_outs``，让 RSL-RL 能在
   被截断的回合上正确自举。
3. **动作裁剪**：当 runner 配置设置了 ``clip_actions`` 时应用可选的
   动作裁剪。

包装器还在构造期间调用 ``env.reset()``，因为 RSL-RL 在开始采样前不会
调用 reset。

正常使用中你不需要直接接触包装器。训练脚本会自动完成包装。


配置
----

``RslRlOnPolicyRunnerCfg`` 是顶层训练配置，分组了 runner 设置、网络
结构（``RslRlModelCfg``）和 PPO 超参数（``RslRlPpoAlgorithmCfg``）。
下面这个来自 Unitree G1 速度任务的示例展示了一个典型配置：

.. code-block:: python

    from mjlab.rl import (
        RslRlModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
    )

    def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
        return RslRlOnPolicyRunnerCfg(
            actor=RslRlModelCfg(
                hidden_dims=(512, 256, 128),
                activation="elu",
                obs_normalization=True,
            ),
            critic=RslRlModelCfg(
                hidden_dims=(512, 256, 128),
                activation="elu",
                obs_normalization=True,
            ),
            algorithm=RslRlPpoAlgorithmCfg(
                value_loss_coef=1.0,
                use_clipped_value_loss=True,
                clip_param=0.2,
                entropy_coef=0.01,
                num_learning_epochs=5,
                num_mini_batches=4,
                learning_rate=1.0e-3,
                schedule="adaptive",
                gamma=0.99,
                lam=0.95,
                desired_kl=0.01,
                max_grad_norm=1.0,
            ),
            experiment_name="g1_velocity",
            save_interval=50,
            num_steps_per_env=24,
            max_iterations=30_000,
        )

所有字段都有合理默认值，可以从命令行覆盖（例如
``--agent.algorithm.learning-rate 3e-4``）。用 ``--help`` 查看全部可用
字段及其默认值。


检查点与日志
------------

训练产物写入：

.. code-block:: text

    logs/rsl_rl/{experiment_name}/{timestamp}/
        model_{iteration}.pt      # policy checkpoints
        params/
            env.yaml              # full environment config
            agent.yaml            # full runner config

检查点默认每 ``save_interval`` 次迭代保存一次，并作为模型工件上传到
W&B。在 runner 配置中设 ``upload_model=False`` 可禁用上传、保留指标
记录。

.. rubric:: 从检查点恢复

.. code-block:: bash

    uv run train Mjlab-Velocity-Flat-Unitree-G1 \
        --num-envs 4096 \
        --agent.resume True

runner 会在 ``logs/rsl_rl/{experiment_name}/`` 下搜索最近的运行目录并
加载编号最高的检查点。用 ``--agent.load-run``（目录名正则）和
``--agent.load-checkpoint``（检查点文件名正则）缩小搜索范围。

``--agent.max-iterations`` 控制从检查点再跑多少 *额外* 迭代。如果从
迭代 11500 恢复并保持默认 ``--agent.max-iterations 300``，训练将运行
11500 到 11800。把它设成你想要的新增迭代数即可。

从 W&B run 恢复：

.. code-block:: bash

    uv run train Mjlab-Velocity-Flat-Unitree-G1 \
        --num-envs 4096 \
        --agent.resume True \
        --wandb-run-path your-entity/mjlab/run-id


引用
----

如果你在研究中使用了 RSL-RL，请考虑引用：

.. code-block:: bibtex

    @article{schwarke2025rslrl,
        title={RSL-RL: A Learning Library for Robotics Research},
        author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
        journal={arXiv preprint arXiv:2509.10771},
        year={2025}
    }

.. toctree::
   :maxdepth: 1

   motion_imitation
