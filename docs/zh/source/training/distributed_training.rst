.. _distributed-training:

分布式训练
==========

mjlab 通过 `torchrunx <https://github.com/apoorvkh/torchrunx>`_ 支持
多 GPU 分布式训练。每块 GPU 用自己的环境独立采样回合，策略更新期间
同步梯度。吞吐量随 GPU 数量接近线性扩展。


用法
----

.. code-block:: bash

    # Single GPU (default).
    uv run train <task-name> --gpu-ids "[0]"

    # Two GPUs.
    uv run train <task-name> --gpu-ids "[0, 1]"

    # All available GPUs.
    uv run train <task-name> --gpu-ids all

    # CPU mode.
    uv run train <task-name> --gpu-ids None

要点：

- 若设置了 ``CUDA_VISIBLE_DEVICES``，GPU 索引相对于它。例如
  ``CUDA_VISIBLE_DEVICES=2,3 uv run train ... --gpu-ids "[0, 1]"`` 使用
  物理 GPU 2 和 3。
- 单 GPU 与 CPU 模式直接运行，不经过 torchrunx。


扩展行为
--------

多 GPU 训练是 **数据并行，不是任务切分**。每块 GPU 独立运行完整的
``num-envs`` 数量，因此每次迭代收集的总经验为：

.. code-block:: text

    experience per iteration = num_envs x num_steps_per_env x num_gpus

迭代速度大致不变，因为每块 GPU 做的工作量相同。收益在于每次策略更新
看到更多样的经验，策略在真实时间上收敛更快。

.. important::

   由于 ``max-iterations`` 不会自动调整，用更多 GPU 训练会按比例跑得更
   久。如果想要相同的总训练时长，请按 GPU 数量缩小 ``max-iterations``
   （例如从 1 卡翻倍到 2 卡时减半）。


工作原理
--------

mjlab 的职责是用 ``wp.ScopedDevice`` **把 MuJoCo Warp 仿真隔离到每块
GPU 上**。其余由 torchrunx 处理。

**进程派生。** ``torchrunx.Launcher`` 为每块 GPU 派生一个进程，并设置
``RANK``、``LOCAL_RANK`` 和 ``WORLD_SIZE`` 协调它们。每个进程在分派的
GPU 上执行训练函数。

**独立采样。** 每个进程维护自己的：

- 环境实例（含 ``num-envs`` 个并行环境），经 ``wp.ScopedDevice`` 隔离
  在分派的 GPU 上
- 策略网络副本
- 经验缓冲区（大小 ``num_steps_per_env * num_envs``）

每个进程使用 ``seed = cfg.seed + local_rank``，保证各 GPU 的随机经验
不同，提高样本多样性。

**梯度同步。** 更新阶段，RSL-RL 通过 ``reduce_parameters()`` 方法在每个
mini-batch 之后同步梯度：

1. 每个进程在自己的本地 mini-batch 上独立计算梯度
2. 所有策略梯度被展平为单个张量
3. ``torch.distributed.all_reduce`` 跨所有 GPU 平均梯度
4. 平均后的梯度拷回各参数，保持策略同步

**单写者 I/O。** 只有 rank 0 写配置文件、视频和 W&B 日志，避免竞态
条件。


日志
----

默认情况下，torchrunx 进程日志保存在 ``{log_dir}/torchrunx/``。可以
自定义：

.. code-block:: bash

    # Disable torchrunx file logging.
    uv run train <task-name> --gpu-ids "[0, 1]" --torchrunx-log-dir ""

    # Custom log directory.
    uv run train <task-name> --gpu-ids "[0, 1]" --torchrunx-log-dir /path/to/logs

    # Environment variable (takes precedence over the flag).
    TORCHRUNX_LOG_DIR=/tmp/logs uv run train <task-name> --gpu-ids "[0, 1]"
