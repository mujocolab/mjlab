.. _distributed-training:

Distributed Training
====================

mjlab supports multi-GPU distributed training using
`torchrunx <https://github.com/apoorvkh/torchrunx>`_. Each GPU runs
independent rollouts with its own environments, and gradients are
synchronized during policy updates. Throughput scales nearly linearly with
GPU count.


Usage
-----

.. code-block:: bash

    # Single GPU (default).
    uv run train <task-name> --gpu-ids 0

    # Two GPUs.
    uv run train <task-name> --gpu-ids 0 1

    # All available GPUs.
    uv run train <task-name> --gpu-ids all

    # CPU mode.
    uv run train <task-name> --gpu-ids None

Key points:

- GPU indices are relative to ``CUDA_VISIBLE_DEVICES`` if set. For example,
  ``CUDA_VISIBLE_DEVICES=2,3 uv run train ... --gpu-ids 0 1`` uses physical
  GPUs 2 and 3.
- Single-GPU and CPU modes run directly without torchrunx.


Scaling behavior
----------------

Multi-GPU training is **data-parallel, not work-splitting**. Each GPU runs
the full ``num-envs`` count independently, so the total experience collected
per iteration is:

.. code-block:: text

    experience per iteration = num_envs x num_steps_per_env x num_gpus

Iteration speed stays roughly the same because each GPU does the same amount
of work. The benefit is that each policy update sees more diverse experience,
so the policy converges faster in wall-clock time.

.. important::

   Because ``max-iterations`` is not automatically adjusted, training with
   more GPUs runs for proportionally longer. If you want the same total
   training time, scale ``max-iterations`` down by the number of GPUs
   (e.g., halve it when doubling from 1 to 2 GPUs).


How it works
------------

mjlab's role is to **isolate MuJoCo Warp simulations on each GPU** using
``wp.ScopedDevice``. torchrunx handles the rest.

**Process spawning.** ``torchrunx.Launcher`` spawns one process per GPU and
sets ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE`` to coordinate them. Each
process executes the training function with its assigned GPU.

**Independent rollouts.** Each process maintains its own:

- Environment instances (with ``num-envs`` parallel environments), isolated
  on its assigned GPU via ``wp.ScopedDevice``
- Policy network copy
- Experience buffer (sized ``num_steps_per_env * num_envs``)

Each process uses ``seed = cfg.seed + local_rank`` to ensure different
random experiences across GPUs, increasing sample diversity.

**Gradient synchronization.** During the update phase, RSL-RL synchronizes
gradients after each mini-batch through its ``reduce_parameters()`` method:

1. Each process computes gradients independently on its local mini-batch
2. All policy gradients are flattened into a single tensor
3. ``torch.distributed.all_reduce`` averages gradients across all GPUs
4. Averaged gradients are copied back to each parameter, keeping policies
   synchronized

**Single-writer I/O.** Only rank 0 writes config files, videos, and W&B
logs to avoid race conditions.


Logging
-------

By default, torchrunx process logs are saved to ``{log_dir}/torchrunx/``.
This can be customized:

.. code-block:: bash

    # Disable torchrunx file logging.
    uv run train <task-name> --gpu-ids 0 1 --torchrunx-log-dir ""

    # Custom log directory.
    uv run train <task-name> --gpu-ids 0 1 --torchrunx-log-dir /path/to/logs

    # Environment variable (takes precedence over the flag).
    TORCHRUNX_LOG_DIR=/tmp/logs uv run train <task-name> --gpu-ids 0 1
