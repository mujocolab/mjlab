.. _rsl_rl:

Training with RSL-RL
====================

mjlab uses `RSL-RL <https://github.com/leggedrobotics/rsl_rl>`_ for on-policy
reinforcement learning. The integration has three parts: a **task registry**
that bundles environment and training configs under a single name, a
**VecEnv wrapper** that adapts mjlab environments to the interface RSL-RL
expects, and a set of **configuration dataclasses** that control the training
run.


Task registry
-------------

Every task in mjlab is a pair: an environment configuration
(``ManagerBasedRlEnvCfg``) and a training configuration
(``RslRlOnPolicyRunnerCfg``). The task registry maps a string name to this
pair so that training can be launched by name from the CLI.

Tasks are registered by calling ``register_mjlab_task`` in the task's
``__init__.py``:

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

Each registration takes:

- ``task_id``: a unique name following the convention
  ``Mjlab-{Category}-{Terrain}-{Robot}``
- ``env_cfg``: the ``ManagerBasedRlEnvCfg`` used for training
- ``play_env_cfg``: a variant with randomization disabled and episode length
  set to infinity, used for evaluation
- ``rl_cfg``: the ``RslRlOnPolicyRunnerCfg`` with PPO hyperparameters and
  network architecture
- ``runner_cls``: an optional custom runner class (defaults to
  ``MjlabOnPolicyRunner``)

All task packages under ``src/mjlab/tasks/`` are auto-discovered at import
time, so adding a new task only requires creating the config package and
calling ``register_mjlab_task``.


Training and playback
---------------------

**Launching a training run:**

.. code-block:: bash

    uv run train Mjlab-Velocity-Flat-Unitree-G1 --num-envs 4096

The task name is the first positional argument. The entire configuration
hierarchy (environment, scene, rewards, PPO hyperparameters, etc.) is
exposed as CLI flags through `tyro <https://brentyi.github.io/tyro/>`_.
Every field in ``ManagerBasedRlEnvCfg`` and ``RslRlOnPolicyRunnerCfg`` can
be overridden from the command line using dot-separated paths:

.. code-block:: bash

    uv run train Mjlab-Velocity-Flat-Unitree-G1 \
        --num-envs 4096 \
        --agent.max-iterations 10000 \
        --agent.algorithm.learning-rate 3e-4 \
        --env.decimation 2

.. important::

   - **Hyphens, not underscores**: Python field names use underscores
     (``num_envs``), but CLI flags use POSIX-style hyphens (``--num-envs``).
   - **Explicit booleans**: boolean flags require an explicit ``True`` or
     ``False`` value (e.g., ``--agent.resume True``, not ``--agent.resume``).
     This is intentional for compatibility with W&B sweep configs.

To discover available flags, use ``--help`` and pipe through ``grep``:

.. code-block:: bash

    # See all flags.
    uv run train Mjlab-Velocity-Flat-Unitree-G1 --help

    # Search for a specific field.
    uv run train Mjlab-Velocity-Flat-Unitree-G1 --help | grep learning-rate

Some commonly used top-level flags:

``--num-envs``
    Number of parallel simulation environments.

``--gpu-ids``
    GPU indices to use. Pass multiple indices for multi-GPU training (see
    :ref:`distributed-training`), or ``None`` for CPU mode.

``--video``
    Record training rollout videos to ``{log_dir}/videos/train/``.

``--enable-nan-guard``
    Enable NaN detection and state capture (see :ref:`nan-guard`).


**Playing back a trained policy:**

.. code-block:: bash

    # From W&B.
    uv run play Mjlab-Velocity-Flat-Unitree-G1 \
        --wandb-run-path your-entity/mjlab/run-id

    # From a local checkpoint.
    uv run play Mjlab-Velocity-Flat-Unitree-G1 \
        --checkpoint-file logs/rsl_rl/g1_velocity/2025-01-27_14-30-00/model_1000.pt

Key ``play`` arguments:

``--agent``
    Policy mode: ``"trained"`` (default), ``"zero"`` (zero actions), or
    ``"random"`` (uniform random).

``--viewer``
    Viewer backend: ``"native"`` (MuJoCo viewer) or ``"viser"``
    (browser-based).

``--no-terminations``
    Disable termination conditions so the policy runs indefinitely.


VecEnv wrapper
--------------

``RslRlVecEnvWrapper`` adapts a ``ManagerBasedRlEnv`` to RSL-RL's ``VecEnv``
interface. It handles three things:

1. **Observation format**: translates observation dictionaries into the
   ``TensorDict`` format RSL-RL expects.
2. **Done signal**: merges ``terminated`` and ``truncated`` into a single
   ``dones`` tensor and passes ``time_outs`` through ``extras`` so RSL-RL can
   bootstrap correctly on truncated episodes.
3. **Action clipping**: applies optional action clipping when ``clip_actions``
   is set in the runner config.

The wrapper also calls ``env.reset()`` during construction because RSL-RL does
not call reset before beginning rollout collection.

In normal usage you do not interact with the wrapper directly. The training
script handles wrapping automatically.


Configuration
-------------

``RslRlOnPolicyRunnerCfg`` is the top-level training configuration. It groups
runner settings, network architecture (``RslRlModelCfg``), and PPO
hyperparameters (``RslRlPpoAlgorithmCfg``). The following example from the
Unitree G1 velocity task shows a typical configuration:

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

All fields have sensible defaults and can be overridden from the command line
(e.g., ``--agent.algorithm.learning-rate 3e-4``). Use ``--help`` to see the
full list of available fields and their defaults.


Checkpoints and logging
-----------------------

Training artifacts are written to:

.. code-block:: text

    logs/rsl_rl/{experiment_name}/{timestamp}/
        model_{iteration}.pt      # policy checkpoints
        params/
            env.yaml              # full environment config
            agent.yaml            # full runner config

Checkpoints are saved every ``save_interval`` iterations and uploaded to W&B
as model artifacts by default. Set ``upload_model=False`` in the runner
config to disable uploads while keeping metric logging.

Checkpoint writes use a temporary file beside the destination, flush it, and
atomically replace the destination only after serialization succeeds. Readers see
the previous complete checkpoint until replacement. This protects against partial
writes; it does not guarantee recovery after disk loss. Temporary filenames are
excluded from the default checkpoint selector.

.. rubric:: Resuming from a checkpoint

.. code-block:: bash

    uv run train Mjlab-Velocity-Flat-Unitree-G1 \
        --num-envs 4096 \
        --agent.resume True

The runner searches for the most recent run directory under
``logs/rsl_rl/{experiment_name}/`` and loads the highest-numbered checkpoint.
Narrow the search with ``--agent.load-run`` (regex on directory names) and
``--agent.load-checkpoint`` (regex on checkpoint filenames).

``--agent.max-iterations`` controls how many *additional* iterations to run
from the checkpoint. If you are resuming from iteration 11500 with
``--agent.max-iterations 300`` (the default), training will run iterations
11500 through 11800. Set this to the number of new iterations you want.

To resume from a W&B run:

.. code-block:: bash

    uv run train Mjlab-Velocity-Flat-Unitree-G1 \
        --num-envs 4096 \
        --agent.resume True \
        --wandb-run-path your-entity/mjlab/run-id


.. rubric:: Stateful continuation at PPO update boundaries

Ordinary ``resume=True`` restores the policy, optimizer, iteration and curriculum
counter into a newly reset simulation. To save the additional state needed to
continue an in-progress rollout stream, opt in on both the original run and the
resumed run:

.. code-block:: bash

    uv run train Mjlab-Cartpole-Balance --gpu-ids None \
        --agent.stateful-resume True --agent.max-iterations 100

    uv run train Mjlab-Cartpole-Balance --gpu-ids None \
        --agent.stateful-resume True --agent.resume True \
        --agent.load-run ORIGINAL_RUN --agent.max-iterations 100

Stateful checkpoints contain native learner/optimizer/normalizer state, Python,
NumPy and Torch RNG states, live per-world Warp model parameters and data,
environment counters, manager timers/buffers and curriculum parameters,
observation/action delay and history buffers, the next observation, and recurrent
actor/critic memory. Saving happens after a complete PPO update. Continuation
starts at the next iteration and skips initial episode-length randomization,
environment reset and observation recomputation. ``max_iterations`` remains the
number of additional updates. Wall-clock timing and external logging sessions
are not reproducibility guarantees. Full snapshots copy live simulation arrays
to host memory and can be much larger than ordinary policy checkpoints; allow
for their host-memory, disk-space and update-boundary I/O costs.

The initial supported contract is the base ``MjlabOnPolicyRunner`` with native
uncompiled RSL-RL 5.5.1 PPO and
``MLPModel`` or ``RNNModel`` actor/critic models, including raycast-only sensing.
Raycast acceleration structures are rebuilt in the fresh environment and refit
before their next query; sensor buffers are restored. RND, symmetry, camera
render contexts, video wrappers, external recorders and opaque custom resources
are rejected. Runner subclasses are rejected before learner allocation, since
their extra state and overridden training behavior are outside this contract.
A checkpoint must match the task configuration, model, mjlab source,
software versions, batch/rank topology and math-backend settings. Mutable numeric
Warp options are saved; changing host-side solver/feature flags or array layouts
during training is rejected. This does not promise bitwise GPU determinism or
cross-device/cross-version reproducibility; MuJoCo Warp's existing determinism
limitations still apply.

Distributed snapshots store each rank's state. All ranks participate at the same
update boundary, then rank zero atomically publishes one checkpoint. No collective
runs inside the ordinary rank-zero ``save()`` hook. Direct distributed API users
must call ``save_training_checkpoint(path)`` and ``load(path)`` on every rank;
calling ``save()`` directly is rejected in stateful distributed mode. Capture and
publication errors propagate to peers, and newly created process groups use a
bounded timeout. CPU/gloo and CUDA/NCCL use the same protocol. Restore requires the
same rank topology; rank-specific RNG, simulation state and recurrent memory are
not broadcast from rank zero. Call ``runner.close()`` after API-driven training to
release a process group owned by the runner (the training script does this).
Stateful ``load()`` requires ``map_location`` to match the learner device. Failed
restore hooks stop all peers; discard the failed runner, since rollback of a
partially applied custom hook is not guaranteed.

Custom terms, commands, actuators and sensors must implement the explicit
``training_state_dict()`` / ``load_training_state_dict(state)`` protocol. Return a
data tree and include every mutable value that influences future outputs,
including private generators. Restore into the existing runtime object and
preserve references to simulation arrays. Components with no such state may
declare ``training_state_stateless = True``; custom functions must explicitly
declare that stateless contract too. Their defaults and closure captures must be
immutable data and are fingerprinted; mutable captures require a component hook.
Unsupported objects and incompatible tensor
storage/view layouts fail clearly instead of silently restoring partial state.
Lazy tensors with shared storage require an explicit hook. Checkpoints are trusted
PyTorch artifacts, with the same trust requirement as ordinary RSL checkpoints.


Citation
--------

If you use RSL-RL in your research, consider citing:

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
