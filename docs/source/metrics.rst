.. _metrics:

Metrics
=======

The metrics manager logs per-step scalar values as episode averages. Unlike
rewards, metrics carry no weight and are not scaled by the step duration.
They exist purely for diagnostics: tracking quantities such as tracking
error, contact forces, or energy consumption alongside reward curves
without influencing the optimization.

Metrics are computed every environment step, accumulated per environment,
and averaged over the episode length when the environment resets. The
resulting averages are written to the training logger (TensorBoard or
Weights & Biases) under the ``Episode_Metrics/`` prefix.

If the ``metrics`` dictionary on ``ManagerBasedRlEnvCfg`` is empty, the
environment substitutes a lightweight no-op manager with zero overhead.


Registration
------------

Each metric term is registered by name in the ``metrics`` dictionary of
``ManagerBasedRlEnvCfg``. The configuration is minimal: a callable and an
optional ``params`` dictionary.

.. code-block:: python

    from mjlab.managers.metrics_manager import MetricsTermCfg

    metrics = {
        "base_height": MetricsTermCfg(
            func=base_height,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    }

The callable receives ``env`` as its first argument and any entries in
``params`` as keyword arguments. It must return a tensor of shape
``[num_envs]``, one scalar per environment per step.


How metrics are computed
-------------------------

The manager maintains a running sum and a step counter for each
environment. On every call to ``compute()``:

1. The step counter increments for all environments.
2. Each term function is called with the current environment state.
3. The returned per-environment values are added to the running sums.

When an environment resets, the manager divides each term's accumulated sum
by the step count for that environment, averages the result across all
resetting environments, and returns the scalar under the key
``Episode_Metrics/<term_name>``. The sums and counters are then zeroed for
the reset environments. Division is per-environment, so environments that
terminated early are not diluted by longer-running ones.

These scalars flow through ``env.extras["log"]`` into the training runner,
which writes them to the configured logger. In a typical training run they
appear as:

.. code-block:: text

    Episode_Metrics/base_height
    Episode_Metrics/contact_force

alongside the ``Episode_Reward/`` entries produced by the reward manager.


Writing custom metric functions
--------------------------------

A metric function follows the same pattern as reward and observation
functions. It takes the environment as its first argument, reads whatever
state it needs, and returns a ``[num_envs]`` tensor.

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

For metrics that require cached setup or per-episode state, implement the
term as a class with ``__init__(self, cfg, env)`` and a ``__call__``
method. If the class defines a ``reset(env_ids)`` method, the manager
calls it automatically on episode resets.
