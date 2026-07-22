.. _observations:

Observations
============

Observations define what the agent perceives at each step. The
observation manager assembles individual observation terms into the
tensor the policy receives as input. Each term passes through a
configurable processing pipeline: noise injection, clipping, scaling,
sensor delay, and history stacking.


Observation groups
------------------

Each group is an ``ObservationGroupCfg`` that holds a ``terms`` dict
mapping string names to ``ObservationTermCfg`` entries. The manager
concatenates term outputs in registration order along the last dimension.

.. code-block:: python

    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.envs.mdp import observations as obs_fns

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                "projected_gravity": ObservationTermCfg(
                    func=obs_fns.projected_gravity
                ),
                "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel),
                "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                "last_action": ObservationTermCfg(func=obs_fns.last_action),
            },
            enable_corruption=True,
        ),
    }

This dictionary is passed to ``ManagerBasedRlEnvCfg(observations=...)``.
The observation manager resolves term functions at initialization and
allocates any required history or delay buffers at that point.

By default, term outputs within a group are concatenated along the last
dimension into a single ``[num_envs, D]`` tensor. Set
``concatenate_terms=False`` to receive a dict mapping term names to
individual tensors instead.

The ``enable_corruption`` flag gates noise application for the entire
group: when ``False``, noise configs on individual terms are ignored.
This makes it straightforward to share term definitions between a noisy
actor group and a noise-free critic group, as shown in the
:ref:`asymmetric actor-critic <obs-asymmetric>` section below.

History and delay can also be set at the group level to apply uniformly
across all terms; see :ref:`obs-history-delay`.


Processing pipeline
-------------------

Each step, every term in every group passes through the following
pipeline in order:

.. code-block:: text

    compute → noise → clip → scale → delay → history

1. **compute**: the term function is called. It must return a
   ``[num_envs, D]`` tensor.

2. **noise**: if ``enable_corruption=True`` on the group and the term
   has a ``noise`` config, noise is applied. Stateless noise
   (``NoiseCfg``) is applied directly; stateful noise (``NoiseModelCfg``)
   is maintained by the manager across steps.

3. **clip**: if ``clip=(lo, hi)`` is set on the term, values are clamped
   to that range.

4. **scale**: if ``scale`` is set, the output is multiplied
   element-wise. Accepts a scalar, a tuple, or a tensor.

5. **delay**: if ``delay_max_lag > 0``, the term's output is stored in a
   ring buffer and a value from an earlier step is returned. See
   :ref:`obs-history-delay`.

6. **history**: if ``history_length > 0``, past outputs are stacked.
   See :ref:`obs-history-delay`.

.. note::

   Delay is applied before history. This models real systems where old
   sensor readings are buffered: the history stacks delayed observations,
   not future ones.


.. _obs-history-delay:

Observation history and delay
------------------------------

Observations support two temporal features: history and delay. History
stacks past frames to give the policy temporal context; delay models
sensor latency by returning observations from earlier timesteps.

Both are configured per term via fields on ``ObservationTermCfg``.
They can also be set at the group level on ``ObservationGroupCfg``,
which applies uniformly to all terms in the group. Term-level settings
override group-level settings.

History
^^^^^^^

Setting ``history_length=N`` stacks the N most recent outputs of a term.
When ``flatten_history_dim=True`` (the default), the history dimension
is folded into the feature dimension, producing a ``[num_envs, N * D]``
tensor suitable for MLPs. When ``flatten_history_dim=False``, the output
retains the time dimension as ``[num_envs, N, D]``, suitable for RNNs.

History buffers are cleared on environment reset. The first observation
after reset is backfilled across all history slots, so the policy
receives valid data from step zero.

When ``flatten_history_dim=True`` and ``concatenate_terms=True``, mjlab
uses **term-major** ordering: each term's full history is flattened
before concatenating across terms.

.. code-block:: text

    Term A (D=4, history=3), Term B (D=2, history=3):
    [A_t0, A_t1, A_t2, B_t0, B_t1, B_t2]
     └─ A history ──┘  └─ B history ─┘

Some frameworks use **time-major** ordering instead, where full frames
are built at each timestep before concatenating across time. Transferring
policies between frameworks with different orderings requires reindexing
the observation vector.

Delay
^^^^^

Setting ``delay_max_lag > 0`` enables a ring buffer that stores past
outputs and returns one from an earlier step. The lag is sampled
uniformly from ``[delay_min_lag, delay_max_lag]`` in integer steps.
A lag of zero returns the current observation; a lag of two returns the
observation from two steps ago.

.. code-block:: text

    50Hz control (20ms/step), lag=2:

    Sensor captures:  A     B     C     D     E     F     G     H
    Control steps:    0     1     2     3     4     5     6     7

    Policy sees:      A     A     A     B     C     D     E     F
                      └clamp┘     └ 40ms delay from here on

    Steps 0-1: lag clamped because the buffer is not yet full.
    Step 2 onward: each step returns the observation from 2 steps ago.

To convert real-world latency to lag steps:
``lag = latency_seconds / step_dt``. At 50 Hz control (20 ms per step),
a 40 ms sensor latency corresponds to a lag of 2. Delays are quantized
to integer steps; to approximate a latency that falls between steps, set
``delay_min_lag`` and ``delay_max_lag`` to the two nearest integers.

By default each environment samples its own lag independently
(``delay_per_env=True``). Additional parameters control resampling
frequency (``delay_update_period``), hold probability
(``delay_hold_prob``), and phase staggering
(``delay_per_env_phase``).

Both history and delay buffers are allocated only when enabled; terms
with default settings incur no overhead.


Built-in observation functions
--------------------------------

The functions below live in ``mjlab.envs.mdp.observations`` (also
re-exported as ``mjlab.envs.mdp``). All return ``[num_envs, D]``
tensors.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Function
     - Description
   * - ``base_lin_vel``
     - Linear velocity of the robot base in the base frame.
   * - ``base_ang_vel``
     - Angular velocity of the robot base in the base frame.
   * - ``projected_gravity``
     - Gravity vector projected into the base frame. Provides roll and
       pitch information without an explicit orientation representation.
   * - ``joint_pos_rel``
     - Joint positions relative to the default pose. Pass
       ``biased=True`` for encoder-biased positions (for sim2real with
       ``dr.encoder_bias``).
   * - ``joint_vel_rel``
     - Joint velocities relative to the default velocities.
   * - ``last_action``
     - The most recent action tensor. Optionally pass ``action_name``
       to select a single action term.
   * - ``generated_commands``
     - The current command tensor from a named command term. Requires
       ``params={"command_name": "<name>"}``.
   * - ``builtin_sensor``
     - Raw data from a named ``BuiltinSensor`` (MuJoCo ``sensordata``
       slice). Requires ``params={"sensor_name": "<entity>/<sensor>"}``.
   * - ``height_scan``
     - Height above each raycast hit point from a ``RayCastSensor``.
       Requires ``params={"sensor_name": "<name>"}``.

For ``builtin_sensor`` and ``height_scan``, the ``sensor_name`` parameter
must match a sensor registered in the scene. See :ref:`sensors` for how
to configure sensors.


.. _obs-asymmetric:

Asymmetric actor-critic
-----------------------

Multiple observation groups enable asymmetric actor-critic
architectures. The actor group contains only the observations that
would be available on real hardware; the critic group can include
privileged simulation state that is only accessible during training.

The velocity locomotion task uses this pattern. The actor group
receives noisy IMU readings and joint state; the critic group adds
noise-free height scan data and foot contact information. The
``enable_corruption`` flag makes this separation clean: actor terms
carry noise configs but the critic group disables them entirely.

.. code-block:: python

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,   # Noise active during training.
        ),
        "critic": ObservationGroupCfg(
            terms={**actor_terms, **privileged_terms},
            concatenate_terms=True,
            enable_corruption=False,  # No noise on critic.
        ),
    }

The training framework receives both groups. The policy network reads
``obs["actor"]`` at inference time; the value network reads
``obs["critic"]`` during training only.


Writing custom observation functions
--------------------------------------

An observation function accepts ``env`` as its first argument and
returns a ``[num_envs, D]`` tensor. Additional parameters are declared
as function arguments and supplied via
``ObservationTermCfg(params={...})``.

.. code-block:: python

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg


    def my_observation(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        robot = env.scene[asset_cfg.name]
        return robot.data.root_lin_vel_b

When a term needs to cache setup work or maintain per-episode state,
implement it as a class with ``__init__(self, cfg, env)`` and
``__call__(self, env, ...)``. If the class has a ``reset(env_ids)``
method, the manager calls it automatically on episode resets. See
:ref:`env-config-term-pattern` for the general pattern.
