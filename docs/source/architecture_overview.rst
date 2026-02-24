.. _architecture_overview:

Architecture Overview
=====================

mjlab is organized into two layers: a **simulation layer** that models
the robot and world, and a **manager layer** that defines the
reinforcement learning problem on top of it. Understanding this separation
is the fastest way to build a mental map of the system.

.. figure:: _static/architecture_diagram.png
   :width: 60%
   :align: center
   :alt: mjlab architecture diagram

   Entities are composed into an MjSpec, compiled, and transferred to
   MuJoCo Warp for GPU simulation. The ManagerBasedRlEnv orchestrates the
   MDP; RSL-RL handles training.


The simulation layer
--------------------

**Scene pipeline.**
mjlab constructs scenes by composing entity descriptions into a single
`MjSpec <https://mujoco.readthedocs.io/en/stable/programming/modeledit.html>`_.
Each entity starts from an
`MJCF <https://mujoco.readthedocs.io/en/latest/modeling.html>`_ file
loaded via ``MjSpec.from_file()``. Users who define everything in XML can
use this directly. For more control, Python dataclasses can extend or
override properties on the loaded spec: actuators, collision rules,
materials, sensors, and initial state. This hybrid approach lets users
start from existing MuJoCo models and layer on task-specific configuration
without modifying the original XML. The composed specification is compiled
into an ``MjModel`` on the CPU, then transferred to the GPU via
`MuJoCo Warp <https://mujoco.readthedocs.io/en/stable/mjwarp/index.html>`_,
which is built on `NVIDIA Warp <https://nvidia.github.io/warp/>`_.

**MuJoCo Warp.**
MuJoCo Warp is a GPU-accelerated backend for MuJoCo. It preserves
MuJoCo's ``MjModel``/``MjData`` paradigm but adds a leading *world*
dimension: a single ``MjData`` object holds the state of N independent
simulation instances in parallel, enabling thousands of environments to
be stepped simultaneously. Model parameters are shared across all worlds
by default, and individual fields can be expanded to vary per-world when
domain randomization requires it. mjlab captures the simulation step as a
`CUDA graph <https://developer.nvidia.com/blog/cuda-graphs>`_: the kernel
execution sequence is recorded once and replayed on subsequent calls,
eliminating CPU-side dispatch overhead.

.. note::

   CUDA graph capture is a one-time cost at environment startup. Per-episode
   resets and domain randomization events run as regular Python between graph
   replays and do not break the capture.

**Components.**
The simulation layer provides four core components, each with its own
documentation page:

- :ref:`entity`: a robot, a manipulated object, or a static fixture in
  the scene, defined by an MJCF description plus optional Python
  configuration for actuators, collision rules, and initial state.
- :ref:`actuators`: how entities are controlled. Users can wrap actuators
  already defined in MJCF or create new ones from Python configuration.
- :ref:`sensors`: how the world is observed. Includes MuJoCo-native
  sensors as well as custom sensors like RGB-D cameras and raycasters.
- :ref:`scene`: scene composition and environment placement.
- :ref:`terrain`: procedural terrain generation and curriculum learning.


The manager layer
-----------------

On top of the simulation layer, mjlab adopts the manager-based environment
design introduced by Isaac Lab. Users define their environment by composing
small, self-contained *terms* (reward functions, observation computations,
domain randomization events) and register them with the appropriate manager.
Each manager handles the lifecycle of its terms: calling them at the right
point in the simulation loop, aggregating their outputs, and exposing
diagnostics.

Terms can be plain functions for stateless computations, or classes that
inherit from ``ManagerTermBase`` when they need to cache expensive setup
(such as resolving regex patterns to joint indices at initialization) or
maintain per-episode state through a ``reset()`` hook.

Environments are configured through ``ManagerBasedRlEnvCfg``, a plain
dataclass that holds term configuration dictionaries for each manager.

.. code-block:: python

    from mjlab.envs import ManagerBasedRlEnvCfg

    cfg = ManagerBasedRlEnvCfg(
        decimation=4,           # 4 physics steps per policy step
        episode_length_s=20.0,
        scene=...,              # SceneCfg: terrain, entities, sensors
        sim=...,                # SimulationCfg: timestep, solver, integrator
        observations={...},     # ObservationManager terms
        actions={...},          # ActionManager terms
        rewards={...},          # RewardManager terms
        terminations={...},     # TerminationManager terms
        events={...},           # EventManager terms (resets, DR)
        commands={...},         # CommandManager terms (velocity targets, etc.)
        curriculum={...},       # CurriculumManager terms
        metrics={...},          # MetricsManager terms
    )

.. rubric:: The eight managers

- **ObservationManager**: assembles observation groups with configurable
  processing (clipping, noise, delay, history). Supports asymmetric
  actor-critic. See :ref:`observations`.
- **ActionManager**: routes the policy's output tensor to entity actuators,
  handling scaling and offset. See :ref:`actions`.
- **RewardManager**: computes a weighted sum of reward terms, scaled by step
  duration for frequency invariance. See :ref:`rewards`.
- **TerminationManager**: evaluates stop conditions, distinguishing terminal
  resets from timeouts. See :ref:`terminations`.
- **EventManager**: fires terms at lifecycle points (startup, reset,
  interval). Domain randomization is implemented through event terms.
  See :ref:`events` and :ref:`domain_randomization`.
- **CommandManager**: generates and resamples goal signals (velocity
  targets, pose targets). See :ref:`commands`.
- **CurriculumManager**: adjusts training conditions based on policy
  performance. See :ref:`curriculum`.
- **MetricsManager**: logs custom per-step values as episode averages.
  See :ref:`metrics`.

For the full configuration reference covering all managers, see
:ref:`environment_config`.


The environment lifecycle
-------------------------

Each environment instance passes through four phases.

1. **Build.** ``Scene`` composes entity MJCF files via ``MjSpec`` and
   compiles ``MjModel`` on the CPU. ``Simulation`` uploads the model to the
   GPU via MuJoCo Warp, allocating a single ``MjData`` with N parallel
   worlds. CUDA graphs for ``step``, ``forward``, ``reset``, and ``sense``
   are captured.

2. **Initialize.** Managers are constructed from the term configuration
   dictionaries. Regex patterns are matched to joint, body, and geom
   indices. Observation history and delay buffers are allocated. Model
   fields required by domain randomization terms are expanded from shared
   to per-world storage, and CUDA graphs are rebuilt to reflect the new
   layout. Startup events are fired once.

3. **Reset.** Called at the start of training and whenever an environment
   terminates or times out. The ``EventManager`` fires ``reset`` terms,
   which return the scene to an initial state with optional randomization.
   Command targets are resampled. Observation history buffers are cleared.

4. **Step.** The policy action is processed by the ``ActionManager``. The
   physics simulation advances ``decimation`` times, with actuator commands
   applied and entity state updated each sub-step. After the decimation
   loop, the ``TerminationManager`` checks stop conditions, the
   ``RewardManager`` computes the reward signal, and any terminated
   environments are reset. A single ``forward()`` call refreshes derived
   quantities for all environments. The ``CommandManager`` advances or
   resamples goals. Interval events fire if scheduled. Sensors update. The
   ``ObservationManager`` assembles the observation for the next policy
   query.

The step sequence in order:

.. code-block:: text

    action_manager.process_action(action)
    for _ in range(decimation):
        action_manager.apply_action()
        sim.step()
        scene.update()
    termination_manager.compute()
    reward_manager.compute()
    metrics_manager.compute()
    [reset terminated envs]
    sim.forward()
    command_manager.compute()
    event_manager.apply(mode="interval")
    sim.sense()
    observation_manager.compute()

With this mental model in place, the Concepts pages cover each simulation
layer component in detail, and The Manager Layer pages walk through each
manager's configuration and built-in terms. If you are coming from Isaac
Lab, :ref:`migration_isaac_lab` describes the key API differences.
