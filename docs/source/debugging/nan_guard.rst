.. _nan-guard:

NaN Guard
=========

The NaN guard captures simulation states when NaN/Inf is detected, helping
debug numerical instability issues.


Quick start
-----------

Enable the NaN guard with a single CLI flag:

.. code-block:: bash

    uv run train <task-name> --enable-nan-guard True

This automatically captures and saves simulation states when NaN/Inf is
detected. You can also enable it programmatically:

.. code-block:: python

    from mjlab.sim.sim import SimulationCfg
    from mjlab.utils.nan_guard import NanGuardCfg

    cfg = SimulationCfg(
        nan_guard=NanGuardCfg(
            enabled=True,
            buffer_size=100,
            output_dir="/tmp/mjlab/nan_dumps",
            max_envs_to_dump=5,
        ),
    )


Configuration
-------------

``enabled`` *(default: False)*
    Enable/disable NaN detection and dumping.

``buffer_size`` *(default: 100)*
    Number of recent simulation states to keep in the rolling buffer.

``output_dir`` *(default: "/tmp/mjlab/nan_dumps")*
    Directory where NaN dump files are saved.

``max_envs_to_dump`` *(default: 5)*
    Maximum number of NaN environments to dump to disk. All environments are
    tracked in the buffer, but only the first N are saved to reduce dump
    size.


Behavior
--------

- **Captures** ``mjSTATE_INTEGRATION`` before each physics step: time, positions,
  velocities, activations, delayed-input history, solver warmstart, controls,
  applied generalized/body forces, active equalities, mocap poses and user data.
  It also snapshots the then-current expanded model fields, including mesh and
  material variant assignments, and the actual Warp simulation options.
- **Detects** NaN/Inf in ``qpos``, ``qvel``, ``qacc``,
  ``qacc_warmstart``, and ``sensordata`` after each step
- **Dumps** the rolling buffer and model to disk on first detection
- **Stops** after the first dump to avoid spam

When disabled, all operations are no-ops with negligible overhead.
When enabled, the ring buffer keeps immutable device copies of model fields as
well as integration inputs for every world. Budget memory for both, especially
with large batches, many randomized fields or long buffers. Shared options are
stored once per frame. Capturing stops and the device buffer is released after
a successful dump.

As a representative memory budget, 512 Go1 worlds with body-mass randomization
and its derived fields plus ``geom_friction`` used 739,404 bytes of tensor storage
per frame on MuJoCo 3.17. A 100-frame buffer therefore projects to 70.5 MiB, plus
35.5 MiB for the private native model and assets. This excludes allocator and
Python overhead; sizes depend on the model and expanded fields.


Output format
-------------

Each NaN detection creates uniquely named files (timestamp, process ID and UUID)
plus latest symlinks, or copies when symlinks are unavailable:

- ``nan_dump_TIMESTAMP.npz``: compressed state buffer

  - ``states_step_NNNNNN``: captured states per step
    (shape: ``[num_envs_dumped, state_size]``)
  - ``_metadata``: dict with ``num_envs_total``, ``nan_env_ids``,
    ``dumped_env_ids``, ``format_version=2``, ``model_files`` and model checksums.
    ``model_files[state_key][dumped_environment_index]`` selects the MJB for that
    exact frame and world. ``warp_options`` records backend-specific settings for
    each revision in addition to native options stored in the MJB.

- ``model_TIMESTAMP.mjb`` and optional ``model_TIMESTAMP_rNNNNNN.mjb``: immutable
  MuJoCo model revisions. Repeated field/option snapshots share a revision.
- ``nan_dump_latest.npz``: symlink to most recent dump
- ``model_latest.mjb``: compatibility alias for the latest archive's first model

Models are completed before the unique NPZ is atomically published. A bounded
file lock serializes latest-alias publication across guards and ranks. The NPZ
and the model filenames in its metadata are authoritative: one model alias
cannot describe every world or historical frame in a randomized run. A failed
latest-alias update leaves the uniquely named complete archive available.
Its path is printed before attempting the aliases, so publication errors do not
hide a successfully committed archive.

Do not run ``mj_setConst`` on a restored model: this would overwrite the captured
derived constants. The model retains all compiled mesh assets and receives the
actual expanded field values for that frame/world. Changes made outside mjlab's
expanded-model-field mechanism are not tracked.

The archive contains trusted-source-only pickle metadata, as older NaN dumps do.
The viewer checks new archives' MJB checksums and continues to read older dumps
with their original single-model and state-specification semantics. Old dumps
cannot recover randomized model fields or inputs that were never recorded.


Visualizing dumps
-----------------

Use the interactive viewer to scrub through captured states:

.. code-block:: bash

    # View latest dump.
    uv run viz-nan /tmp/mjlab/nan_dumps/nan_dump_latest.npz

    # View a specific dump.
    uv run viz-nan /tmp/mjlab/nan_dumps/nan_dump_20251014_123456.npz


.. figure:: ../_static/content/nan_debug.gif
   :alt: NaN Debug Viewer

   NaN debug viewer.

The viewer provides:

- Step slider to scrub through the buffer
- Environment slider to compare different environments
- Info panel showing which environments have NaN/Inf
- 3D visualization of the robot and terrain at each state

The viewer selects the corresponding model revision when either slider changes,
rebuilding geometry and scene controls when necessary. Native replay preserves
the captured physics inputs; differences between native MuJoCo and Warp numeric
execution can still affect a subsequent simulation step. These are physics
diagnostics, rather than training checkpoints or policy/optimizer snapshots.
MuJoCo's ``mjSTATE_INTEGRATION`` does not include its separate sleeping-tree
counters; these archives do not promise to reproduce sleeping-engine scheduling.

For a headless inspection or native replay:

.. code-block:: python

    import mujoco
    from mjlab.utils.nan_replay import NanDump

    dump = NanDump("nan_dump_latest.npz")
    model, data = dump.restore(frame=0, environment=0)
    # environment is an index into metadata["dumped_env_ids"].
    mujoco.mj_step(model, data)


NaN detection termination
-------------------------

While the NaN guard helps **debug** NaN issues by capturing states, you can
also **prevent** training crashes using the ``nan_detection`` termination
term. This marks NaN environments as terminated, allowing them to reset
while training continues:

.. code-block:: python

    from mjlab.envs.mdp.terminations import nan_detection
    from mjlab.managers.termination_manager import TerminationTermCfg

    nan_term: TerminationTermCfg = field(
        default_factory=lambda: TerminationTermCfg(
            func=nan_detection,
            time_out=False,
        )
    )

Terminations are logged as ``Episode_Termination/nan_term`` in your metrics.

.. important::

   ``nan_detection`` is a band-aid, not a cure. If NaNs occur during your
   task objective (e.g., NaNs happen when grasping), the policy will never
   learn to complete the task since it resets before receiving rewards.
   Monitor your ``Episode_Termination/nan_term`` metrics carefully.

**When to use which:**

- ``nan_guard``: debug and understand why NaNs occur (always do this first)
- ``nan_detection``: keep training stable while working on a permanent fix
