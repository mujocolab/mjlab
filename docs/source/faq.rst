.. _faq:

FAQ & Troubleshooting
=====================

This page collects common questions about **platform support**, **performance**,
**training stability**, and **visualization**, along with practical debugging
tips and links to further resources.

Platform Support
----------------

Does it work on macOS?
   Yes, but only with limited performance. mjlab runs on macOS
   using **CPU-only** execution through MuJoCo Warp.

   - **Training is not recommended on macOS**, as it lacks GPU acceleration.
   - **Evaluation works**, but is significantly slower than on Linux with CUDA.

   For serious training workloads, we strongly recommend **Linux with an NVIDIA GPU**.

Does it work on Windows?
   We have performed preliminary testing on **Windows** and **WSL**, but some
   workflows are not guaranteed to be stable.

   - Windows support may **lag behind** Linux.
   - Windows will be **tested less frequently**, since Linux is the primary
     development and deployment platform.
   - Community contributions that improve Windows support are very welcome.

CUDA Compatibility
   Not all CUDA versions are supported by MuJoCo Warp.

   - See `mujoco_warp#101 <https://github.com/google-deepmind/mujoco_warp/issues/101>`_
     for details on CUDA compatibility.
   - **Recommended**: CUDA **12.4+** (for conditional execution support in CUDA
     graphs).

Performance
-----------

Is it faster than Isaac Lab?
   Based on our experience over the last few months, mjlab is **on par or
   faster** than Isaac Lab.

What GPU do you recommend?
   - **RTX 40-series GPUs** (or newer)
   - **L40s, H100**

Does mjlab support multi-GPU training?
   Yes, mjlab supports **multi-GPU distributed training** using
   `torchrunx <https://github.com/apoorvkh/torchrunx>`_.

   - Use ``--gpu-ids 0 1`` (or ``--gpu-ids all``) when running the ``train``
     command.
   - See the :doc:`distributed_training` for configuration details and examples.

Training & Debugging
--------------------

My training crashes with NaN errors
   A typical error when using ``rsl_rl`` looks like:

   .. code-block:: bash

      RuntimeError: normal expects all elements of std >= 0.0

   This occurs when NaN/Inf values in the **physics state** propagate to the
   policy network, causing its output standard deviation to become negative or NaN.

   There are many possible causes, including potential bugs in **MuJoCo Warp**
   (which is still in beta). mjlab offers two complementary mechanisms to help
   you handle this:

   1. **For training stability** - NaN termination

   Add a ``nan_detection`` termination to reset environments that hit NaN:

   .. code-block:: python

      from dataclasses import dataclass, field

      from mjlab.envs.mdp.terminations import nan_detection
      from mjlab.managers.termination_manager import TerminationTermCfg

      @dataclass
      class TerminationCfg:
         # Your other terminations...
         nan_term: TerminationTermCfg = field(
            default_factory=lambda: TerminationTermCfg(
                  func=nan_detection,
                  time_out=False,
            )
         )

   This marks NaN environments as terminated so they can reset while training
   continues. Terminations are logged as
   ``Episode_Termination/nan_term`` in your metrics.

   .. warning::

      This is a **band-aid solution**. If NaNs correlate with your task objective
      (for example, NaNs occur exactly when the agent tries to grasp an object),
      the policy will never learn to complete that part of the task. Always
      investigate the **root cause** using ``nan_guard`` in addition to this
      termination.

   2. **For debugging** - NaN guard

   Enable ``nan_guard`` to capture the simulation state when NaNs occur:

   .. code-block:: bash

      uv run train.py --enable-nan-guard True

   See the :doc:`NaN Guard documentation <nan_guard>` for details.

   The ``nan_guard`` tool makes it easier to:

   - Inspect the simulation state at the moment NaNs appear.
   - Build a minimal reproducible example (MRE).
   - Report potential framework bugs to the
     `MuJoCo Warp team <https://github.com/google-deepmind/mujoco_warp/issues>`_.

   Reporting well-isolated issues helps improve the framework for everyone.

Why aren't my training runs reproducible even with a fixed seed?
   MuJoCo Warp does not yet guarantee determinism, so running the same
   simulation with identical inputs may produce slightly different outputs.
   This is a known limitation being tracked in
   `mujoco_warp#562 <https://github.com/google-deepmind/mujoco_warp/issues/562>`_.

   Until determinism is implemented upstream, mjlab training runs will not be
   perfectly reproducible even when setting a seed.

Rendering & Visualization
-------------------------

What visualization options are available?
   mjlab currently supports two visualizers for policy evaluation and
   debugging:

   - **Native MuJoCo visualizer** - the built-in visualizer that ships with MuJoCo.
   - **Viser** - `Viser <https://github.com/nerfstudio-project/viser>`_,
     a web-based 3D visualization tool.

   We are exploring **training-time visualization** (e.g., live rollout viewers),
   but this is not yet available.

   As an alternative, mjlab supports **video logging to Weights & Biases
   (W&B)**, so you can monitor rollout videos directly in the experiment dashboard.

What about camera/pixel rendering for vision-based RL?
   Camera rendering for **pixel-based agents** is not yet available.

   The MuJoCo Warp team is actively developing **camera support**. Once mature, it
   will be integrated into mjlab for vision-based RL workflows.

How many environments can I visualize at once?
   Visualizers are **limited to 32 environments maximum** for performance reasons.

   - **Offscreen renderer** (for video recording): Hard-capped at 32 envs
     (see ``_MAX_ENVS`` in ``viewer/offscreen_renderer.py:12``)
   - **Native/Viser viewers**: Limited by MuJoCo's geometry buffer
     (default 10,000 geoms, configurable via ``max_geom`` parameter)

   With thousands of environments, only a subset will be rendered. The viewer
   shows whichever environments fit within the geometry budget.

Why are my fixed-base robots all stacked at the origin instead of in a grid?
   Fixed-base robots require an **explicit reset event** to position them at
   their ``env_origins``. If your robots appear stacked at (0, 0, 0):

   **Common causes:**

   1. **Missing reset event** - Most common issue.
   2. **env_spacing is 0 or very small** - Check your ``SceneCfg(env_spacing=...)``.
      Even with proper reset events, if ``env_spacing=0.0``, all robots will
      be at the same position. If ``env_spacing`` is very small (e.g., 0.01),
      they'll be clustered in a tiny area that looks like a line from a distance.

   **Solution**: Add a reset event that calls ``reset_root_state_uniform``:

   .. code-block:: python

      # In your ManagerBasedRlEnvCfg
      events = {
        # For positioning the base of the robot at env_origins.
        "reset_base": EventTermCfg(
          func=mdp.reset_root_state_uniform,
          mode="reset",
          params={
            "pose_range": {},  # Empty = use default pose + env_origins
            "velocity_range": {},
          },
        ),
        # ... other events
      }

   This pattern is used in the example manipulation task (see ``lift_cube_env_cfg.py:84-93``).

   **Why this is needed**: Fixed-base robots are automatically wrapped in mocap
   bodies by ``auto_wrap_fixed_base_mocap()``, but mocap positioning only happens
   when you explicitly call a reset event. The ``env_origins`` offset is applied
   inside ``reset_root_state_uniform()`` at line 127 of ``envs/mdp/events.py``.

   See `issue #560 <https://github.com/mujocolab/mjlab/issues/560>`_ for examples.

How does env_origins determine robot layout?
   Robot spacing depends on your terrain configuration:

   **Plane terrain** (``terrain_type="plane"``):
     - Creates an approximately square grid automatically
     - Grid size: ``ceil(sqrt(num_envs))`` rows x cols
     - Spacing controlled by ``env_spacing`` parameter (default: 2.0m)
     - Examples with ``env_spacing=2.0``:
       - 32 envs → 7x5 grid spanning 12m x 8m
       - 4096 envs → 64x64 grid spanning 126m x 126m
     - **Important**: If ``env_spacing=0``, all robots will be at (0, 0, 0)
     - Implementation: ``terrain_importer.py:_compute_env_origins_grid()``

   **Procedural terrain** (``terrain_type="generator"``):
     - Origins loaded from pre-generated terrain sub-patches
     - Grid size: ``TerrainGeneratorCfg.num_rows x num_cols``
     - Row index = difficulty level (curriculum mode)
     - Column index = terrain type variant
     - **Important allocation behavior**: Columns (terrain types) are evenly distributed
       across environments, but rows (difficulty levels) are randomly sampled. This means
       multiple environments can spawn on the same (row, col) patch, leaving others unoccupied,
       even when ``num_envs > num_patches``.
     - Example: 5x5 grid (25 patches), 100 envs → each column gets exactly 20 envs,
       but those 20 are randomly distributed across 5 rows, so some patches remain empty.
     - Supports ``randomize_env_origins()`` to shuffle positions during training

How do I ensure each terrain type gets its own column?
   Set ``curriculum=True`` in your ``TerrainGeneratorCfg``. This makes column
   allocation deterministic - each column gets one terrain type based on normalized
   proportions.

   Example with 2 terrain types:

   .. code-block:: python

      TerrainGeneratorCfg(
        num_rows=3,
        num_cols=2,
        curriculum=True,  # Required for deterministic column allocation!
        sub_terrains={
          "flat": BoxFlatTerrainCfg(proportion=0.5),  # Gets column 0
          "pillars": HfDiscreteObstaclesTerrainCfg(
            proportion=0.5,  # Gets column 1
          ),
        },
      )

   Without ``curriculum=True``, every patch is randomly sampled and you'll get
   a random mix of both terrain types scattered across all patches.

   **Note**: When ``num_cols`` equals the number of terrain types, each terrain
   gets exactly one column regardless of proportion values (they're normalized).
   When ``num_cols > num_terrain_types``, proportions determine how many columns
   each terrain type occupies.

What is flat patch sampling and how does it affect robot spawning?
   Flat patch sampling detects flat regions on heightfield terrains where robots
   can safely spawn. It uses morphological filtering on the heightfield to find
   circular areas where height variation is within a tolerance.

   Configure it on any sub-terrain via ``flat_patch_sampling``:

   .. code-block:: python

      from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg

      "obstacles": HfDiscreteObstaclesTerrainCfg(
        ...,
        flat_patch_sampling={
          "spawn": FlatPatchSamplingCfg(
            num_patches=10,      # patches to sample per sub-terrain
            patch_radius=0.5,    # flatness check radius (meters)
            max_height_diff=0.05,  # max height variation within radius
          ),
        },
      )

   Then use ``reset_root_state_from_flat_patches`` as your reset event to spawn
   robots on detected patches instead of at the sub-terrain center.

   **Key details:**

   - Only heightfield (``Hf*``) terrains support actual flat patch detection.
     Box terrains (``Box*``) don't have heightfield data to analyze.
   - If any sub-terrain in the grid configures ``flat_patch_sampling``, the
     flat patches array is allocated for **all** cells. Sub-terrains that don't
     produce patches have their slots filled with the sub-terrain's spawn origin,
     so ``reset_root_state_from_flat_patches`` always gets valid positions.
   - Without ``flat_patch_sampling``, use ``reset_root_state_uniform`` which
     spawns at the sub-terrain origin (``env_origins``) plus an optional random
     offset.

Development & Extensions
------------------------

Can I develop custom tasks in my own repository?
   Yes, mjlab has a **plugin system** that lets you develop tasks in separate
   repositories while still integrating seamlessly with the core:

   - Your tasks appear as regular entries for the ``train`` and ``play`` commands.
   - You can version and maintain your task repositories independently.

   A complete guide will be available in a future release.

Assets & Compatibility
----------------------

What robots are included?
   mjlab includes two **reference robots**:

   - **Unitree Go1** (quadruped).
   - **Unitree G1** (humanoid).

   These robots serve as:

   - Minimal examples for **robot integration**.
   - Stable, well-tested baselines for **benchmark tasks**.

   To keep the core library lean, we do **not** plan to aggressively expand the
   built-in robot library. Additional robots may be provided in separate
   repositories or community-maintained packages.

Can I use USD or URDF models?
   No, mjlab expects **MJCF (MuJoCo XML)** models.

   - You will need to **convert** USD or URDF assets to MJCF.
   - For many common robots, you can directly use
     `MuJoCo Menagerie <https://github.com/google-deepmind/mujoco_menagerie>`_,
     which ships high-quality MJCF models and assets.

Getting Help
------------

GitHub Issues
~~~~~~~~~~~~~

Use GitHub issues for:

- **Bug reports**
- **Performance regressions**
- **Documentation gaps**

When filing a bug, please include:

- CUDA driver and runtime versions
- GPU model
- A minimal reproduction script
- Complete error logs and stack traces
- Appropriate labels (for example: ``bug``, ``performance``, ``docs``)

`Open an issue <https://github.com/mujocolab/mjlab/issues>`_

Discussions
~~~~~~~~~~~

Use GitHub Discussions for:

- Usage questions (config, debugging, best practices)
- Performance tuning tips
- Asset conversion and modeling questions
- Design discussions and roadmap ideas

`Start a discussion <https://github.com/mujocolab/mjlab/discussions>`_

Known Limitations
-----------------

We're tracking missing features for the stable release in
https://github.com/mujocolab/mjlab/issues/100. Check our
`open issues <https://github.com/mujocolab/mjlab/issues>`_ to see what's actively
being worked on.

If something isn't working or if we've missed something, please
`file a bug report <https://github.com/mujocolab/mjlab/issues/new>`_.

.. important::

   mjlab has reached v1.0.0! While we strive for stability, please note that
   MuJoCo Warp (our physics backend) is still in beta. Feedback and
   contributions are very welcome.
