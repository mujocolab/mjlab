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
      from mjlab.managers.manager_term_config import TerminationTermCfg

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

   mjlab is in **beta**. Breaking changes, missing features, and rough edges
   are expected. Feedback and contributions are very welcome — they directly
   shape the stable release.
