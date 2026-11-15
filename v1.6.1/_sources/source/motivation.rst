.. _motivation:

Why mjlab?
==========

Reinforcement learning has become a powerful tool for training robot
controllers in simulation and transferring them to real hardware. The
fidelity of this pipeline hinges on getting simulation details right.

Several frameworks address this.
`Isaac Lab <https://github.com/isaac-sim/IsaacLab>`_
provides a comprehensive manager-based API for composing RL environments,
but requires the Omniverse runtime, which adds installation complexity and
startup latency.
`MuJoCo Playground <https://playground.mujoco.org/>`_ takes the opposite
approach: minimal
abstractions and monolithic environment definitions that are easy to hack
and quick to prototype, but code duplication across robots and tasks makes
multi-task codebases difficult to maintain. There remains a gap for a
framework that is both lightweight and built on a proven orchestration API
with access to best-in-class physics.

mjlab fills this gap. It adopts Isaac Lab's manager-based design, where
users compose self-contained building blocks for observations, rewards,
events, and commands, and pairs it with MuJoCo Warp for GPU-accelerated
physics simulation. The result is a framework with minimal dependencies,
fast startup, direct access to native MuJoCo model and data structures,
and a PyTorch-native interface for policy training.


Design philosophy
-----------------

mjlab is designed around three core engineering commitments:

1. **Minimal installation friction.** A single
   ``uvx --from mjlab --refresh demo`` command is enough to run the
   framework. No heavyweight runtimes, no multi-gigabyte downloads. The
   dependency footprint is kept intentionally small.

2. **Transparent and inspectable physics.** mjlab targets a single physics
   stack, MuJoCo Warp, to prioritize simulation transparency and
   debuggability. The framework exposes MuJoCo-native ``MjModel`` and
   ``MjData`` structures for direct inspection and state access.
   Cross-simulator portability is a non-goal; mjlab favors precise control
   and interpretability over backend generality.

3. **Tight MuJoCo ecosystem integration.** Users work directly with MuJoCo
   models and conventions. MJCF files, MuJoCo Menagerie assets, and
   standard MuJoCo tooling all work without translation layers.


Scope
-----

mjlab provides infrastructure for rigid-body robot learning. It includes
depth and raycast sensors for geometric perception. High-fidelity RGB
rendering is out of scope. This does not preclude vision-based policies:
a common approach is to train privileged policies using full state, then
distill into vision-based controllers using external rendering.

The framework is intended to be extended to custom robots, tasks, sensors,
and actuators. It ships with reference implementations of velocity tracking,
motion imitation, and manipulation tasks.


Comparison
----------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Framework
     - Strengths
     - Best for
   * - **mjlab**
     - Lightweight, fast iteration, native MuJoCo, PyTorch
     - MuJoCo users who want structured RL environments with GPU acceleration
   * - **Isaac Lab**
     - Photorealistic rendering, USD pipeline, Omniverse ecosystem
     - Projects that need Isaac Sim capabilities
   * - **MuJoCo Playground**
     - Minimal abstractions, easy to hack, quick prototyping
     - One-off experiments and rapid iteration on single tasks
   * - **Newton**
     - Multi-physics solvers (deformables, VBD), differentiable simulation
     - Projects that need solver flexibility beyond rigid-body MuJoCo
