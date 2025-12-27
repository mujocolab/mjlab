.. _changelog:

Changelog
=========

All notable changes to mjlab are documented here.

We follow `Semantic Versioning <https://semver.org/>`_. Given a version number
``MAJOR.MINOR.PATCH``:

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

.. note::

   ``mujoco-warp`` is not yet available on PyPI. Until then, you must install it
   from GitHub. See the :ref:`installation` guide for details.

----

v1.0.0 (December 2025)
----------------------

**First stable release.**

This release marks mjlab's transition from beta to production-ready. The API is
now stable and ready for use in research and development.

Highlights
^^^^^^^^^^

- **Stable API**: The manager-based architecture and core abstractions are now
  finalized and documented.
- **Comprehensive documentation**: Full API reference, installation guides, and
  core concept explanations.
- **Multi-GPU training**: Support for distributed training across multiple GPUs
  using ``torchrunx``.
- **Domain randomization**: Flexible randomization system for sim-to-real transfer.
- **NaN debugging tools**: Built-in utilities for detecting and debugging numerical
  instabilities.

For migration from Isaac Lab, see :ref:`migration_isaac_lab`.

----

v0.1.0 (Initial Release)
------------------------

Initial beta release of mjlab.

- Core manager-based architecture
- Integration with MuJoCo Warp for GPU-accelerated physics
- Velocity tracking and motion imitation tasks
- Support for Unitree G1 and Go1 robots
- Interactive viewer with Viser
- Weights & Biases integration for experiment tracking
