=========
Changelog
=========

Upcoming version (not yet released)
-----------------------------------

Changed
^^^^^^^

- Replaced the single ``scale`` parameter in ``DifferentialIKActionCfg`` with
  separate ``delta_pos_scale`` and ``delta_ori_scale`` for independent scaling
  of position and orientation components.

Version 1.1.1 (February 14, 2026)
---------------------------------

Added
^^^^^

- Added reward term visualization to the native viewer (toggle with ``P``).
- Added ``DifferentialIKAction`` for task-space control via damped
  least-squares IK. Supports weighted position/orientation tracking,
  soft joint-limit avoidance, and null-space posture regularization.
  Includes an interactive viser demo (``scripts/demos/differential_ik.py``).

Fixed
^^^^^

- Fixed ``play.py`` defaulting to the base rsl-rl ``OnPolicyRunner`` instead
  of ``MjlabOnPolicyRunner``, which caused a ``TypeError`` from an unexpected
  ``cnn_cfg`` keyword argument. Contribution by
  `@griffinaddison <https://github.com/griffinaddison>`_.

Changed
^^^^^^^

- Removed ``body_mass``, ``body_inertia``, ``body_pos``, and ``body_quat``
  from ``FIELD_SPECS`` in domain randomization. These fields have derived
  quantities that require ``set_const`` to recompute; without that call,
  randomizing them silently breaks physics.
- Replaced ``moviepy`` with ``mediapy`` for video recording. ``mediapy``
  handles cloud storage paths (GCS, S3) natively.

.. figure:: _static/changelog/native_reward.png
   :width: 80%

Version 1.1.0 (February 12, 2026)
---------------------------------

Added
^^^^^

- Added RGB and depth camera sensors and BVH-accelerated raycasting.
- Added ``MetricsManager`` for logging custom metrics during training.
- Added terrain visualizer. Contribution by
  `@mktk1117 <https://github.com/mktk1117>`_.

.. figure:: _static/changelog/terrain_visualizer.jpg
   :width: 80%

- Added many new terrains including ``HfDiscreteObstaclesTerrainCfg``,
  ``HfPerlinNoiseTerrainCfg``, ``BoxSteppingStonesTerrainCfg``,
  ``BoxNarrowBeamsTerrainCfg``, ``BoxRandomStairsTerrainCfg``, and
  more. Added flat patch sampling for heightfield terrains.
- Added site group visualization to the Viser viewer (Geoms and Sites
  tabs unified into a single Groups tab).
- Added ``env_ids`` parameter to ``Entity.write_ctrl_to_sim``.

Changed
^^^^^^^

- Upgraded ``rsl-rl-lib`` to 4.0.0 and replaced the custom ONNX
  exporter with rsl-rl's built-in ``as_onnx()``.
- ``sim.forward()`` is now called unconditionally after the decimation
  loop. See :ref:`faq-sim-forward` for details.
- Unnamed freejoints are now automatically named to prevent
  ``KeyError`` during entity init.

Fixed
^^^^^

- Fixed ``randomize_pd_gains`` crash with ``num_envs > 1``.
- Fixed ``ctrl_ids`` index error with multiple actuated entities.
  Reported by `@bwrooney82 <https://github.com/bwrooney82>`_.
- Fixed Viser viewer rendering textured robots as gray.
- Fixed Viser plane rendering ignoring MuJoCo size parameter.
- Fixed ``HfDiscreteObstaclesTerrainCfg`` spawn height.
- Fixed ``RaycastSensor`` visualization ignoring the all-envs toggle.
  Contribution by `@oxkitsune <https://github.com/oxkitsune>`_.

Version 1.0.0 (January 28, 2026)
--------------------------------

Initial release of mjlab.
