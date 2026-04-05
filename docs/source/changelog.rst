=========
Changelog
=========

Upcoming version (not yet released)
-----------------------------------

Added
^^^^^

- Added :func:`~mjlab.envs.mdp.curriculums.termination_curriculum` for
  scheduling changes to termination term parameters during training,
  matching the existing ``reward_curriculum`` pattern. Both now share a
  single internal engine with init-time validation of stage ordering,
  field existence, and param keys.
- Added ``reduce`` field to ``MetricsTermCfg``. Setting ``reduce="last"``
  reports the value from the final step of the episode rather than the
  episode mean, which is useful for binary success metrics.
- Added :class:`~mjlab.envs.mdp.actions.RelativeJointPositionAction` for
  joint position control relative to the current configuration. The target is
  ``current_pos + action * scale``, so a zero action holds the current
  configuration rather than commanding the default pose.
- Added :func:`~mjlab.envs.mdp.dr.pair_friction` for randomizing geom-pair
  friction overrides (``pair_friction`` in ``mjModel``).
- Added ``STAIRS_TERRAINS_CFG`` terrain preset for progressive stair
  curriculum training and ``@terrain_preset`` decorator for composing
  terrain configurations from reusable presets.
- Added cartpole balance and swingup tasks (``Mjlab-Cartpole-Balance`` and
  ``Mjlab-Cartpole-Swingup``) with a :ref:`tutorial <tutorial-cartpole>`
  that walks through building an environment from scratch.
- Added :ref:`motion imitation <motion-imitation>` documentation with
  preprocessing instructions. The README now links here instead of the
  BeyondMimic repository, which produced incompatible NPZ files when used
  with mjlab (:issue:`777`).
- Added ``margin``, ``gap``, and ``solmix`` fields to ``CollisionCfg``
  for per geom contact parameter configuration (:issue:`766`).
- NaN guard now captures mocap body poses (``mocap_pos``, ``mocap_quat``)
  when the model has mocap bodies, enabling full state reconstruction in
  the dump viewer for fixed-base entities.
- Implemented ``ActionTermCfg.clip`` for clamping processed actions after
  scale and offset (:issue:`771`).
- Added ``qfrc_actuator`` and ``qfrc_external`` generalized force accessors
  to ``EntityData``. ``qfrc_actuator`` gives actuator forces in joint space
  (projected through the transmission). ``qfrc_external`` recovers the
  generalized force from body external wrenches (``xfrc_applied``)
  (:issue:`776`).
- Added ``RewardBarPanel`` to the Viser viewer, showing horizontal bars for
  each reward term with a running mean over ~1 second (:issue:`800`).
- Added ``per_substep`` flag to ``MetricsTermCfg`` for evaluating metrics
  once per physics substep inside the decimation loop. The per substep
  values are averaged within each environment step, so episode averages
  remain comparable to regular per step metrics.
- Added ``project-instinct/InstinctMJ`` to the research page's list of
  projects built on mjlab.
- Added a Checkpoints tab to the Viser play viewer for hot-swapping
  checkpoints without restarting. Works with local directories and W&B
  runs (:issue:`751`). Contribution by @omarrayyann.
- Added ``"segmentation"`` camera data type for per-pixel geom ID output
  alongside RGB and depth, and a multi-cube goal-conditioned lifting task
  (``Mjlab-Multi-Cube-Seg-Yam``) that uses it (:issue:`862`).
  Contribution by @pthangeda.

Changed
^^^^^^^

- Actuator delay is now configured inline on any ``ActuatorCfg`` subclass
  (e.g. ``BuiltinPositionActuatorCfg(..., delay_min_lag=2, delay_max_lag=5)``)
  instead of wrapping with ``DelayedActuatorCfg``. ``DelayedActuator``,
  ``DelayedActuatorCfg``, and ``DelayedBuiltinActuatorGroup`` are removed.
- Removed ``delay_target`` from ``ActuatorCfg``. Delay now always applies to
  the actuator's ``command_field`` automatically. Multi-target delay
  (``delay_target=("position", "velocity")``) is no longer supported.
- ``XmlPositionActuatorCfg``, ``XmlVelocityActuatorCfg``, ``XmlMotorActuatorCfg``,
  and ``XmlMuscleActuatorCfg`` are replaced by a single ``XmlActuatorCfg`` that auto
  detects the actuator type from XML. Pass ``command_field=...`` to override detection.
- Replaced the viser viewer internals with the ``mjviser`` package. Scene
  creation, mesh conversion, and overlay rendering (contacts, forces,
  inertia, tendons, joints, frames) are now provided by mjviser. The viewer
  exposes a new Visualization tab for overlay controls and a Groups tab for
  geom/site visibility. Debug visualization and warp tensor conversion remain
  in mjlab's ``MjlabViserScene`` subclass (:issue:`839`).
- In curriculum terrain mode, each terrain type now gets exactly one column
  (``num_cols`` is set to ``len(sub_terrains)``). The ``proportion`` field
  now controls robot spawning distribution across columns rather than column
  count. Random mode is unchanged (:issue:`811`).
- ``BoxSteppingStonesTerrainCfg`` stone size now decreases with difficulty,
  interpolating from the large end of ``stone_size_range`` at difficulty 0
  to the small end at difficulty 1 (:issue:`785`).
- Removed deprecated ``TerrainImporter`` and ``TerrainImporterCfg`` aliases.
  Use ``TerrainEntity`` and ``TerrainEntityCfg`` instead (:issue:`667`).
- ``Entity.clear_state()`` is deprecated. Use ``Entity.reset()`` instead.
  ``clear_state`` only zeroed actuator targets without resetting actuator
  internal state (e.g. delay buffers), which could cause stale commands
  after teleporting the robot to a new pose.
- Removed ``EntityData.generalized_force``. The property was bugged (indexed
  free joint DOFs instead of articulated DOFs) and the name was ambiguous.
  Use ``qfrc_actuator`` or ``qfrc_external`` instead (:issue:`776`).

Fixed
^^^^^

- Fixed ``SceneEntityCfg`` names and IDs ordering mismatch when
  ``preserve_order=False`` (:issue:`876`). Contribution by @jsw7460.
- Fixed ONNX export path resolution in the velocity, manipulation, and
  tracking runners when a parent directory name contains the word
  ``"model"`` (:issue:`867`). Contribution by @gokulp01.
- ``export-scene`` now writes only referenced assets and places them
  correctly under the output directory. Previously, asset keys containing
  path traversal could write files outside the output directory, and all
  spec assets were included regardless of whether the scene XML referenced
  them (:issue:`858`).
- ``electrical_power_cost`` now uses ``qfrc_actuator`` (joint space) instead
  of ``actuator_force`` (actuation space) for mechanical power computation.
  Previously the reward was incorrect for actuators with gear ratios other
  than 1 (:issue:`776`).
- ``create_velocity_actuator`` no longer sets ``ctrllimited=True`` with
  ``inheritrange=1.0``. This caused a ``ValueError`` for continuous joints
  (e.g. wheels) that have no position range defined (:issue:`787`).
- ``write_root_com_velocity_to_sim`` no longer fails with tensor ``env_ids``
  on floating base entities (:issue:`793`).
- Joint limits for unlimited joints are now set to [-inf, inf] instead of
  [0, 0]. Previously the zero range caused incorrect clamping for entities
  with unlimited hinge or slide joints.
- Contact force visualization now copies ``ctrl`` into the CPU ``MjData``
  before calling ``mj_forward``. Actuators that compute torques in Python
  (``DcMotorActuator``, ``IdealPdActuator``) previously showed incorrect
  contact forces because the viewer ran with ``ctrl=0``
  (:issue:`786`).
- ``BoxSteppingStonesTerrainCfg`` no longer creates a large gap around the
  platform. Stones are now only skipped when their center falls inside the
  platform; edges that extend under the platform are allowed since the
  platform covers them (:issue:`785`).
- ``dr.pseudo_inertia`` no longer loads cuSOLVER, eliminating ~4 GB of
  persistent GPU memory overhead. Cholesky and eigendecomposition are now
  computed analytically for the small matrices involved (4x4 and 3x3)
  (:issue:`753`).
- Set terrain geom mass to zero so that the static terrain body does not
  inflate ``stat.meanmass``, which made force arrow visualization invisible
  on rough terrain (:issue:`734`, :issue:`537`).
- Native viewer now syncs ``qpos0`` when domain randomized, fixing incorrect
  body positions after ``dr.joint_default_pos`` randomization
  (:issue:`760`).
- ``command_manager.compute()`` is now called during ``reset()`` so that
  derived command state (e.g. relative body positions in tracking
  environments) is populated before the first observation is returned
  (:issue:`761`).
- ``RayCastSensor`` with ``ray_alignment="yaw"`` or ``"world"`` now correctly
  aligns the frame offset when attached to a site or geom with a local offset
  from its parent body. Previously only ray directions and pattern offsets were
  aligned, causing the frame position to swing with body pitch/roll
  (:issue:`775`).

Version 1.2.0 (March 6, 2026)
-----------------------------

.. admonition:: Breaking API changes
   :class: attention

   - ``randomize_field`` no longer exists. Replace calls with typed functions
     from the new ``dr`` module (e.g. ``dr.geom_friction``, ``dr.body_mass``).
   - ``EventTermCfg`` no longer accepts ``domain_randomization``. The
     ``@requires_model_fields`` decorator on each ``dr`` function takes care
     of field expansion automatically.
   - ``Scene.to_zip()`` is deprecated. Use ``Scene.write(path, zip=True)``.
   - ``RslRlModelCfg`` no longer accepts ``stochastic``, ``init_noise_std``,
     or ``noise_std_type``. Use ``distribution_cfg`` instead
     (e.g. ``{"class_name": "GaussianDistribution", "init_std": 1.0,
     "std_type": "scalar"}``). Existing checkpoints are automatically
     migrated on load.

Added
^^^^^

- Added ``"step"`` event mode that fires every environment step.
- Added ``apply_body_impulse`` event for applying transient external wrenches
  to bodies with configurable duration and optional application point offset.
- ONNX auto-export and metadata attachment for manipulation tasks (lift cube)
  on every checkpoint save, matching the velocity and tracking task behavior.
- Multi-frame ``RayCastSensor``: pass a tuple of ``ObjRef`` to ``frame`` for
  per-site raycasting with independent body exclusion. New properties:
  ``num_frames``, ``num_rays_per_frame``. New ``RayCastData`` fields:
  ``frame_pos_w`` and ``frame_quat_w``.
- ``RingPatternCfg`` ray pattern for concentric ring sampling around each
  frame.
- ``TerrainHeightSensor``, a ``RayCastSensor`` subclass that computes
  per-frame vertical clearance above terrain (``sensor.data.heights``).
  Velocity task configs now use it for ``feet_clearance``,
  ``feet_swing_height``, and ``foot_height``, replacing the previous
  world-Z proxy that was incorrect on rough terrain.
- Cloud training support via `SkyPilot <https://skypilot.readthedocs.io/>`_
  and Lambda Cloud, with documentation covering setup, monitoring, and
  cost management.
- W&B hyperparameter sweep scripts that distribute one agent per GPU
  across a multi-GPU instance.
- Contributing guide with documentation for shared Claude Code commands
  (``/update-mjwarp``, ``/commit-push-pr``).
- Added optional ``ViewerConfig.fovy`` and apply it in native viewer camera
  setup when provided.
- Native viewer now tracks the first non-fixed body by default (matching
  the Viser viewer behavior introduced in
  ``716aaaa58ad7bfaf34d2f771549d461204d1b4ba``).
- New ``dr`` module (``mjlab.envs.mdp.dr``) replacing ``randomize_field``
  with typed per-field domain randomization functions. Each function
  automatically recomputes derived fields via ``set_const``. Highlights:

  - Camera and light randomization: ``dr.cam_fovy``, ``dr.cam_pos``,
    ``dr.cam_quat``, ``dr.cam_intrinsic``, ``dr.light_pos``,
    ``dr.light_dir``. Camera and light names are now supported in
    ``SceneEntityCfg`` (``camera_names`` / ``light_names``).
  - ``dr.pseudo_inertia`` for physics-consistent randomization of
    ``body_mass``, ``body_ipos``, ``body_inertia``, and ``body_iquat``
    via the pseudo-inertia matrix parameterization (Rucker & Wensing
    2022). Replaces the removed ``dr.body_inertia`` /
    ``dr.body_iquat``.
  - ``dr.geom_size`` with automatic recomputation of ``geom_rbound``
    and ``geom_aabb`` for broadphase consistency.
  - ``dr.tendon_armature`` and ``dr.tendon_frictionloss``.
  - ``dr.body_quat``, ``dr.geom_quat``, and ``dr.site_quat`` with RPY
    perturbation composed onto the default quaternion.
  - Extensible ``Operation`` and ``Distribution`` types. Users can define
    custom operations and distributions as class instances and pass them
    anywhere a string is accepted. Built-in instances (``dr.abs``,
    ``dr.scale``, ``dr.add``, ``dr.uniform``, ``dr.log_uniform``,
    ``dr.gaussian``) are exported from the ``dr`` module.
  - ``dr.mat_rgba`` for per-world material color randomization. Tints
    the texture color, useful for randomizing appearance of textured
    surfaces. Material names are now supported in ``SceneEntityCfg``
    (``material_names``).
  - Fixed ``dr.effort_limits`` drifting on repeated randomization.
  - Fixed ``dr.body_com_offset`` not triggering ``set_const``.

- ``export-scene`` CLI script to export any task scene or asset_zoo entity
  (``g1``, ``go1``, ``yam``) to a directory or zip archive for inspection
  and debugging.

- ``yam_lift_cube_vision_env_cfg`` now randomizes cube color (``dr.geom_rgba``)
  on every reset when ``cam_type="rgb"``.

- The native viewer now reflects per-world DR changes to visual model fields
  on each reset. Geom appearance, body and site poses, camera parameters,
  and light positions are all synced from the GPU model before rendering.
  Inertia boxes (press ``I``) and camera frustums (press ``Q``) update
  correctly when the corresponding fields are randomized. See
  :doc:`randomization` for viewer-specific caveats.

- ``MaterialCfg.geom_names_expr`` for assigning materials to geoms by
  name pattern during ``edit_spec``.

- ``TerrainEntityCfg`` now exposes ``textures``, ``materials``, and
  ``lights`` as configurable fields (previously hardcoded). Set
  ``textures=()``, ``materials=()`` to use flat ``dr.geom_rgba``
  instead of the default checker texture.

- ``DebugVisualizer`` now supports ellipsoid visualization via
  ``add_ellipsoid``.

- Interactive velocity joystick sliders in the Viser viewer. Enable the
  joystick under Commands/Twist to override velocity commands with manual
  sliders for ``lin_vel_x``, ``lin_vel_y``, and ``ang_vel_z``
  (`#666 <https://github.com/mujocolab/mjlab/issues/666>`_).
- Per-term debug visualization toggles in the Viser viewer. Individual
  command term visualizers (e.g. velocity arrows) can now be toggled
  independently under Scene/Debug Viz.
- Viewer single-step mode: press RIGHT arrow (native) or click "Step"
  (Viser) to advance exactly one physics step while paused.
- Viewer error recovery: exceptions during stepping now pause the viewer
  and log the traceback instead of crashing the process.
- Native viewer runs forward kinematics while paused, keeping
  perturbation visuals accurate.
- Viewer speed multipliers use clean power-of-2 fractions (1/32x to 1x).

- Visualizers display the realtime factor alongside FPS.

- ``joint_torques_l2`` now respects ``SceneEntityCfg.actuator_ids``,
  allowing penalization of a subset of actuators instead of all of them
  (`#703 <https://github.com/mujocolab/mjlab/pull/703>`_). Contribution by
  `@saikishor <https://github.com/saikishor>`_.

- Terrain is now a proper ``Entity`` subclass (``TerrainEntity``). This
  allows domain randomization functions to target terrain parameters
  (friction, cameras, lights) via ``SceneEntityCfg("terrain", ...)``.
  ``TerrainImporter`` / ``TerrainImporterCfg`` remain as aliases but will be
  deprecated in a future version.
- Added ``upload_model`` option to ``RslRlBaseRunnerCfg`` to control W&B model
  file uploads (``.pt`` and ``.onnx``) while keeping metric logging enabled
  (`#654 <https://github.com/mujocolab/mjlab/pull/654>`_).
- ``Scene.write(output_dir, zip=False)`` exports the scene XML and mesh
  assets to a directory (or zip archive). Replaces ``Scene.to_zip()``.
- ``Entity.write_xml()`` and ``Scene.write()`` now apply XML fixups
  (empty defaults, duplicate nested defaults) and strip buffer textures
  that ``MjSpec.to_xml()`` cannot serialize.
- ``fix_spec_xml`` and ``strip_buffer_textures`` utilities in
  ``mjlab.utils.xml``.

Changed
^^^^^^^

- Native viewer now syncs ``xfrc_applied`` to the render buffer and draws
  arrows for any nonzero applied forces. Mouse perturbation forces are
  converted to ``qfrc_applied`` (generalized joint space) so they coexist
  with programmatic forces on ``xfrc_applied`` without conflict.
- ``ViewerConfig.OriginType.WORLD`` now configures a free camera at the
  specified lookat point instead of auto tracking a body. A new ``AUTO``
  origin type (now the default) preserves the previous auto tracking
  behavior.
- Upgraded ``rsl-rl-lib`` from 4.0.1 to 5.0.1. ``RslRlModelCfg`` now
  uses ``distribution_cfg`` dict instead of ``stochastic`` /
  ``init_noise_std`` / ``noise_std_type``. Existing checkpoints are
  automatically migrated on load.
- Reorganized the Viser Controls tab into a cleaner folder hierarchy:
  Info, Simulation, Commands, Scene (with Environment, Camera, Debug Viz,
  Contacts sub-folders), and Camera Feeds. The Environment folder is
  hidden for single-env tasks and the Commands folder is hidden when no
  command terms are active.
- Viser camera tracking is now enabled by default so the agent stays in
  frame on launch.
- Self collision and illegal contact sensors now use ``history_length`` to
  catch contacts across decimation substeps. Reward and termination functions
  read ``force_history`` with a configurable ``force_threshold``.
- Replaced the single ``scale`` parameter in ``DifferentialIKActionCfg`` with
  separate ``delta_pos_scale`` and ``delta_ori_scale`` for independent scaling
  of position and orientation components.
- Improved offscreen multi environment framing by selecting neighboring
  environments around the focused env instead of first N envs.
- Tuned tracking task viewer defaults for tighter camera framing.
- Disabled shadow casting on the G1 tracking light to avoid duplicate
  stacked shadows when robots are close.

Fixed
^^^^^

- Fixed actuator target resolution for entities whose ``spec_fn`` uses
  internal ``MjSpec.attach(prefix=...)``
  (`#709 <https://github.com/mujocolab/mjlab/issues/709>`_).
- Fixed viewer physics loop starving the renderer by replacing the single
  sim-time budget with a two-clock design (tracked vs actual sim time).
  Physics now self-corrects after overshooting, keeping FPS smooth at all
  speed multipliers.
- Bundled ``ffmpeg`` for ``mediapy`` via ``imageio-ffmpeg``, removing the
  requirement for a system ``ffmpeg`` install. Thanks to
  `@rdeits-bd <https://github.com/rdeits-bd>`_ for the suggestion.
- Fixed ``height_scan`` returning ~0 for missed rays; now defaults to
  ``max_distance``. Replaced ``clip=(-1, 1)`` with ``scale`` normalization
  in the velocity task config. Thanks to `@eufrizz <https://github.com/eufrizz>`_
  for reporting and the initial fix (`#642 <https://github.com/mujocolab/mjlab/pull/642>`_).
- Fixed ghost mesh visualization for fixed-base entities by extending
  ``DebugVisualizer.add_ghost_mesh`` to optionally accept ``mocap_pos`` and
  ``mocap_quat`` (`#645 <https://github.com/mujocolab/mjlab/pull/645>`_).
- Fixed viser viewer crashing on scenes with no mocap bodies by adding
  an ``nmocap`` guard, matching the native viewer behavior.
- Fixed offscreen rendering artifacts in large vectorized scenes by applying
  a render local extent override in ``OffscreenRenderer`` and restoring the
  original extent on close.
- Fixed ``RslRlVecEnvWrapper.unwrapped`` to return the base environment,
  ensuring checkpoint state restore and logging work correctly when wrappers
  such as ``VideoRecorder`` are enabled.

Version 1.1.1 (February 14, 2026)
---------------------------------

Added
^^^^^

- Added reward term visualization to the native viewer (toggle with ``P``) (`#629 <https://github.com/mujocolab/mjlab/pull/629>`_).
- Added ``DifferentialIKAction`` for task-space control via damped
  least-squares IK. Supports weighted position/orientation tracking,
  soft joint-limit avoidance, and null-space posture regularization.
  Includes an interactive viser demo (``scripts/demos/differential_ik.py``) (`#632 <https://github.com/mujocolab/mjlab/pull/632>`_).

Fixed
^^^^^

- Fixed ``play.py`` defaulting to the base rsl-rl ``OnPolicyRunner`` instead
  of ``MjlabOnPolicyRunner``, which caused a ``TypeError`` from an unexpected
  ``cnn_cfg`` keyword argument (`#626 <https://github.com/mujocolab/mjlab/pull/626>`_). Contribution by
  `@griffinaddison <https://github.com/griffinaddison>`_.

Changed
^^^^^^^

- Removed ``body_mass``, ``body_inertia``, ``body_pos``, and ``body_quat``
  from ``FIELD_SPECS`` in domain randomization. These fields have derived
  quantities that require ``set_const`` to recompute; without that call,
  randomizing them silently breaks physics (`#631 <https://github.com/mujocolab/mjlab/pull/631>`_).
- Replaced ``moviepy`` with ``mediapy`` for video recording. ``mediapy``
  handles cloud storage paths (GCS, S3) natively (`#637 <https://github.com/mujocolab/mjlab/pull/637>`_).

.. figure:: _static/changelog/native_reward.png
   :width: 80%

Version 1.1.0 (February 12, 2026)
---------------------------------

Added
^^^^^

- Added RGB and depth camera sensors and BVH-accelerated raycasting (`#597 <https://github.com/mujocolab/mjlab/pull/597>`_).
- Added ``MetricsManager`` for logging custom metrics during training (`#596 <https://github.com/mujocolab/mjlab/pull/596>`_).
- Added terrain visualizer (`#609 <https://github.com/mujocolab/mjlab/pull/609>`_). Contribution by
  `@mktk1117 <https://github.com/mktk1117>`_.

.. figure:: _static/changelog/terrain_visualizer.jpg
   :width: 80%

- Added many new terrains including ``HfDiscreteObstaclesTerrainCfg``,
  ``HfPerlinNoiseTerrainCfg``, ``BoxSteppingStonesTerrainCfg``,
  ``BoxNarrowBeamsTerrainCfg``, ``BoxRandomStairsTerrainCfg``, and
  more. Added flat patch sampling for heightfield terrains (`#542 <https://github.com/mujocolab/mjlab/pull/542>`_, `#581 <https://github.com/mujocolab/mjlab/pull/581>`_).
- Added site group visualization to the Viser viewer (Geoms and Sites
  tabs unified into a single Groups tab) (`#551 <https://github.com/mujocolab/mjlab/pull/551>`_).
- Added ``env_ids`` parameter to ``Entity.write_ctrl_to_sim`` (`#567 <https://github.com/mujocolab/mjlab/pull/567>`_).

Changed
^^^^^^^

- Upgraded ``rsl-rl-lib`` to 4.0.0 and replaced the custom ONNX
  exporter with rsl-rl's built-in ``as_onnx()`` (`#589 <https://github.com/mujocolab/mjlab/pull/589>`_, `#595 <https://github.com/mujocolab/mjlab/pull/595>`_).
- ``sim.forward()`` is now called unconditionally after the decimation
  loop. See :ref:`faq-sim-forward` for details (`#591 <https://github.com/mujocolab/mjlab/pull/591>`_).
- Unnamed freejoints are now automatically named to prevent
  ``KeyError`` during entity init (`#545 <https://github.com/mujocolab/mjlab/pull/545>`_).

Fixed
^^^^^

- Fixed ``randomize_pd_gains`` crash with ``num_envs > 1`` (`#564 <https://github.com/mujocolab/mjlab/pull/564>`_).
- Fixed ``ctrl_ids`` index error with multiple actuated entities (`#573 <https://github.com/mujocolab/mjlab/pull/573>`_).
  Reported by `@bwrooney82 <https://github.com/bwrooney82>`_.
- Fixed Viser viewer rendering textured robots as gray (`#544 <https://github.com/mujocolab/mjlab/pull/544>`_).
- Fixed Viser plane rendering ignoring MuJoCo size parameter (`#540 <https://github.com/mujocolab/mjlab/pull/540>`_).
- Fixed ``HfDiscreteObstaclesTerrainCfg`` spawn height (`#552 <https://github.com/mujocolab/mjlab/pull/552>`_).
- Fixed ``RaycastSensor`` visualization ignoring the all-envs toggle (`#607 <https://github.com/mujocolab/mjlab/pull/607>`_).
  Contribution by `@oxkitsune <https://github.com/oxkitsune>`_.

Version 1.0.0 (January 28, 2026)
--------------------------------

Initial release of mjlab.
