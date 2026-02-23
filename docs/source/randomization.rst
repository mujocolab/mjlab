Domain Randomization
====================

Domain randomization varies physical parameters during training so that policies
are robust to modeling errors and real-world variation. This guide shows
how to attach randomization terms to your environment using ``EventTermCfg`` and
the ``dr`` module.

Quick Start
-----------

Use an ``EventTermCfg`` that calls a typed function from ``dr`` with a
**value range** and an **operation** describing how to apply the draw.

.. code-block:: python

    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    foot_friction: EventTermCfg = EventTermCfg(
        mode="reset",  # randomize each episode
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=[".*_foot.*"]),
            "ranges": (0.3, 1.2),
            "operation": "abs",
        },
    )

Each ``dr`` function is decorated with ``@requires_model_fields`` which
automatically tracks the fields that need to be expanded for per-world
storage **and** the ``RecomputeLevel`` needed to keep
:ref:`derived quantities <dr-recomputation>` consistent.

The ``mode`` parameter controls when the event fires:

* ``"startup"`` randomizes once at initialization
* ``"reset"`` randomizes at every episode reset
* ``"interval"`` randomizes at regular time intervals


Available Functions
-------------------

Model field functions
^^^^^^^^^^^^^^^^^^^^^

Each function writes to a single field on ``sim.model`` (the MuJoCo Warp
model). For example, ``dr.geom_friction`` writes to ``sim.model.geom_friction``,
``dr.body_mass`` writes to ``sim.model.body_mass``, and so on. Most share the
signature ``(env, env_ids, ranges, ...)``, with ``distribution`` and
``operation`` controlling sampling and application (see :ref:`dr-parameters`).
Some functions have a more readable name than the underlying MuJoCo field; in
those cases the raw field name is available as an alias (``dr.body_com_offset``
and ``dr.body_ipos`` are the same function).

.. rubric:: Geom fields

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - Function
     - MuJoCo field
     - Description
     - Notes
   * - ``dr.geom_friction``
     - ``geom_friction``
     - Sliding, torsional, and rolling friction coefficients
     - Default axis: 0 (tangential only)
   * - ``dr.geom_pos``
     - ``geom_pos``
     - Position of the geom in the parent body frame
     -
   * - ``dr.geom_quat``
     - ``geom_quat``
     - Orientation of the geom frame
     - Accepts roll/pitch/yaw ranges (radians); composes with default
   * - ``dr.geom_rgba``
     - ``geom_rgba``
     - Color and transparency (RGBA)
     -
   * - ``dr.geom_size``
     - ``geom_size``
     - Geom-specific size parameters (radius, half-lengths, etc.)
     - Automatically recomputes ``geom_rbound`` and ``geom_aabb``

.. rubric:: Body fields

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - Function
     - MuJoCo field
     - Description
     - Notes
   * - ``dr.body_mass``
     - ``body_mass``
     - Mass of the body
     - Triggers ``set_const`` recomputation
   * - ``dr.body_com_offset`` (alias ``body_ipos``)
     - ``body_ipos``
     - Center of mass position relative to the body frame
     - Triggers ``set_const``
   * - ``dr.body_pos``
     - ``body_pos``
     - Position of the body frame in the parent frame
     - Triggers ``set_const_0``
   * - ``dr.body_quat``
     - ``body_quat``
     - Orientation of the body frame
     - Accepts roll/pitch/yaw ranges (radians); composes with default;
       triggers ``set_const_0``

.. rubric:: Joint fields

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - Function
     - MuJoCo field
     - Description
     - Notes
   * - ``dr.joint_damping`` (alias ``dof_damping``)
     - ``dof_damping``
     - Velocity-proportional damping force (passive)
     -
   * - ``dr.joint_armature`` (alias ``dof_armature``)
     - ``dof_armature``
     - Added rotor inertia (models geared transmissions)
     - Triggers ``set_const_0``
   * - ``dr.joint_friction`` (alias ``dof_frictionloss``)
     - ``dof_frictionloss``
     - Dry friction loss in the joint
     -
   * - ``dr.joint_stiffness`` (alias ``jnt_stiffness``)
     - ``jnt_stiffness``
     - Spring stiffness pulling toward the reference position
     -
   * - ``dr.joint_limits`` (alias ``jnt_range``)
     - ``jnt_range``
     - Lower and upper joint position limits
     -
   * - ``dr.joint_default_pos`` (alias ``qpos0``)
     - ``qpos0``
     - Reference joint position (zero-spring equilibrium)
     - Triggers ``set_const_0``

.. rubric:: Site fields

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - Function
     - MuJoCo field
     - Description
     - Notes
   * - ``dr.site_pos``
     - ``site_pos``
     - Position of the site frame in the parent body frame
     -
   * - ``dr.site_quat``
     - ``site_quat``
     - Orientation of the site frame
     - Accepts roll/pitch/yaw ranges (radians); composes with default

.. rubric:: Camera fields

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - Function
     - MuJoCo field
     - Description
     - Notes
   * - ``dr.cam_fovy``
     - ``cam_fovy``
     - Vertical field of view (degrees)
     -
   * - ``dr.cam_pos``
     - ``cam_pos``
     - Camera position in the parent body frame
     -
   * - ``dr.cam_quat``
     - ``cam_quat``
     - Camera orientation
     - Accepts roll/pitch/yaw ranges (radians); composes with default
   * - ``dr.cam_intrinsic``
     - ``cam_intrinsic``
     - Focal length and principal point ``[fx, fy, cx, cy]``
     -

.. rubric:: Light fields

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - Function
     - MuJoCo field
     - Description
     - Notes
   * - ``dr.light_pos``
     - ``light_pos``
     - Light position in the parent body frame
     -
   * - ``dr.light_dir``
     - ``light_dir``
     - Light direction vector
     -

.. rubric:: Tendon fields

.. list-table::
   :header-rows: 1
   :widths: 28 18 34 20

   * - Function
     - MuJoCo field
     - Description
     - Notes
   * - ``dr.tendon_damping``
     - ``tendon_damping``
     - Velocity-proportional damping along the tendon
     -
   * - ``dr.tendon_stiffness``
     - ``tendon_stiffness``
     - Spring stiffness along the tendon
     -
   * - ``dr.tendon_friction`` (alias ``tendon_frictionloss``)
     - ``tendon_frictionloss``
     - Dry friction loss along the tendon
     -
   * - ``dr.tendon_armature``
     - ``tendon_armature``
     - Inertia associated with tendon velocity
     - Triggers ``set_const_0``

Entity-level functions
^^^^^^^^^^^^^^^^^^^^^^

The functions above all write to a single ``sim.model`` field. The functions
below operate at the mjlab entity level instead, because they touch multiple
model fields at once or modify entity state that doesn't live on the MuJoCo
model.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - What it does
   * - ``dr.pseudo_inertia``
     - Physics-consistent joint randomization of ``body_mass``,
       ``body_ipos``, ``body_inertia``, and ``body_iquat`` via the
       pseudo-inertia matrix parameterization (Rucker & Wensing 2022).
       Guarantees a positive-definite inertia tensor and positive mass.
       See :ref:`dr-pseudo-inertia` for details.
   * - ``dr.pd_gains``
     - Randomizes stiffness (kp) and damping (kd) together. For
       ``BuiltinPositionActuator`` and ``XmlPositionActuator`` it writes to
       ``actuator_gainprm`` and ``actuator_biasprm``. For
       ``IdealPdActuator`` it sets gains on the entity directly. All three
       can be wrapped with ``DelayedActuator``.
   * - ``dr.effort_limits``
     - Randomizes actuator force range (``actuator_forcerange``). For
       ``IdealPdActuator`` also updates the entity's internal force limit.
       Supports ``BuiltinPositionActuator``, ``XmlPositionActuator``, and
       ``IdealPdActuator``.
   * - ``dr.encoder_bias``
     - Adds a fixed per-joint bias to position readings, simulating encoder
       calibration errors. Writes to ``entity.data.encoder_bias``, not the
       MuJoCo model.
   * - ``dr.sync_actuator_delays``
     - Samples a single lag value per environment and applies it to all
       ``DelayedActuator`` instances on the entity, ensuring consistent
       delay across joints.


.. _dr-pseudo-inertia:

Pseudo-inertia randomization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``dr.pseudo_inertia`` jointly randomizes ``body_mass``, ``body_ipos``,
``body_inertia``, and ``body_iquat`` while guaranteeing physical consistency
for any perturbation magnitude. Unlike randomizing these fields independently
(which can produce negative masses, imaginary principal moments, or
triangle-inequality violations), ``pseudo_inertia`` parameterizes inertia
through the *pseudo-inertia matrix* :math:`J \succ 0` and ensures the result
is always physically valid.

The pseudo-inertia matrix :math:`J` is a :math:`4 \times 4` symmetric
positive-definite matrix that encodes mass, center of mass, and the full
rotational inertia tensor in a single object:

.. math::

   J = \begin{bmatrix}
     \Sigma & h \\
     h^\top & m
   \end{bmatrix}, \qquad
   \Sigma = \tfrac{1}{2}\operatorname{tr}(I)\,I_3 - I, \qquad
   h = m\,c

where :math:`m` is mass (``body_mass``), :math:`c` is the center of mass
(``body_ipos``), :math:`d` is the vector of principal inertia moments
(``body_inertia``), and :math:`I` is the :math:`3 \times 3` inertia tensor
at the body-frame origin. The inertia tensor is constructed by rotating the
diagonal principal moments into the body frame and applying the
parallel-axis theorem:

.. math::

   I_{\text{com}} &= V \operatorname{diag}(d)\, V^\top,
   \qquad V = R(q)^\top \\
   I &= I_{\text{com}} + m\bigl(\lVert c \rVert^2 I_3 - c\,c^\top\bigr)

where :math:`q` is the body-to-principal-frame quaternion (``body_iquat``)
and :math:`R(q)` is its rotation matrix.

The math follows `Rucker & Wensing, "Smooth Parameterization of Rigid-Body
Inertia," IEEE RA-L 2022 <https://par.nsf.gov/servlets/purl/10347458>`_.
:math:`J` is factored via Cholesky as :math:`J = LL^\top`. A perturbation
is applied through an upper-triangular matrix :math:`U`:

.. math::

   J' = (UL)(UL)^\top

which is guaranteed positive definite for any :math:`U`. The perturbed
inertia tensor is then decomposed back into MuJoCo fields: the inverse
parallel-axis theorem shifts :math:`I` back to the COM, and
eigendecomposition extracts the principal moments (``body_inertia``) and
principal-frame rotation (``body_iquat``). This is exact for any
perturbation magnitude.

The 10 parameters of :math:`U` control different physical effects:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Physical effect
   * - ``alpha``
     - Global mass-density log-scale. Mass and all principal inertia moments
       scale by :math:`e^{2\alpha}`. Center of mass is unchanged.
   * - ``d1, d2, d3``
     - Axis-aligned stretch/compress along x, y, z in the inertia frame.
       Use ``d_range`` to set all three to the same range (isotropic).
   * - ``s12, s13, s23``
     - Shear perturbation in the xy, xz, and yz planes. Redistributes
       mass asymmetrically; produces off-diagonal inertia contributions.
   * - ``t1, t2, t3``
     - Shifts the center of mass along x, y, z (body frame). For a pure
       ``t1`` shift, mass is unchanged and ``body_ipos[0]`` shifts by
       exactly ``t1``. Use ``t_range`` to set all three to the same range.

.. rubric:: Example

.. code-block:: python

    events = {
        # Isotropic mass scaling + small COM variation.
        "body_inertia_dr": EventTermCfg(
            mode="reset",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["torso"]),
                "alpha_range": (-0.1, 0.1),   # ±10% mass/inertia scaling
                "t_range": (-0.02, 0.02),     # ±2 cm COM shift
            },
        ),
        # Anisotropic stretching (x stiffer than y/z).
        "body_inertia_aniso_dr": EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
                "alpha_range": (-0.2, 0.2),
                "d1_range": (-0.1, 0.1),
                "d2_range": (-0.3, 0.3),
                "d3_range": (-0.3, 0.3),
            },
        ),
    }


.. _dr-safety:

Safety of Runtime Model Changes
-------------------------------

In C MuJoCo, modifying ``mjModel`` fields at runtime can be unsafe — some
changes invalidate internal acceleration structures (BVH) or leave derived
quantities stale. MuJoCo Warp has a different collision pipeline, so many of
these concerns do not apply.

Two architectural differences matter most:

1. **No collision BVH.** C MuJoCo builds a static bounding-volume hierarchy
   (BVH) for midphase collision pruning. Changing ``body_pos``/``body_quat``
   of a static body invalidates that tree. MuJoCo Warp uses NxN or
   sweep-and-prune broadphase instead — there is no static tree to
   invalidate.

2. **Local bounding boxes.** ``geom_aabb`` in MuJoCo Warp is a *local*
   bounding box (center + half-size in the geom frame). The broadphase
   transforms it to world space every step using ``geom_xpos``/``geom_xmat``
   from forward kinematics. In C MuJoCo, the BVH caches world-space bounds,
   so any change to ``geom_pos``/``geom_quat`` makes them stale.

The table below shows which fields are safe to randomize in mjlab and how
that compares to C MuJoCo.

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 38

   * - Field(s)
     - C MuJoCo
     - mjlab / MuJoCo Warp
     - Why the difference
   * - ``body_pos``, ``body_quat``
     - Safe with ``mj_setConst``, but **unsafe for static bodies**
       (invalidates midphase BVH)
     - **Safe** with ``set_const_0``
     - No collision BVH to invalidate (see above). Body FK runs for
       all bodies regardless of ``body_treeid``. See the
       :ref:`static-body caveat <dr-static-body-caveat>` below.
   * - ``body_mass``, ``body_inertia``, ``body_ipos``, ``body_iquat``
     - Safe with ``mj_setConst``
     - **Safe** with ``set_const``
     - Same approach. ``dr.pseudo_inertia`` jointly randomizes all
       four while guaranteeing physical consistency.
   * - ``geom_pos``, ``geom_quat``
     - **Unsafe** (no ``mj_setConst`` support)
     - **Safe** for geoms on dynamic bodies
     - FK recomputes ``geom_xpos``/``geom_xmat`` every step from
       ``geom_pos``/``geom_quat``, and local ``geom_aabb`` stays
       valid (see above). See the
       :ref:`static-body caveat <dr-static-body-caveat>` below.
   * - ``geom_size``
     - **Unsafe**
     - **Safe** (recomputes bounds automatically)
     - ``dr.geom_size`` recomputes ``geom_rbound`` and ``geom_aabb``
       inline after writing new sizes. Only primitive types (sphere,
       capsule, ellipsoid, cylinder, box) are supported.
   * - ``geom_rbound``, ``geom_aabb``
     - **Unsafe** (internal derived quantities)
     - **Not randomized** (derived)
     - Broadphase acceleration data. Only set once during model
       loading. Would need recomputation if ``geom_size`` changes.
   * - ``geom_friction``, ``geom_rgba``
     - Safe
     - **Safe**
     - No derived quantities. Contact friction is read directly each
       step.
   * - ``dof_armature``
     - Safe with ``mj_setConst``
     - **Safe** with ``set_const_0``
     - Same approach.
   * - ``dof_damping``, ``dof_frictionloss``, ``jnt_stiffness``,
       ``jnt_range``
     - Safe
     - **Safe**
     - No derived quantities.
   * - ``qpos0``
     - Safe with ``mj_setConst``
     - **Safe** with ``set_const_0``
     - Same approach.
   * - ``tendon_stiffness``, ``tendon_damping``, ``tendon_frictionloss``
     - Mostly safe (``mj_setConst`` if changing from/to zero)
     - **Safe**
     - MuJoCo Warp does not use the sleep mechanism that makes the
       zero/non-zero transition special in C MuJoCo.
   * - ``tendon_armature``
     - Safe with ``mj_setConst``
     - **Safe** with ``set_const_0``
     - Contributes to the mass matrix via ``smooth.tendon_armature()``.
       Same approach as ``dof_armature``.
   * - ``actuator_gainprm``, ``actuator_biasprm``
     - Mostly safe (``mj_setConst`` for dampratio actuators)
     - **Safe**
     - mjlab's ``dr.pd_gains`` handles dampratio internally.
   * - ``site_pos``, ``site_quat``
     - Mostly safe (``mj_setConst`` for tracking cameras/lights)
     - **Safe**
     - Sites are recomputed via FK. No tracking camera concern in
       typical RL usage.
   * - ``bvh_aabb``, ``oct_aabb``, ``oct_coeff``
     - **Unsafe**
     - Not applicable / not randomized
     - MuJoCo Warp uses BVH only for rendering, not collision. Octree
       data (``oct_*``) is used for SDF collision and should not be
       modified.

.. _dr-static-body-caveat:

.. admonition:: Static-body caveat for ``geom_pos``/``geom_quat`` and
   ``body_pos``/``body_quat``

   MuJoCo Warp's forward kinematics skips geoms that are both
   **world-welded** (``body_weldid == 0``) and **not descended from a mocap
   body** (``body_mocapid[root] == -1``). For such geoms,
   ``geom_xpos``/``geom_xmat`` are computed once during ``make_data`` and
   never updated. Changing ``geom_pos`` or the parent ``body_pos`` would
   leave the world-space collision position stale.

   In practice this affects only bare ``<geom>`` elements placed directly on
   the ``<worldbody>`` in XML (e.g. a ground plane) that are not part of any
   mjlab entity.  All mjlab entities — including fixed-base ones — are
   auto-wrapped in a mocap body by ``auto_wrap_fixed_base_mocap``, which
   makes them exempt from the FK skip.  The built-in ``dr`` functions target
   named bodies/geoms on entities, so they are always safe.

.. _dr-geom-size:

How ``geom_size`` recomputation works
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Changing ``geom_size`` at runtime requires updating two derived fields that
the broadphase reads every step:

- ``geom_rbound`` — bounding sphere radius (used for sphere-filter pruning)
- ``geom_aabb`` — local axis-aligned bounding box (used for AABB/OBB pruning)

Both are copied from ``MjModel`` during model loading and never recomputed
by MuJoCo Warp. ``dr.geom_size`` handles this by recomputing both fields
inline (pure PyTorch, no Warp kernel needed) after writing the new sizes.

The formulas are type-dependent:

.. list-table::
   :header-rows: 1
   :widths: 18 30 30

   * - Geom type
     - ``geom_rbound``
     - ``geom_aabb`` half-size
   * - Sphere
     - ``s[0]``
     - ``(s[0], s[0], s[0])``
   * - Capsule
     - ``s[0] + s[1]``
     - ``(s[0], s[0], s[0] + s[1])``
   * - Cylinder
     - ``sqrt(s[0]² + s[1]²)``
     - ``(s[0], s[0], s[1])``
   * - Ellipsoid
     - ``max(s[0], s[1], s[2])``
     - ``(s[0], s[1], s[2])``
   * - Box
     - ``sqrt(s[0]² + s[1]² + s[2]²)``
     - ``(s[0], s[1], s[2])``

Plane, heightfield, mesh, and SDF geoms are not supported — their bounds
come from vertex data or are infinite, not derivable from ``geom_size``.


Fields without ``dr`` functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added as needed
""""""""""""""""

These continuous model fields could use a standard ``dr.*`` function but
don't have one yet. We will add them as demand arises.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Category
     - Field(s)
     - Notes
   * - Body
     - ``body_gravcomp``
     - Gravity compensation weight. Needs ``set_const_fixed``.
   * - Joint / DOF
     - ``jnt_margin``
     - Distance threshold for joint-limit detection.
   * - Tendon
     - ``tendon_range``
     - Tendon length limits.
   * -
     - ``tendon_margin``
     - Distance threshold for tendon-limit detection.
   * -
     - ``tendon_lengthspring``
     - Spring rest-length range.
   * - Actuator
     - ``actuator_dynprm``, ``actuator_gear``,
       ``actuator_ctrlrange``, ``actuator_actrange``
     - ``pd_gains`` and ``effort_limits`` cover common cases.
   * - Material
     - ``mat_rgba``, ``mat_texrepeat``
     - Continuous per-world fields. Requires material-level entity
       indexing (selecting by material name).

Better as custom code
"""""""""""""""""""""

These fields have coupled semantics that make a generic ``dr.*`` function
more misleading than helpful. For example, ``solref`` is interpreted
differently depending on the solver type (elliptic vs. direct), ``solimp``
has ordering constraints (dmin < dmax, width > 0), and ``qpos_spring`` is
coupled with ``qpos0``. The right ranges depend on your specific modeling
choices. Write a custom event term instead (see
:ref:`Custom Class-Based Event Terms <dr-custom-event-terms>` or use
``@requires_model_fields`` to handle field expansion automatically).

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Category
     - Field(s)
     - Notes
   * - Solver parameters
     - ``geom_solref``, ``geom_solimp``, ``geom_solmix``,
       ``jnt_solref``, ``jnt_solimp``, ``dof_solref``,
       ``dof_solimp``, ``pair_solref``, ``pair_solimp``,
       ``eq_solref``, ``eq_solimp``
     - Semantics depend on solver type and timestep.
   * - Contact thresholds
     - ``geom_margin``, ``geom_gap``, ``pair_margin``, ``pair_gap``
     - Interact with solver parameters above.
   * - Pair overrides
     - ``pair_friction``, ``eq_data``
     - Per-pair friction and constraint anchor overrides.
   * - Spring reference
     - ``qpos_spring``
     - Coupled with ``qpos0``; randomizing independently is
       error-prone.

Requires dedicated API
""""""""""""""""""""""

These fields need specialized handling because they are integer/categorical,
involve vertex data, or lack the per-world dimension needed for independent
per-environment values.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Category
     - Field(s)
     - Notes
   * - Material / texture swapping
     - ``geom_matid``, ``mat_texid``
     - Integer IDs that need a swapping API, not continuous
       sampling. Requires material-level entity indexing.
   * - Mesh
     - ``mesh_vert``, ``mesh_normal``, ``mesh_face``, etc.
     - Shape variation for manipulation objects. These fields
       lack the per-world dimension, so per-world variation is
       not possible with the current expand infrastructure.
       Heterogeneous-world support is
       `in progress <https://github.com/google-deepmind/mujoco_warp/pull/1009>`_.
   * - Deformable
     - ``flex_*``
     - Deformable body parameters for soft-object manipulation.
       Like mesh fields, most ``flex_*`` fields lack the per-world
       dimension.

Not a DR target
""""""""""""""""

- ``light_active``, ``light_castshadow``, ``light_type``,
  ``cam_projection``: boolean/integer toggles, not continuous parameters.
- ``jnt_pos``, ``jnt_axis``: structural joint geometry; changing at
  runtime is fragile and not a standard use case.
- ``hfield_data``, ``hfield_size``: terrain data; use the terrain
  system instead.


.. _dr-parameters:

Parameters
----------

The model field functions share three parameters that control randomization:
``distribution`` controls how values are sampled from ``ranges``, and
``operation`` controls how those sampled values are applied to the model field.
(Entity-level functions like ``dr.pseudo_inertia`` and ``dr.pd_gains`` have
their own signatures; see their docstrings for details.)

Distribution
^^^^^^^^^^^^

The ``distribution`` parameter controls how random values are sampled from
the provided ``ranges``. You can pass a built-in string or a
``dr.Distribution`` instance for custom sampling logic.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Behavior
   * - ``"uniform"`` (default)
     - Samples uniformly between ``ranges[0]`` and ``ranges[1]``
   * - ``"log_uniform"``
     - Samples in log space, useful for parameters that span orders of
       magnitude (e.g. torsional friction). Both range values must be > 0.
   * - ``"gaussian"``
     - ``ranges`` is interpreted as ``(mean, std)``

To define a custom distribution, create a ``dr.Distribution`` instance. The
``sample`` callable receives ``(lower, upper, shape, device)`` and returns a
tensor. For example, a truncated normal that clamps samples to the given
bounds:

.. code-block:: python

    import torch
    from mjlab.envs.mdp import dr

    truncated_normal = dr.Distribution(
        name="truncated_normal",
        sample=lambda lo, hi, shape, device: torch.clamp(
            torch.normal(
                mean=(lo + hi) / 2,
                std=(hi - lo) / 4,  # 95% of samples within bounds
            ).expand(shape),
            min=lo,
            max=hi,
        ),
    )

    params={"distribution": truncated_normal, "ranges": (0.3, 1.2)}

Operation
^^^^^^^^^

The ``operation`` parameter controls how the sampled value is applied to the
model field. You can pass a built-in string or a ``dr.Operation`` instance
for custom logic.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Behavior
   * - ``"abs"`` (default)
     - Sets the field to the sampled value directly
   * - ``"scale"``
     - Multiplies the original default value by the sampled value
   * - ``"add"``
     - Adds the sampled value to the original default value

For ``"scale"`` and ``"add"``, the DR engine always applies the random draw to
the **original default values** captured from the compiled ``MjModel`` on CPU,
not the current values. This prevents accumulation: scaling friction by 2x
three times in a row gives 2x the original, not 8x.

To define a custom operation, create a ``dr.Operation`` instance. The four
fields are:

- ``name``: a human-readable label for error messages.
- ``initialize``: creates the result tensor that gets filled axis by axis
  with sampled values. For example, ``scale`` starts from ones so that
  unsampled axes multiply by 1 (no change), while ``add`` starts from zeros.
- ``combine``: takes ``(base_values, random_values)`` and returns the final
  tensor written into the model field. For example, ``scale`` returns
  ``base * random`` and ``add`` returns ``base + random``.
- ``uses_defaults``: when ``True``, the base values are the compile-time
  defaults (preventing accumulation across repeated calls). When ``False``,
  the base values are the current model values.

As an example, the built-in ``add`` always adds to the *default* values, so
repeated calls reset rather than drift. A custom ``drift`` operation that
adds to the *current* values is useful for ``mode="interval"`` events where
parameters should slowly wander over time:

.. code-block:: python

    import torch
    from mjlab.envs.mdp import dr

    drift = dr.Operation(
        name="drift",
        initialize=torch.zeros_like,
        combine=torch.add,
        uses_defaults=False,  # read current values, not defaults
    )

    # Friction slowly wanders each interval step.
    friction_drift: EventTermCfg = EventTermCfg(
        mode="interval",
        interval_range_s=(0.5, 1.0),
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=[".*_foot.*"]),
            "ranges": (-0.01, 0.01),
            "operation": drift,
        },
    )

Axis selection
^^^^^^^^^^^^^^

Many model fields are multi-dimensional. For example, ``geom_friction`` has
three components ``[tangential, torsional, rolling]`` and ``body_pos`` has
three spatial axes ``[x, y, z]``. You can target specific axes using the
``axes`` parameter or by passing a dict for ``ranges``.

For ``geom_friction`` with ``condim=3`` (standard frictional contact), only
**axis 0 (tangential)** affects contact behavior. See the `MuJoCo contact
documentation
<https://mujoco.readthedocs.io/en/stable/computation/index.html#contact>`_
for details on condim and friction coefficients.

.. code-block:: python

    # Tangential friction only (this is the default for geom_friction)
    params={"ranges": {0: (0.3, 1.2)}}

    # Tangential + torsional (torsional matters for condim >= 4)
    params={"ranges": {0: (0.5, 1.0), 1: (0.001, 0.01)}}

    # X and Y position with the same range
    params={"axes": [0, 1], "ranges": (-0.1, 0.1)}

Per-component string-keyed ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can apply different ranges to different entities in a single call using
regex patterns as dict keys:

.. code-block:: python

    dr.joint_damping(
        env, env_ids,
        ranges={".*knee.*": (0.5, 1.5), ".*hip.*": (0.8, 1.2)},
        operation="scale",
        asset_cfg=SceneEntityCfg("robot", joint_names=[".*"]),
    )

Each pattern is matched against the entity names (joint names, geom names, etc.)
and the corresponding range is applied to the matching components. In this
example, knee joints get their damping scaled by 0.5x to 1.5x while hip joints
get a tighter 0.8x to 1.2x range, all in a single event term.


Examples
--------

Friction (reset)
^^^^^^^^^^^^^^^^

.. code-block:: python

    foot_friction: EventTermCfg = EventTermCfg(
        mode="reset",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=[".*_foot.*"]),
            "ranges": (0.3, 1.2),
            "operation": "abs",
        },
    )

.. note::

     Give your robot's collision geoms higher **priority** than terrain
     (geom priority defaults to 0). Then you only need to randomize robot
     friction. MuJoCo will use the higher-priority geom's friction in
     (robot, terrain) contacts.

.. code-block:: python

    from mjlab.utils.spec_config import CollisionCfg

    robot_collision = CollisionCfg(
        geom_names_expr=[".*_foot.*"],
        priority=1,
        friction=(0.6,),
        condim=3,
    )


Joint Offset (startup)
^^^^^^^^^^^^^^^^^^^^^^

Randomize default joint positions to simulate joint offset calibration errors:

.. code-block:: python

    joint_offset: EventTermCfg = EventTermCfg(
        mode="startup",
        func=dr.joint_default_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "ranges": (-0.01, 0.01),
            "operation": "add",
        },
    )


Center of Mass (COM) (startup)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    com: EventTermCfg = EventTermCfg(
        mode="startup",
        func=dr.body_com_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso"]),
            "ranges": {0: (-0.02, 0.02), 1: (-0.02, 0.02)},
            "operation": "add",
        },
    )


Common Pitfalls
---------------

``dr.body_mass`` does not scale inertia
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scaling mass without scaling inertia is only physically correct when modelling
a point mass added at the COM (which contributes zero rotational inertia).
For the common DR use case, simulating manufacturing variation or uncertainty
in link density, mass and inertia should scale together. Use
:func:`dr.pseudo_inertia` with ``alpha_range`` instead:

.. code-block:: python

    # Wrong: mass changes, inertia stays fixed (physically inconsistent).
    EventTermCfg(func=dr.body_mass, params={"ranges": (0.8, 1.2)})

    # Correct: mass and inertia both scale by e^{2alpha} (uniform density change).
    EventTermCfg(func=dr.pseudo_inertia, params={"alpha_range": (-0.1, 0.1)})

``dr.body_mass`` emits a ``UserWarning`` at runtime to flag this.

``*_quat`` ranges are in radians, not degrees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All quaternion randomization functions (:func:`dr.geom_quat`,
:func:`dr.body_quat`, :func:`dr.site_quat`, :func:`dr.cam_quat`) accept
roll/pitch/yaw ranges in **radians**. Passing degree values silently produces
rotations roughly 57x larger than intended. There is no runtime check.

``*_quat`` perturbations are relative to the default, not the current value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sampled RPY perturbation is composed with the **default** quaternion, not
the current one. This is consistent with the no-accumulation guarantee of all
other ``dr`` functions, but it means repeated calls do not stack rotations.
Each call independently samples a new perturbation from the default orientation.

``dr.geom_friction`` only randomizes tangential friction by default
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default axis is 0 (tangential friction). Axes 1 (torsional) and 2
(rolling) only affect contacts with ``condim >= 4``, so the default is correct
for standard ``condim=3`` contacts. If your model uses high-dimensional
contacts, pass ``axes=[0, 1, 2]`` explicitly.

``dr.geom_size`` raises on non-primitive geoms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`dr.geom_size` only supports primitive geom types (sphere, capsule,
ellipsoid, cylinder, box) because broadphase bounds for mesh, plane, and
heightfield geoms cannot be recomputed analytically. Selecting a non-primitive
geom raises a ``ValueError``. Filter your ``SceneEntityCfg`` to primitive geoms
only:

.. code-block:: python

    # Select only primitive geoms by name pattern.
    geom_cfg = SceneEntityCfg("robot", geom_names=(".*sphere.*", ".*box.*"))

How It Works Under the Hood
---------------------------

Understanding the internals helps when writing custom DR terms or debugging
unexpected behavior.

Per-world storage
^^^^^^^^^^^^^^^^^

MuJoCo Warp batches thousands of worlds into a single simulation. To save
memory, model arrays like ``geom_friction`` are stored once with shape
``(1, ngeom, 3)`` and a **stride of 0** along the first (world) dimension.
GPU kernels index into these arrays with ``worldid % arr.shape[0]``. When
``shape[0]`` is 1, every world reads the same row, so they all share identical
model parameters.

On the PyTorch side, mjlab wraps these stride-0 arrays using ``torch.expand``
so they appear to have shape ``(num_envs, ngeom, 3)`` while still backed by a
single row of memory. You can index with ``tensor[env_id]`` and it looks like
each world has its own data, but writes to any world affect all of them because
they all point to the same underlying memory.

To actually give each world independent values, we **expand** the underlying
Warp array from shape ``(1, N)`` to ``(num_worlds, N)`` with real per-world
memory and normal strides. This is what ``sim.expand_model_fields()`` does: it
allocates a new array, copies the shared data into each world's row, and
replaces the old array on the model. After expansion, writes to one world no
longer affect others and each world can have its own friction, mass, or
damping values.

Each ``dr`` function declares which fields it needs via the
``@requires_model_fields`` decorator, and the ``EventManager`` collects these
at startup so expansion happens automatically. If you are writing a custom DR
term that directly modifies model arrays, make sure those arrays have been
expanded. Either decorate your function with ``@requires_model_fields`` or call
``sim.expand_model_fields()`` manually.

.. note::

   Expanding a field allocates new GPU memory and invalidates any captured CUDA
   graph because the graph holds pointers to the old arrays. mjlab recreates
   the graph automatically after expansion. This is a one-time cost at
   environment startup, not per-episode.

Why not just recompile the model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cleanest way to get per-world variation would be to modify the ``MjSpec``
for each world, compile each one into its own ``MjModel``, and transfer them all
to the GPU. That would give perfectly consistent derived quantities for free
because ``mj_setConst`` runs during compilation.

We don't do this for two reasons:

1. **Cost.** Compiling an ``MjSpec`` is a CPU operation. Doing it once per world
   on every episode reset would be far too slow for thousands of environments.

2. **Architecture.** MuJoCo Warp expects a single ``Model`` shared across all
   worlds. There is no mechanism to load N independent models into one
   simulation.

Instead, we modify the expanded arrays in place on the GPU and selectively
recompute only the derived quantities that depend on what changed. This is
what the ``RecomputeLevel`` system handles.

.. _dr-recomputation:

Recomputation of derived fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some model fields are **derived** from others. For example,
``body_subtreemass`` (the total mass of a body and all its descendants) depends
on ``body_mass``. If you change ``body_mass`` without updating
``body_subtreemass``, the constraint solver will use stale impedance values
and the simulation will be subtly wrong.

In C MuJoCo, you would call `mj_setConst
<https://mujoco.readthedocs.io/en/latest/programming/simulation.html#mjmodel-changes>`_
after modifying model parameters at runtime. MuJoCo Warp provides an equivalent
set of functions (``set_const``, ``set_const_0``, ``set_const_fixed``) that run
on the GPU and operate on all worlds in parallel. mjlab calls these for you
automatically.

The three recomputation levels, from cheapest to most expensive:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Level
     - What it recomputes
     - When to use
   * - ``set_const_fixed``
     - ``body_subtreemass``
     - After changing ``body_gravcomp``
   * - ``set_const_0``
     - ``dof_invweight0``, ``body_invweight0``, ``tendon_length0``,
       ``tendon_invweight0``, plus camera and light references
     - After changing ``dof_armature``, ``tendon_armature``,
       ``body_inertia``, ``body_pos``, ``body_quat``, or ``qpos0``
   * - ``set_const``
     - Everything above
     - After changing ``body_mass`` or ``body_ipos`` (center of mass)

The built-in ``dr`` functions already declare the correct level. When the
``EventManager`` fires multiple DR terms in a single ``apply()`` call, it
tracks the strongest level among them and calls ``sim.recompute_constants()``
once at the end. You never need to call it yourself unless you are writing
custom DR logic.

Fields that only affect contact or joint behavior directly (``geom_friction``,
``dof_damping``, ``dof_frictionloss``, etc.) have no derived quantities and
need no recomputation.

.. note::

   mjlab captures ``sim.step()``, ``sim.forward()``, ``sim.reset()``, and
   ``sim.sense()`` as individual CUDA graphs for performance. All event
   manager logic, including ``recompute_constants``, runs as regular
   Python between these graph replays, so it will not break graph capture.

   That said, ``set_const`` is expensive: it runs forward kinematics, the
   composite rigid body algorithm, and mass matrix factorization across all
   worlds. Calling it every step via an ``interval`` event would add
   significant overhead. In practice, fields that need recomputation
   (``body_mass``, ``body_com_offset``, ``joint_armature``, etc.) are best
   randomized with ``startup`` or ``reset`` modes. Fields that need no
   recomputation (``geom_friction``, ``dof_damping``, etc.) are cheap to
   randomize at any frequency.


.. _dr-custom-event-terms:

Custom Class-Based Event Terms
------------------------------

You can create custom event terms using classes instead of functions. This is
useful for event terms that need to maintain state or perform initialization
logic:

.. code-block:: python

    class RandomizeTerrainFriction:
        """Custom event term that randomizes terrain friction."""

        def __init__(self, cfg, env):
            # Find the terrain geom index during initialization
            self._terrain_idx = None
            for idx, geom in enumerate(env.scene.spec.geoms):
                if geom.name == "terrain":
                    self._terrain_idx = idx

            if self._terrain_idx is None:
                raise ValueError("Terrain geom not found in the model.")

        def __call__(self, env, env_ids, ranges):
            """Called each time the event is triggered."""
            from mjlab.utils.math import sample_uniform
            env.sim.model.geom_friction[env_ids, self._terrain_idx, 0] = (
                sample_uniform(ranges[0], ranges[1], len(env_ids), env.device)
            )


    # Use the custom class in your environment config
    terrain_friction: EventTermCfg = EventTermCfg(
        mode="reset",
        func=RandomizeTerrainFriction,
        params={"ranges": (0.3, 1.2)},
    )


Visualizing DR Changes
----------------------

Both viewers reflect DR changes, but with different coverage.

**Native viewer**

The native viewer syncs per-world model fields from the GPU to a local
``MjModel`` before each render. All of MuJoCo's built-in visualization
toggles then work correctly against the randomized model:

- Geom appearance (``geom_rgba``, ``geom_size``, ``geom_pos``, ``geom_quat``)
- Body and site poses (``body_pos``, ``body_quat``, ``body_ipos``,
  ``site_pos``, ``site_quat``)
- Inertia (``body_inertia``, ``body_iquat``, ``body_mass``) — press ``I``
  to toggle inertia boxes
- Camera parameters (``cam_pos``, ``cam_quat``, ``cam_fovy``,
  ``cam_intrinsic``) — press ``Q`` to toggle camera frustums
- Lights (``light_pos``, ``light_dir``)

.. note::

   ``cam_fovy`` has no effect on the frustum visualization for cameras that
   use intrinsic parameters (``sensorsize`` / ``focal`` set in XML). MuJoCo
   draws those frustums from ``cam_intrinsic`` and ``cam_sensorsize``
   instead. Use ``dr.cam_intrinsic`` to vary the frustum for such cameras.

**Viser**

Camera frustums and body poses are always current because viser reads
world-space positions directly from GPU simulation data (``cam_xpos``,
``body_xpos``) every frame.

.. note::

   ``geom_rgba`` and ``geom_size`` DR are **not** reflected in viser. Geom
   colors and sizes are baked into the scene's GLB meshes at construction
   time. The underlying viser API (``add_batched_meshes_simple``) supports
   per-instance color updates via ``batched_colors``, but this requires
   routing color-only geoms through a different handle type than the current
   ``add_batched_meshes_trimesh`` path. Deferred for a future update.


Migrating from Isaac Lab
------------------------

Isaac Lab exposes explicit friction combination modes (``multiply``, ``average``,
``min``, ``max``). MuJoCo instead uses **priority-based selection**: if one
contacting geom has higher ``priority``, its friction is used. Otherwise the
**element-wise maximum** is used. See the
`MuJoCo contact documentation <https://mujoco.readthedocs.io/en/stable/computation/index.html#contact>`_
for details.
