Domain Randomization
====================

Domain randomization varies physical parameters during training so that policies
are robust to modeling errors and real-world variation. mjlab provides
**per-field functions** — ``dr.geom_friction()``, ``dr.body_mass()``,
``dr.joint_damping()``, etc. — that are typed, self-documenting, and
automatically handle derived-field recomputation.

TL;DR
-----

Use an ``EventTermCfg`` that calls the appropriate ``dr.*`` function
with a **value range** and **operation**.

.. code-block:: python

    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.envs.mdp import dr

    foot_friction = EventTermCfg(
        mode="reset",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_foot.*",)),
            "ranges": (0.3, 1.2),
            "operation": "abs",
        },
    )

Each function uses the ``@requires_model_fields`` decorator, which tells the
framework which model fields to expand per-world and which derived quantities
to recompute after randomization.

Event Modes
-----------

* ``"startup"`` — randomize once at initialization
* ``"reset"`` — randomize at every episode reset
* ``"interval"`` — randomize at regular time intervals

Available Functions
-------------------

**Geom:**

* ``dr.geom_friction`` — ``geom_friction`` (default axes: [0])
* ``dr.geom_pos`` — ``geom_pos`` (default axes: [0,1,2])
* ``dr.geom_quat`` — ``geom_quat`` (default axes: [0,1,2,3])
* ``dr.geom_rgba`` — ``geom_rgba`` (default axes: [0,1,2,3])

**Site:**

* ``dr.site_pos`` — ``site_pos`` (default axes: [0,1,2])
* ``dr.site_quat`` — ``site_quat`` (default axes: [0,1,2,3])

**Body:**

* ``dr.body_mass`` — ``body_mass`` (recomputes ``set_const``)
* ``dr.body_inertia`` — ``body_inertia`` (recomputes ``set_const_0``)
* ``dr.body_inertia_quat`` — ``body_iquat`` (recomputes ``set_const_0``)
* ``dr.body_com_offset`` — ``body_ipos`` (recomputes ``set_const``)
* ``dr.body_pos`` — ``body_pos`` (recomputes ``set_const_0``)
* ``dr.body_quat`` — ``body_quat`` (recomputes ``set_const_0``)

**Joint/DOF:**

* ``dr.joint_damping`` — ``dof_damping``
* ``dr.joint_armature`` — ``dof_armature`` (recomputes ``set_const_0``)
* ``dr.joint_friction`` — ``dof_frictionloss``
* ``dr.joint_stiffness`` — ``jnt_stiffness``
* ``dr.joint_limits`` — ``jnt_range``
* ``dr.joint_default_pos`` — ``qpos0`` (recomputes ``set_const_0``)

**Tendon:**

* ``dr.tendon_damping`` — ``tendon_damping``
* ``dr.tendon_stiffness`` — ``tendon_stiffness``
* ``dr.tendon_friction`` — ``tendon_frictionloss``
* ``dr.tendon_length_spring`` — ``tendon_lengthspring``

**Actuator:**

* ``dr.pd_gains`` — PD stiffness/damping gains
* ``dr.effort_limits`` — actuator force limits

**Other:**

* ``dr.encoder_bias`` — joint encoder calibration bias

Randomization Parameters
------------------------

**Distribution:** ``"uniform"`` (default), ``"log_uniform"`` (values must be > 0),
``"gaussian"`` (``mean, std``)

**Operation:** ``"abs"`` (set), ``"scale"`` (multiply), ``"add"`` (offset).
Default varies per function (e.g., ``"scale"`` for mass, ``"abs"`` for friction).

Axis selection
^^^^^^^^^^^^^^

Multi-dimensional fields can be randomized per-axis.

**Friction.** Geoms have three coefficients ``[tangential, torsional, rolling]``.
For ``condim=3`` (standard frictional contact), only **axis 0 (tangential)**
affects contact behavior:

.. code-block:: python

    # Tangential friction only (default)
    params={"ranges": (0.3, 1.2)}

    # Tangential + torsional
    params={"ranges": (0.5, 1.0), "axes": [0, 1]}

Per-component ranges
^^^^^^^^^^^^^^^^^^^^

String-keyed dictionaries let you apply different ranges to different
entities matched by name pattern:

.. code-block:: python

    EventTermCfg(
        mode="reset",
        func=dr.joint_damping,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "ranges": {".*knee.*": (0.5, 1.5), ".*hip.*": (0.8, 1.2)},
            "operation": "scale",
        },
    )

Derived Quantity Recomputation
------------------------------

Some model fields have derived quantities that must be recomputed after
modification (e.g., ``body_subtreemass`` after changing ``body_mass``).
The ``@requires_model_fields`` decorator declares the recompute level,
and the ``EventManager`` automatically calls ``sim.recompute_constants()``
at the end of ``apply()`` with the strongest level needed.

Recompute levels (strongest to weakest):

* ``"set_const"`` — full recomputation (mass + qpos0-dependent)
* ``"set_const_0"`` — qpos0-dependent quantities only
* ``"set_const_fixed"`` — mass-dependent quantities only
* ``"none"`` — no recomputation needed

Examples
--------

Friction (reset)
^^^^^^^^^^^^^^^^

.. code-block:: python

    foot_friction = EventTermCfg(
        mode="reset",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_foot.*",)),
            "ranges": (0.3, 1.2),
            "operation": "abs",
        },
    )

.. note::

     Give your robot's collision geoms higher **priority** than terrain
     (geom priority defaults to 0). Then you only need to randomize robot friction.
     MuJoCo will use the higher-priority geom's friction in (robot, terrain)
     contacts.

.. code-block:: python

    from mjlab.utils.spec_config import CollisionCfg

    robot_collision = CollisionCfg(
        geom_names_expr=(".*_foot.*",),
        priority=1,
        friction=(0.6,),
        condim=3,
    )


Body Mass (reset)
^^^^^^^^^^^^^^^^^

.. code-block:: python

    mass = EventTermCfg(
        mode="reset",
        func=dr.body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
            "ranges": (0.8, 1.2),
            "operation": "scale",
        },
    )

The ``body_subtreemass`` and other derived fields are automatically
recomputed via ``set_const`` at the end of the event batch.


Joint Offset (startup)
^^^^^^^^^^^^^^^^^^^^^^

Randomize default joint positions to simulate joint offset calibration errors:

.. code-block:: python

    joint_offset = EventTermCfg(
        mode="startup",
        func=dr.joint_default_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "ranges": (-0.01, 0.01),
            "operation": "add",
        },
    )


Center of Mass (COM) (startup)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    com = EventTermCfg(
        mode="startup",
        func=dr.body_com_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
            "ranges": {0: (-0.02, 0.02), 1: (-0.02, 0.02)},
            "operation": "add",
        },
    )

Custom Class-Based Event Terms
------------------------------

You can create custom event terms using classes instead of functions. This is useful
for event terms that need to maintain state or perform initialization logic:

.. code-block:: python

    from mjlab.managers.event_manager import requires_model_fields

    @requires_model_fields("geom_friction")
    class RandomizeTerrainFriction:
        """Custom event term that randomizes terrain friction."""

        def __init__(self, cfg, env):
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

    terrain_friction = EventTermCfg(
        mode="reset",
        func=RandomizeTerrainFriction,
        params={"ranges": (0.3, 1.2)},
    )


Migrating from Isaac Lab
------------------------

Isaac Lab exposes explicit friction combination modes (``multiply``, ``average``,
``min``, ``max``). MuJoCo instead uses **priority-based selection**: if one
contacting geom has higher ``priority``, its friction is used; otherwise the
**element-wise maximum** is used. See the
`MuJoCo contact documentation <https://mujoco.readthedocs.io/en/stable/computation/index.html#contact>`_
for details.
