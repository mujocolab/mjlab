.. _entity:

Entity
======

An ``Entity`` represents a physical object in the simulation: a robot, a
manipulated object, or a fixed fixture like a table. It is the central
abstraction in mjlab's physics layer.

A single ``Entity`` class covers all variants (contrast Isaac Lab, which
splits this across ``Articulation``, ``RigidObject``, and several other
subclasses of ``AssetBase``). Two orthogonal boolean properties classify
each instance:

**Base type.**
  A *fixed-base* entity is welded to the world and has no free joint. A
  *floating-base* entity has a free joint giving it 6-DOF movement.

**Articulation.**
  An *articulated* entity has internal joints (revolute, prismatic, etc.).
  A *non-articulated* entity has none beyond a possible free joint.

.. list-table::
   :header-rows: 1
   :widths: 30 25 15 15 15

   * - Type
     - Example
     - ``is_fixed_base``
     - ``is_articulated``
     - ``is_actuated``
   * - Fixed non-articulated
     - Table, wall
     - True
     - False
     - False
   * - Fixed articulated
     - Robot arm, door
     - True
     - True
     - True/False
   * - Floating non-articulated
     - Box, ball, mug
     - False
     - False
     - False
   * - Floating articulated
     - Humanoid, quadruped
     - False
     - True
     - True/False

.. note::

   mjlab automatically wraps every fixed-base entity in a
   `mocap body <https://mujoco.readthedocs.io/en/stable/modeling.html#mocap-bodies>`_
   so that each parallel environment can place the entity at a different
   position. Without this wrapping, all fixed-base entities would be
   welded to the world origin. The wrapping is transparent, but
   **positioning only happens when a reset event runs**. You must
   include a reset event such as ``reset_root_state_uniform`` in your
   event config; without one, every fixed-base entity will remain at
   the origin. See the :ref:`FAQ <faq>` for a full example. Mocap
   entities can also be repositioned at runtime via
   ``entity.write_mocap_pose_to_sim()``.


Configuring an entity
---------------------

Every entity is described by an ``EntityCfg``. Only ``spec_fn`` is
required in practice; all other fields have sensible defaults. A passive
floating object needs nothing more than:

.. code-block:: python

    from mjlab.entity import EntityCfg

    cube_cfg = EntityCfg(spec_fn=get_cube_spec)

An actuated robot uses more of the interface:

.. code-block:: python

    from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
    from mjlab.actuator import IdealPDActuatorCfg

    robot_cfg = EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={".*_hip_.*": 0.5, ".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                IdealPDActuatorCfg(
                    target_names_expr=(".*",),
                    stiffness={".*": 50.0},
                    damping={".*": 5.0},
                ),
            ),
        ),
        collisions=(my_collision_cfg,),
    )

The following sections describe each field.

``spec_fn``
^^^^^^^^^^^

A callable that returns an ``mujoco.MjSpec``. The scene calls it during
composition, attaches the returned spec with a name prefix, and compiles
everything into a shared ``MjModel``.

For simple cases a lambda suffices:

.. code-block:: python

    spec_fn = lambda: mujoco.MjSpec.from_file("robot.xml")

For anything more involved, use a regular function. The asset zoo robots
all follow this pattern: load the XML, attach mesh assets, and return
the spec.

.. code-block:: python

    def get_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
        spec.assets = get_assets(spec.meshdir)
        return spec

Because ``spec_fn`` is an arbitrary callable, you can perform any
`MjSpec edits <https://mujoco.readthedocs.io/en/stable/python.html#spec>`_
before returning: add bodies, change joint limits, swap materials,
or build the entire model programmatically without an XML file at all.

``init_state``
^^^^^^^^^^^^^^

Default root pose, root velocity, and joint positions/velocities. These
values are stored as a MuJoCo keyframe and used by reset events to
return the entity to its initial configuration.

``joint_pos`` and ``joint_vel`` are dicts mapping regex patterns to
values. Patterns are matched against joint names in order, so later
entries override earlier ones for any joint that matches both:

.. code-block:: python

    init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),       # root position
        rot=(1.0, 0.0, 0.0, 0.0),  # root quaternion (w, x, y, z)
        joint_pos={
            ".*": 0.0,              # all joints to zero
            ".*_hip_.*": 0.5,       # then override hips to 0.5
        },
    )

Set ``joint_pos=None`` to use an existing keyframe from the MJCF model
instead of defining values here.

``articulation``
^^^^^^^^^^^^^^^^

Actuator configuration. Only needed for entities that have actuated
joints. Passive objects (boxes, tables, walls) can omit this field
entirely. See :ref:`actuators` for details on actuator types.

``soft_joint_pos_limit_factor`` (default 1.0) shrinks the joint range
used by soft-limit penalty rewards, so the policy is penalized before
reaching the physical hard stop. This does not modify the actual joint
limits in the MuJoCo model.

Spec editors
^^^^^^^^^^^^

The remaining fields are optional tuples of spec editor configs that
modify the ``MjSpec`` before compilation:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Purpose
   * - ``collisions``
     - Set contact parameters (contype, conaffinity, friction) per geom.
   * - ``lights``
     - Add lights to specific bodies.
   * - ``cameras``
     - Add cameras to specific bodies.
   * - ``textures``
     - Add procedural textures (checker, gradient, etc.).
   * - ``materials``
     - Add materials and optionally assign them to geoms by regex.

Each editor accepts regex patterns to target specific elements. For
example, a ``CollisionCfg`` with ``geom_names_expr=(".*_foot.*",)``
sets contact parameters only on foot geoms. See the asset zoo
(``mjlab.asset_zoo.robots``) for complete examples.

Subclassing Entity
^^^^^^^^^^^^^^^^^^

``Entity`` and ``EntityCfg`` can be subclassed for specialized behavior.
mjlab itself does this for terrain: ``TerrainEntity`` extends ``Entity``
with procedural terrain generation and per-environment origin
computation, and ``TerrainEntityCfg`` adds fields like
``terrain_type``, ``env_spacing``, and ``terrain_generator``. The same
pattern works for any domain-specific entity that needs logic beyond
what ``EntityCfg`` and spec editors provide.

Finding elements
^^^^^^^^^^^^^^^^

Entity provides ``find_*`` methods that accept regex patterns and return
matched element indices and names:

.. code-block:: python

    ids, names = entity.find_joints((".*_hip_.*", ".*_knee_.*"))
    ids, names = entity.find_geoms((".*foot.*",))
    ids, names = entity.find_bodies((".*",))

Available methods: ``find_bodies()``, ``find_joints()``,
``find_geoms()``, ``find_sites()``, ``find_tendons()``.
These are used internally during scene construction and manager
initialization. In reward and observation terms, prefer
``SceneEntityCfg`` with name patterns as described below.


Reading runtime state
---------------------

Once entities are added to a ``SceneCfg`` and the environment is
constructed, their state is accessible through three interfaces at
decreasing levels of abstraction.

EntityData
^^^^^^^^^^

``entity.data`` is the primary interface for reward, observation, and
termination functions. It exposes kinematic state (poses, velocities, accelerations), actuator forces,
generalized forces, and derived body-frame quantities such as projected
gravity, all as PyTorch tensors with
shape ``(num_envs, ...)``. See :ref:`entity_data` for the full property
reference.

``SceneEntityCfg`` selects which entity and which elements within it a
term operates on. Regex patterns in ``joint_names``, ``body_names``,
``site_names``, etc. are resolved to integer indices once at manager
initialization, so there is no regex overhead at runtime:

.. code-block:: python

    from mjlab.managers.scene_entity_config import SceneEntityCfg

    def flat_orientation_l2(
        env,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Penalize non-flat base orientation using projected gravity."""
        asset = env.scene[asset_cfg.name]
        return torch.sum(
            torch.square(asset.data.projected_gravity_b[:, :2]), dim=1
        )

``SceneEntityCfg`` also supports regex element selection through
``joint_names``, ``body_names``, ``site_names``, etc. The resolved
integer indices (e.g., ``asset_cfg.joint_ids``) make the runtime read a
single tensor slice with no regex overhead.

Sensors
^^^^^^^

Sensors are configured on the **scene**, not on individual entities.
A sensor can reference an entity element (e.g., a contact sensor on the
robot's feet, an accelerometer attached to a body site), but it can also
be independent of any entity. This is why sensors live in ``SceneCfg``
rather than ``EntityCfg``.

At runtime, sensors are accessed by name through ``env.scene``, the same
way entities are:

.. code-block:: python

    def angular_momentum_penalty(env, sensor_name: str) -> torch.Tensor:
        sensor = env.scene[sensor_name]
        return torch.sum(torch.square(sensor.data), dim=-1)

Builtin sensors wrap MuJoCo sensor types (accelerometer, gyro, framepos,
subtreeangmom, etc.). ``ContactSensor``, ``RayCastSensor``, and
``CameraSensor`` provide higher-level abstractions for contact detection,
terrain scanning, and RGB-D rendering. See :ref:`sensors` for details.

Raw simulation data
^^^^^^^^^^^^^^^^^^^

For anything not covered by ``EntityData`` or sensors, the underlying
MuJoCo Warp arrays are accessible through ``env.sim.data`` and
``env.sim.model``. These expose the full ``mjData`` and ``mjModel``
fields as PyTorch tensors (zero-copy), indexed by global MuJoCo IDs
rather than per-entity IDs:

.. code-block:: python

    # Global joint positions across all entities.
    qpos = env.sim.data.qpos          # (num_envs, nq)

    # All body positions.
    xpos = env.sim.data.xpos          # (num_envs, nbody, 3)

    # Model-level constants.
    body_mass = env.sim.model.body_mass  # (nbody,)

This is useful for low-level operations or when you need quantities
that span multiple entities.

.. note::

   The main limitation of raw sim data is that you must manage global
   MuJoCo indices yourself. In the future, we plan to support MuJoCo's
   `bind <https://mujoco.readthedocs.io/en/latest/python.html#relationship-to-pymjcf-and-bind>`_
   functionality, which will allow binding spec elements directly to
   their corresponding data views without manual index bookkeeping.

.. toctree::
   :maxdepth: 1

   entity_data
