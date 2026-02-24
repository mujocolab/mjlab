.. _sensors:

Sensors
=======

As described in :ref:`entity`, sensors sit between ``EntityData`` and
raw simulation arrays in mjlab's data access hierarchy. At their
simplest, they wrap MuJoCo sensor primitives with a clean interface
that maps to real robot hardware. Beyond wrapping, they are a general
abstraction for transforming simulation data into structured outputs:
``ContactSensor`` aggregates contact pairs with reduction and air time
tracking, ``RayCastSensor`` performs GPU-accelerated terrain scanning,
``CameraSensor`` renders RGB and depth images on the GPU, and the base
``Sensor`` class can be subclassed for custom measurement logic.

Sensors are configured at the **scene level**, not on individual entities. A
sensor can reference an entity element (a contact sensor on the robot's
feet, an accelerometer attached to a body site), but it can also be
independent of any entity entirely. This is why sensors live in
``SceneCfg`` rather than ``EntityCfg``.

.. code-block:: python

    from mjlab.sensor import (
        BuiltinSensorCfg, ContactSensorCfg, ContactMatch, ObjRef,
    )

    # A robot with an IMU accelerometer and foot contact detection.
    scene_cfg = SceneCfg(
        entities={"robot": robot_cfg},
        sensors=(
            BuiltinSensorCfg(
                name="imu_acc",
                sensor_type="accelerometer",
                obj=ObjRef(type="site", name="imu_site", entity="robot"),
            ),
            ContactSensorCfg(
                name="feet_contact",
                primary=ContactMatch(
                    mode="geom", pattern=r".*_foot$", entity="robot",
                ),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"),
            ),
        ),
    )

    # Access at runtime.
    imu = env.scene["robot/imu_acc"].data        # [B, 3] acceleration
    feet = env.scene["feet_contact"].data         # ContactData
    feet.found                                    # [B, N] contact count
    feet.force                                    # [B, N, 3] contact force

mjlab provides four sensor types: ``BuiltinSensor`` for native MuJoCo
measurements, ``ContactSensor`` for structured contact detection,
``RayCastSensor`` for GPU-accelerated raycasting, and ``CameraSensor``
for RGB-D rendering. The base ``Sensor`` class can be subclassed for
custom measurement logic; see `Extending: custom sensors`_ below.


BuiltinSensor
-------------

``BuiltinSensor`` wraps MuJoCo's native sensor types. Each sensor is
attached to a MuJoCo element (site, joint, body, etc.) via ``ObjRef``
and returns a ``torch.Tensor`` with shape ``[num_envs, dim]`` where
``dim`` depends on the sensor type (3 for vectors, 4 for quaternions,
1 for scalars).

+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------+
| Category  | Available Sensors                                                                                                                                  |
+===========+====================================================================================================================================================+
| **Site**  | ``accelerometer``, ``velocimeter``, ``gyro``, ``force``, ``torque``, ``magnetometer``, ``rangefinder``                                             |
+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------+
| **Joint** | ``jointpos``, ``jointvel``, ``jointlimitpos``, ``jointlimitvel``, ``jointlimitfrc``, ``jointactuatorfrc``                                          |
+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------+
| **Frame** | ``framepos``, ``framequat``, ``framexaxis``, ``frameyaxis``, ``framezaxis``, ``framelinvel``, ``frameangvel``, ``framelinacc``, ``frameangacc``    |
+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------+
| **Other** | ``actuatorpos``, ``actuatorvel``, ``actuatorfrc``, ``subtreecom``, ``subtreelinvel``, ``subtreeangmom``, ``clock``, ``e_potential``, ``e_kinetic`` |
+-----------+----------------------------------------------------------------------------------------------------------------------------------------------------+

``ObjRef`` identifies which MuJoCo element the sensor attaches to. The
``entity`` field scopes the lookup to a specific entity's namespace, and
the sensor name is auto-prefixed accordingly (e.g., ``"imu_acc"`` on
entity ``"robot"`` becomes ``"robot/imu_acc"``).

.. code-block:: python

    from mjlab.sensor import BuiltinSensorCfg, ObjRef

    # Accelerometer attached to a site.
    BuiltinSensorCfg(
        name="imu_acc",
        sensor_type="accelerometer",
        obj=ObjRef(type="site", name="imu_site", entity="robot"),
    )

    # Joint limit sensor with output clamping.
    BuiltinSensorCfg(
        name="knee_limit",
        sensor_type="jointlimitpos",
        obj=ObjRef(type="joint", name="knee_joint", entity="robot"),
        cutoff=0.1,
    )

    # Relative frame position (end-effector w.r.t. base).
    BuiltinSensorCfg(
        name="ee_pos",
        sensor_type="framepos",
        obj=ObjRef(type="body", name="end_effector", entity="robot"),
        ref=ObjRef(type="body", name="base", entity="robot"),
    )


Auto-discovery
^^^^^^^^^^^^^^

Sensors already defined in an entity's XML are automatically discovered
during scene composition and prefixed with the entity name. There is no
need to create a ``BuiltinSensorCfg`` for these.

.. code-block:: xml

    <!-- In robot.xml -->
    <sensor>
        <accelerometer name="trunk_imu" site="imu_site"/>
        <jointpos name="hip_sensor" joint="hip_joint"/>
    </sensor>

.. code-block:: python

    # Access by prefixed name.
    imu = env.scene["robot/trunk_imu"]
    hip = env.scene["robot/hip_sensor"]


ContactSensor
-------------

Each physics step, MuJoCo produces a flat, unstructured list of contact
pairs across the entire scene. A single foot geom might generate several
simultaneous contacts with the ground, interleaved with contacts from
other entities. ``ContactSensor`` filters this raw list to the pairs you
care about, reduces multiple contacts per element down to a fixed count,
and packages the result into clean, batched tensors your policy can
consume directly. It builds on MuJoCo's native
`contact sensor <https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact>`_.

Primary and secondary
^^^^^^^^^^^^^^^^^^^^^

Contacts are pairwise: you typically want to know "did the robot's feet
touch the terrain?", not just "did something touch something."
``primary`` defines the elements you are measuring (the feet).
``secondary`` optionally restricts what they are contacting (the
terrain). When ``secondary`` is ``None``, any contact with a primary
element counts.

Each side is specified with a ``ContactMatch``. The ``mode`` selects the
MuJoCo element type (``"geom"``, ``"body"``, or ``"subtree"``) and the
``pattern`` accepts a regex or tuple of regexes matched against element
names within the entity.

.. code-block:: python

    from mjlab.sensor import ContactSensorCfg, ContactMatch

    # Foot geoms contacting the terrain body.
    ContactSensorCfg(
        name="feet_ground",
        primary=ContactMatch(
            mode="geom", pattern=r".*_foot$", entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
    )

    # Self-collision: pelvis subtree against itself.
    ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree", pattern="pelvis", entity="robot",
        ),
        secondary=ContactMatch(
            mode="subtree", pattern="pelvis", entity="robot",
        ),
        fields=("found",),
    )

Reduction
^^^^^^^^^

A single foot geom can have many simultaneous contacts with the ground.
Policies typically need one representative contact per element, not the
raw list. The ``reduce`` mode selects which contacts to keep.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Behavior
   * - ``"none"``
     - Fast, non-deterministic selection
   * - ``"mindist"``
     - Closest contacts by penetration depth
   * - ``"maxforce"``
     - Strongest contacts by force magnitude
   * - ``"netforce"``
     - Sums all contacts into a single synthetic contact at the
       force-weighted centroid with the net wrench

``num_slots`` controls how many contacts are retained per primary
element after reduction. The default is 1, which gives one
representative contact per primary match. The output shape is
``[B, N * num_slots, ...]`` where ``N`` is the number of primary
matches. With ``reduce="netforce"``, the output is always one contact
per primary regardless of ``num_slots``.

Fields
^^^^^^

The ``fields`` tuple selects which contact quantities to extract. Only
requested fields are allocated; the rest are ``None`` on the output
dataclass. Available fields are ``"found"``, ``"force"``,
``"torque"``, ``"dist"``, ``"pos"``, ``"normal"``, and ``"tangent"``.

Air time tracking
^^^^^^^^^^^^^^^^^

Locomotion tasks often need to know when feet land and take off for
gait rewards. Setting ``track_air_time=True`` enables per-element
timing. The sensor maintains four additional tensors on
``ContactData``: ``current_air_time``, ``last_air_time``,
``current_contact_time``, and ``last_contact_time``. Two helper
methods provide edge detection for transition events.

.. code-block:: python

    sensor = env.scene["feet_air"]
    first_contact = sensor.compute_first_contact(dt)  # Just landed
    first_air = sensor.compute_first_air(dt)           # Just took off

Output
^^^^^^

``ContactData`` is a dataclass whose fields correspond to the
``fields`` tuple on the config. Unrequested fields are ``None``.

.. code-block:: python

    @dataclass
    class ContactData:
        found: Tensor | None     # [B, N] contact count
        force: Tensor | None     # [B, N, 3]
        torque: Tensor | None    # [B, N, 3]
        dist: Tensor | None      # [B, N] penetration depth
        pos: Tensor | None       # [B, N, 3] contact position
        normal: Tensor | None    # [B, N, 3] surface normal
        tangent: Tensor | None   # [B, N, 3]

        # With track_air_time=True.
        current_air_time: Tensor | None
        last_air_time: Tensor | None
        current_contact_time: Tensor | None
        last_contact_time: Tensor | None


RayCastSensor
-------------

``RayCastSensor`` provides GPU-accelerated raycasting for terrain
scanning and depth sensing. It supports grid and pinhole camera ray
patterns with configurable alignment modes. See :ref:`raycast_sensor`
for full documentation.


RGB-D Camera
------------

``CameraSensor`` renders RGB and depth images from MuJoCo cameras. See
:ref:`rgbd_camera` for full documentation.


Extending: custom sensors
-------------------------

All sensors inherit from ``Sensor[T]``, a generic base class where
``T`` is the data type returned by the ``data`` property (e.g.,
``torch.Tensor`` for ``BuiltinSensor``, ``ContactData`` for
``ContactSensor``).

The base class provides automatic per-step caching. The ``data``
property calls ``_compute_data()`` on first access each step and
caches the result. The cache is invalidated automatically when
``update()`` or ``reset()`` is called, so multiple reads within the
same step (from different observation or reward terms) pay the
computation cost only once.

**Lifecycle methods:**

- ``edit_spec``: Add sensor elements to the MjSpec during scene
  construction.
- ``initialize``: Post-compilation setup. Cache sensor indices,
  allocate buffers, resolve references.
- ``update``: Called each physics step. Invalidates the data cache.
  Override to maintain per-step state (e.g., air time counters).
- ``reset``: Called on environment reset. Invalidates the data cache.
  Override to clear per-environment state.
- ``_compute_data``: Compute and return the sensor output. Called
  lazily by the ``data`` property when the cache is stale.

``ContactSensor`` and ``RayCastSensor`` are the most complete
reference implementations for custom sensor development.

.. toctree::
   :maxdepth: 1
   :hidden:

   raycast_sensor
   rgbd_camera
