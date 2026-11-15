.. _scene:

Scene
=====

The scene merges entities, terrain, and sensors into a single
simulation. ``SceneCfg`` describes the contents of the world, and the
``Scene`` class handles MJCF composition, compilation, and runtime
state management.

.. code-block:: python

    from mjlab.scene import SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    # A robot on a flat ground plane with 4096 parallel environments.
    scene_cfg = SceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": robot_cfg},
    )

A scene with procedural terrain, sensors, and multiple entities:

.. code-block:: python

    from mjlab.scene import SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.terrains.config import ROUGH_TERRAINS_CFG
    from mjlab.sensor import RayCastSensorCfg, ContactSensorCfg

    scene_cfg = SceneCfg(
        num_envs=4096,
        terrain=TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            max_init_terrain_level=5,
        ),
        entities={
            "robot": robot_cfg,
            "cube": cube_cfg,
        },
        sensors=(
            RayCastSensorCfg(name="terrain_scan", ...),
            ContactSensorCfg(name="feet_contact", ...),
        ),
    )


Composition
-----------

The scene starts from a root ``MjSpec`` and
`attaches <https://mujoco.readthedocs.io/en/stable/python.html#attachment>`_
each entity's spec into it with a unique name prefix. A robot entity named ``"robot"`` has all its
internal MuJoCo elements (bodies, joints, geoms, actuators, sensors)
prefixed with ``robot/``, so ``base_link`` becomes ``robot/base_link``,
``joint0`` becomes ``robot/joint0``, and so on. Prefixing prevents name
collisions when multiple entities share element names and provides a
consistent namespace for observation and reward terms.

Terrain, when present, is attached without a prefix (its elements live
in the global namespace). Sensors are added after entities and can
reference entity elements by their prefixed names.

``scene.compile()`` converts the composed ``MjSpec`` into a single
``MjModel``. The ``Simulation`` class then uploads this model to the
GPU via MuJoCo Warp. After the simulation is created,
``scene.initialize()`` resolves each entity's element indices into the
compiled model, allocates state buffers, and sets up GPU rendering
resources for any camera or raycast sensors.

``scene.to_zip(path)`` exports the compiled model as a ``.zip`` file
for offline inspection in the standalone MuJoCo viewer. Each entity's
initial state keyframe is merged into the export, so the model opens
in its default pose.

At runtime, entities and sensors are accessible by name:

.. code-block:: python

    robot = env.scene["robot"]              # Entity
    scan = env.scene["terrain_scan"]        # Sensor
    contact = env.scene["feet_contact"]     # Sensor

    robot.data.joint_pos                    # [B, num_joints]
    scan.data.distances                     # [B, N]
    contact.data.force                      # [B, N, 3]

Builtin sensors defined in an entity's XML are auto-discovered during
composition and accessible with the entity name prefix:

.. code-block:: python

    imu = env.scene["robot/trunk_imu"]      # Auto-discovered sensor


Environment origins
-------------------

Each environment in MuJoCo Warp is an independent world with its own
state. Environments do not share physical space and cannot interact with
each other. Environment origins exist for two purposes: spreading
entities across the world for visualization (so the viewer shows robots
side by side rather than stacked at the origin), and for locomotion
tasks with procedural terrain, placing each environment at a specific
sub-terrain patch.

**Flat terrain.** Origins form a regular grid centered at the world
origin with ``env_spacing`` meters between neighbors.

**Procedural terrain.** The terrain generator produces a
``num_rows x num_cols`` grid of sub-terrain patches, each with its own
center point. Each environment is assigned to one patch, and the
terrain curriculum system moves environments to harder patches as
performance improves. See :ref:`terrain` for details.

.. note::

   All environments currently share the same ``MjModel`` (identical
   meshes, geometries, and kinematic trees). Heterogeneous simulation,
   where different worlds can have different meshes or geometries, is
   `in progress in MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp/pull/1009>`_.
   mjlab will support this once it lands upstream.

Reset event terms read ``scene.env_origins`` to position entities:

.. code-block:: python

    # Inside a reset event term.
    robot.write_root_pose_to_sim(
        default_root_pose + env_origins[env_ids]
    )

Each origin is marked with an invisible sphere site (geom group 4) that
appears in the MuJoCo viewer when group 4 is enabled, useful for
verifying placement during development.


Custom spec editing
-------------------

Most scenes are fully described by their entities, terrain, and sensors.
Occasionally a modification spans multiple entities. A tendon connecting
a ceiling gantry to the robot, for example, cannot be defined inside
either entity's MJCF because it references sites from both.

The ``spec_fn`` callback on ``SceneCfg`` handles this case. It receives
the fully composed ``MjSpec`` after all entities and sensors have been
attached with their prefixed names, but before compilation:

.. code-block:: python

    import mujoco

    def add_gantry(spec: mujoco.MjSpec):
        spec.worldbody.add_site(name="gantry", pos=(0, 0, 2))
        for side in ["left", "right"]:
            tendon = spec.add_tendon(
                name=f"{side}_rope",
                limited=True,
                range=(0, 1),
            )
            tendon.wrap_site("gantry")
            tendon.wrap_site(f"robot/{side}_hook")

    scene_cfg = SceneCfg(
        entities={"robot": robot_cfg},
        spec_fn=add_gantry,
    )

Other common uses include global equality constraints, custom
visualization geometry, and any modification that requires access to the
fully composed scene.
