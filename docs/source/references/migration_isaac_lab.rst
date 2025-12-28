.. _migration-isaaclab:

Migrating from Isaac Lab
========================

.. warning::

   This guide is a work in progress. As more users migrate, we will update this
   page with additional patterns and edge cases. If something is not covered,
   please open an issue on GitHub or start a discussion:

   - Issues: https://github.com/mujocolab/mjlab/issues
   - Discussions: https://github.com/mujocolab/mjlab/discussions

TL;DR
-----

Most Isaac Lab *manager-based* task configs can be ported to ``mjlab`` with
only small changes:

- The overall **MDP structure is the same** (managers for rewards, observations,
  actions, commands, terminations, events, curriculum).
- The **environment base classes are similar**, but naming is slightly
  different.
- The biggest change is **configuration style**: Isaac Lab uses nested
  ``@configclass`` definitions; ``mjlab`` uses dictionaries of config objects.

If you are familiar with Isaac Lab's manager-based API, migration is mostly
mechanical.

Key Differences
---------------

1. Import Paths
~~~~~~~~~~~~~~~

Isaac Lab:

.. code-block:: python

   from isaaclab.envs import ManagerBasedRLEnv

mjlab:

.. code-block:: python

   from mjlab.envs import ManagerBasedRlEnvCfg

.. note::

   ``mjlab`` uses a consistent ``CamelCase`` naming convention (for example,
   ``RlEnv`` instead of ``RLEnv``).

2. Configuration Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Lab uses nested ``@configclass`` blocks for manager terms. ``mjlab``
instead uses **plain dictionaries** mapping names to config objects, which makes
it easy to construct variants, merge configs, or generate them programmatically.

**Isaac Lab:**

.. code-block:: python

   @configclass
   class RewardsCfg:
       """Reward terms for the MDP."""

       motion_global_anchor_pos = RewTerm(
           func=mdp.motion_global_anchor_position_error_exp,
           weight=0.5,
           params={"command_name": "motion", "std": 0.3},
       )
       motion_global_anchor_ori = RewTerm(
           func=mdp.motion_global_anchor_orientation_error_exp,
           weight=0.5,
           params={"command_name": "motion", "std": 0.4},
       )

**mjlab:**

.. code-block:: python

   rewards = {
       "motion_global_anchor_pos": RewardTermCfg(
           func=mdp.motion_global_anchor_position_error_exp,
           weight=0.5,
           params={"command_name": "motion", "std": 0.3},
       ),
       "motion_global_anchor_ori": RewardTermCfg(
           func=mdp.motion_global_anchor_orientation_error_exp,
           weight=0.5,
           params={"command_name": "motion", "std": 0.4},
       ),
   }

   cfg = ManagerBasedRlEnvCfg(
       scene=scene,
       rewards=rewards,
       # ... other manager dictionaries:
       # observations=..., actions=..., commands=..., terminations=...,
       # events=..., curriculum=...
   )

This pattern applies to all managers:

- ``rewards``
- ``observations``
- ``actions``
- ``commands``
- ``terminations``
- ``events``
- ``curriculum``

3. Scene Configuration
~~~~~~~~~~~~~~~~~~~~~~

Scene setup is **simpler** in ``mjlab``:

- No Omniverse / USD scene graph, no ``prim_path`` management.
- Assets are pure MuJoCo (MJCF) with modifier dataclasses applied to
  ``mujoco.MjSpec``.
- Lights, materials, textures, and sensors are configured as part of
  ``SceneCfg`` and robot configs.

**Isaac Lab:**

.. code-block:: python

   from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
   from isaaclab.scene import InteractiveSceneCfg
   from isaaclab.sensors import ContactSensorCfg
   from isaaclab.terrains import TerrainImporterCfg
   import isaaclab.sim as sim_utils
   from isaaclab.assets import ArticulationCfg, AssetBaseCfg

   @configclass
   class MySceneCfg(InteractiveSceneCfg):
       """Configuration for the terrain scene with a legged robot."""

       # ground terrain
       terrain = TerrainImporterCfg(
           prim_path="/World/ground",
           terrain_type="plane",
           collision_group=-1,
           physics_material=sim_utils.RigidBodyMaterialCfg(
               friction_combine_mode="multiply",
               restitution_combine_mode="multiply",
               static_friction=1.0,
               dynamic_friction=1.0,
           ),
           visual_material=sim_utils.MdlFileCfg(
               mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
               project_uvw=True,
           ),
       )
       # lights
       light = AssetBaseCfg(
           prim_path="/World/light",
           spawn=sim_utils.DistantLightCfg(
               color=(0.75, 0.75, 0.75), intensity=3000.0
           ),
       )
       sky_light = AssetBaseCfg(
           prim_path="/World/skyLight",
           spawn=sim_utils.DomeLightCfg(
               color=(0.13, 0.13, 0.13), intensity=1000.0
           ),
       )
       robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

**mjlab:**

.. code-block:: python

   from dataclasses import replace

   from mjlab.scene import SceneCfg
   from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
   from mjlab.utils.spec_config import ContactSensorCfg
   from mjlab.terrains import TerrainImporterCfg

   # Configure contact sensor
   self_collision_sensor = ContactSensorCfg(
       name="self_collision",
       subtree1="pelvis",
       subtree2="pelvis",
       data=("found",),
       reduce="netforce",
       num=10,  # report up to 10 contacts
   )

   # Add sensor to robot config
   g1_cfg = replace(get_g1_robot_cfg(), sensors=(self_collision_sensor,))

   # Create scene
   SCENE_CFG = SceneCfg(
       terrain=TerrainImporterCfg(terrain_type="plane"),
       entities={"robot": g1_cfg},
   )

Key changes:

- No USD ``prim_path`` or cloning; the scene is described directly in MuJoCo.
- Materials, lights, and visual properties are applied via
  ``MjSpec``-modifier dataclasses.
- See ``mjlab.utils.spec_config`` in the repository for helpers that apply
  these changes for you.
- ``asset_name`` has been unified to ``entity_name`` across all configurations.

Complete Example Comparison
---------------------------

A good way to learn the pattern is to compare concrete tasks that have already
been ported:

- Isaac Lab implementation (Beyond Mimic):

  - https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py

- mjlab implementation:

  - https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py

You will see that:

- Manager dictionaries in ``mjlab`` mirror Isaac Lab's config classes,
- Reward, observation, command, and termination logic is almost identical,
- Scene and asset setup are simplified to pure MuJoCo.

Migration Checklist
-------------------

Use this as a quick checklist when porting a task:

1. **Base class and imports**

   - Replace Isaac Lab imports (for example,
     ``from isaaclab.envs import ManagerBasedRLEnv``) with the corresponding
     ``mjlab`` imports (for example,
     ``from mjlab.envs import ManagerBasedRlEnvCfg``).

2. **Manager configuration**

   - Convert each Isaac Lab ``@configclass`` manager (``RewardsCfg``,
     ``ObservationsCfg``, etc.) into a dictionary of config objects.
   - Pass these dictionaries into ``ManagerBasedRlEnvCfg``.

3. **Scene and assets**

   - Replace ``InteractiveSceneCfg`` with a ``SceneCfg`` instance.
   - Replace USD / ``prim_path`` logic with MuJoCo asset configs and scene
     entities (for example, a robot from ``asset_zoo``).

4. **Sensors and contact handling**

   - Convert Isaac Lab ``ContactSensorCfg`` to
     ``mjlab.utils.spec_config.ContactSensorCfg`` and attach it to the robot
     config.

5. **RL entry points**

   - Make sure your training script or entry point uses the correct task id and
     environment config (for example, via Gymnasium registration or direct
     construction, depending on how your project is structured).

Tips and Support
----------------

1. Check the examples in the repository under:

   - ``src/mjlab/tasks/``

2. If you get stuck:

   - Open an issue: https://github.com/mujocolab/mjlab/issues
   - Start a discussion: https://github.com/mujocolab/mjlab/discussions

3. Keep in mind MuJoCo vs Isaac Sim differences:

   - Some Omniverse / USD rendering features do not have direct equivalents.
   - Focus first on matching the **physics and observations**, then polish
     visuals if needed.
