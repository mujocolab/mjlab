.. _events:

Events
======

The event manager executes hooks at specific points in the environment
lifecycle. Any logic that should run at startup, on episode reset, or at
regular intervals during training is registered as an event term. Common
examples include resetting entities to an initial state, applying domain
randomization to model parameters, pushing the robot with random velocity
perturbations, and initializing robot state from a reference motion clip.
All of these are configured through the same ``EventTermCfg`` interface,
differing only in the ``mode`` field that controls when each term fires.

Domain randomization, one of the most common uses of events, has its own
dedicated reference page. See :ref:`domain_randomization` for the full
``dr`` module, available functions, and internals.

.. code-block:: python

    from mjlab.envs.mdp import events as event_fns, dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    events = {
        # Reset all entities to their default state each episode.
        "reset_scene": EventTermCfg(
            func=event_fns.reset_scene_to_default,
            mode="reset",
        ),
        # Randomize foot friction once at startup.
        "foot_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=[".*foot.*"]),
                "ranges": (0.3, 1.2),
                "operation": "abs",
            },
        ),
        # Push the robot at random intervals during the episode.
        "push_robot": EventTermCfg(
            func=event_fns.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            },
        ),
    }


Lifecycle modes
---------------

The ``mode`` field on ``EventTermCfg`` determines when the term fires. The
three modes correspond to the three timescales of an RL training run: once
at process startup, once per episode, and periodically within an episode.

``"startup"``
    Fires once during environment initialization, after all managers are
    constructed. Every environment receives the event simultaneously. This
    mode is intended for parameters that should differ across environments
    but remain fixed for the entire training run, such as link masses or
    joint armatures randomized via the ``dr`` module.

``"reset"``
    Fires on every episode reset, for each environment being reset. This is
    the most common mode. State initialization (writing the robot back to
    its default pose) and episode-level domain randomization both belong
    here.

    The optional ``min_step_count_between_reset`` field prevents the term
    from firing too frequently when episodes are very short. The term is
    skipped for any environment that has not taken at least that many steps
    since its last trigger. The first invocation always fires regardless.

``"interval"``
    Fires at regular time intervals during training, independent of episode
    boundaries. The trigger frequency is controlled by ``interval_range_s``,
    a ``(min, max)`` range in seconds. After each trigger the manager
    samples a new wait time uniformly from that range. Each environment has
    its own independent timer by default; setting ``is_global_time=True``
    synchronizes all environments to a single shared timer. Interval events
    are the natural home for mid-episode perturbations such as external
    pushes or drifting model parameters.

As with all manager terms, ``func`` points to the callable and ``params``
holds keyword arguments forwarded to it alongside ``env`` and ``env_ids``.
Any ``SceneEntityCfg`` values inside ``params`` are resolved once at
manager construction (regex patterns are matched to model indices at that
point, not on every call). Terms can be plain functions or classes; see
:ref:`env-config-term-pattern` for the general pattern.


Built-in event functions
------------------------

The functions below are available in ``mjlab.envs.mdp.events``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - ``reset_scene_to_default``
     - Resets all entities to their default states: root pose and velocity
       for floating-base entities, mocap pose for fixed-base entities, and
       joint positions and velocities for articulated entities. Environment
       origins are applied automatically. This is the default event on
       ``ManagerBasedRlEnvCfg``; most environments keep it and add
       additional terms alongside it.
   * - ``reset_root_state_uniform``
     - Resets a single entity's root pose and velocity with uniform random
       offsets from the default. Accepts ``pose_range`` and
       ``velocity_range`` dictionaries with keys ``"x"``, ``"y"``,
       ``"z"``, ``"roll"``, ``"pitch"``, ``"yaw"``. Orientation
       perturbations compose with the default quaternion. For fixed-base
       robots, this is the only way to position them at their environment
       origins; without it they stack at the world origin.
   * - ``reset_root_state_from_flat_patches``
     - Places an entity on a randomly chosen flat terrain patch based on
       the environment's assigned terrain level and type. Falls back to
       ``reset_root_state_uniform`` when no flat patches are available.
       Useful for locomotion tasks where robots should spawn on level
       ground within their assigned sub-terrain.
   * - ``reset_joints_by_offset``
     - Resets joint positions and velocities by adding a uniform random
       offset to the entity's defaults, clamped to soft joint limits.
   * - ``push_by_setting_velocity``
     - Adds a random velocity increment to the entity's current root
       velocity, simulating an external push. Typically used with
       ``mode="interval"`` to test disturbance rejection.
   * - ``apply_external_force_torque``
     - Applies random forces and torques to one or more bodies via the
       MuJoCo external wrench mechanism.
   * - ``randomize_terrain``
     - Assigns each environment to a random sub-terrain row and column,
       ignoring the curriculum. Useful for evaluation or play mode.


Writing custom event terms
--------------------------

An event function takes ``env`` and ``env_ids`` as its first two arguments
and any additional parameters from ``EventTermCfg.params``. It modifies
simulation state in place and returns nothing. For terms that need
expensive one-time setup (such as loading data from disk), use a class
so that the setup runs once at construction rather than on every call.
For example, the following custom event term resets the robot to a
random pose sampled from a pre-recorded dataset:

.. code-block:: python

    import torch
    from mjlab.managers.manager_base import ManagerTermBase
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    class ResetFromDataset(ManagerTermBase):
        """Reset the robot to a random pose from a dataset."""

        def __init__(self, cfg, env):
            super().__init__(env)
            self._robot = env.scene["robot"]
            self._poses = torch.load(
                cfg.params["dataset_path"],
                map_location=env.device,
            )

        def __call__(self, env, env_ids, **kwargs):
            # Sample with replacement: each env gets an independent pose.
            indices = torch.randint(
                len(self._poses), (len(env_ids),), device=env.device,
            )
            self._robot.write_joint_position_to_sim(
                self._poses[indices], env_ids=env_ids,
            )

When a term needs to maintain state or perform expensive setup, implement
it as a class. See :ref:`env-config-term-pattern` for the general
pattern. For custom DR terms that write to model fields, see
:ref:`domain_randomization`.
