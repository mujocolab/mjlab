.. _environment_config:

Environment Configuration
=========================

A single ``ManagerBasedRlEnvCfg`` dataclass fully specifies an mjlab
environment: the physical world, the agent's interface to it, and the
MDP defined on top. Because everything lives in one flat
object, an environment can be inspected, copied, and modified without
navigating a class hierarchy.

For a broad orientation to mjlab before reading this page, start with
:ref:`architecture_overview`.


.. _env-config-skeleton:

Annotated skeleton
------------------

The complete set of fields on ``ManagerBasedRlEnvCfg`` is shown below with
inline comments. The fields marked with ``...`` must be provided; all others
have defaults.

.. code-block:: python

    from dataclasses import dataclass, field

    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.managers.action_manager import ActionTermCfg
    from mjlab.managers.command_manager import CommandTermCfg
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene.scene import SceneCfg
    from mjlab.sim.sim import SimulationCfg
    from mjlab.viewer.viewer_config import ViewerConfig


    @dataclass
    class MyEnvCfg(ManagerBasedRlEnvCfg):

        # --- Physics ---

        decimation: int = 4
        # Number of physics steps per policy step.
        # Environment step duration = sim.mujoco.timestep * decimation.

        sim: SimulationCfg = field(default_factory=SimulationCfg)
        # Physics parameters: timestep, integrator, solver, contact settings.
        # Default timestep is 0.002 s (500 Hz). Override with MujocoCfg.

        scene: SceneCfg = ...
        # Terrain, entities, and sensors. Also sets num_envs.
        # Required; there is no default.

        # --- Episode ---

        episode_length_s: float = 20.0
        # Episode duration in seconds.
        # Steps = ceil(episode_length_s / (sim.mujoco.timestep * decimation)).

        is_finite_horizon: bool = False
        # False (default): time limit is an artificial cutoff. The agent
        #   receives a truncated signal and bootstraps value beyond the limit.
        # True: time limit defines the task boundary. The agent receives a
        #   terminal done signal with no future value beyond it.

        scale_rewards_by_dt: bool = True
        # When True (default), each reward term is multiplied by step_dt so
        # that cumulative episodic sums are invariant to simulation frequency.
        # Set to False for algorithms that expect unscaled reward signals.

        # --- Managers ---

        observations: dict[str, ObservationGroupCfg] = field(default_factory=dict)
        # Observation groups. Each key is a group name (e.g. "actor", "critic").
        # Groups can differ in noise, history, delay, and concatenation.

        actions: dict[str, ActionTermCfg] = field(default_factory=dict)
        # Action terms. Each term controls one slice of the policy output
        # and routes it to a specific entity's actuators.

        rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
        # Reward terms. The manager computes a weighted sum each step.

        terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
        # Termination conditions. If empty, episodes never terminate early.
        # Add a time_out term to enforce the episode length limit.

        events: dict[str, EventTermCfg] = field(
            default_factory=lambda: {
                "reset_scene_to_default": EventTermCfg(
                    func=reset_scene_to_default,
                    mode="reset",
                )
            }
        )
        # Event terms for domain randomization and state resets.
        # The default includes reset_scene_to_default, which resets all
        # entities to their initial pose each episode. Override this dict
        # to replace or extend the default reset behavior.

        commands: dict[str, CommandTermCfg] = field(default_factory=dict)
        # Command generators (e.g. velocity targets for locomotion).
        # Commands are resampled at configurable intervals and on reset.

        curriculum: dict[str, CurriculumTermCfg] = field(default_factory=dict)
        # Curriculum terms that adjust training conditions based on performance.

        metrics: dict[str, MetricsTermCfg] = field(default_factory=dict)
        # Custom metrics logged as episode averages alongside reward terms.

        # --- Misc ---

        seed: int | None = None
        # Random seed for reproducibility. If None, a random seed is chosen
        # and stored back into this field after initialization.

        viewer: ViewerConfig = field(default_factory=ViewerConfig)
        # Camera position, resolution, and tracking target for rendering.


.. _env-config-term-pattern:

Term configuration pattern
--------------------------

All manager dictionaries follow the same pattern. Each entry maps a string
name to a term configuration object. The configuration always carries at
minimum a ``func`` field pointing to the callable that implements the term,
and a ``params`` dict of extra keyword arguments forwarded to that callable.

The manager calls ``func(env, **params)`` each step (or ``term(env, **params)``
when ``func`` is a class that has been instantiated). Term names are arbitrary;
they appear in training logs and are used only for identification.

.. rubric:: Reward terms

.. code-block:: python

    from mjlab.envs import mdp
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    rewards = {
        "alive": RewardTermCfg(
            func=mdp.is_alive,
            weight=1.0,
        ),
        "joint_torques": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-1e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "action_rate": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.1,
        ),
    }

``weight`` scales the function's output before it is summed into the total
reward. Negative weights produce penalties.

``params`` maps to keyword arguments of the function. For example,
``mdp.joint_torques_l2(env, asset_cfg=...)`` receives ``asset_cfg`` from the
``params`` dict. Any argument not listed in ``params`` must have a default
value in the function signature.

.. rubric:: Termination terms

.. code-block:: python

    from mjlab.envs import mdp
    from mjlab.managers.termination_manager import TerminationTermCfg

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,   # marks this as a truncation, not a failure
        ),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.22},   # ~70 degrees in radians
        ),
    }

The ``time_out`` flag on ``TerminationTermCfg`` tells the manager to treat
this condition as a truncation rather than a terminal failure. Truncations
map to the ``truncated`` signal in the Gym interface; failures map to
``terminated``. This distinction matters for value bootstrapping in RL
algorithms.

.. rubric:: Event terms

.. code-block:: python

    from mjlab.managers.event_manager import EventTermCfg

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"yaw": (-3.14, 3.14)},
                "velocity_range": {},
            },
        ),
    }

The ``mode`` field on ``EventTermCfg`` controls when the term fires:
at startup, on episode reset, or at regular intervals. See :ref:`events`
for the full treatment of lifecycle modes, built-in event functions, and
the relationship between events and domain randomization.

.. rubric:: Function-based vs. class-based terms

Terms can be plain functions or classes. Functions are suitable for stateless
computations; classes are useful when a term needs to cache expensive setup or
maintain state across steps.

A function-based term has the signature ``func(env, **params) -> Tensor``. A
class-based term is instantiated once with ``(cfg, env)`` and then called with
the same signature. Classes can optionally implement a ``reset(env_ids)`` hook
for per-episode state clearing.

.. code-block:: python

    # Function-based (stateless)
    RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.01)

    # Class-based (caches joint indices at init)
    class MyReward:
        def __init__(self, cfg, env):
            self.joint_ids = resolve_joint_ids(cfg.params, env)

        def __call__(self, env) -> torch.Tensor:
            return compute_reward(env, self.joint_ids)

    RewardTermCfg(func=MyReward, weight=1.0)


.. _env-config-timing:

Timing: decimation, timestep, and episode length
-------------------------------------------------

Three parameters jointly determine the temporal structure of the environment.

``sim.mujoco.timestep``
    The physics integration step in seconds. The default is 0.002 s (500 Hz).
    This is one of the most important parameters in any environment: smaller
    values produce more stable physics but slow down simulation. See the MuJoCo
    `performance tuning <https://mujoco.readthedocs.io/en/stable/modeling.html#performance-tuning>`_
    guide for practical advice on choosing timesteps and solver settings.

``decimation``
    The number of physics steps executed per policy step. The policy runs at
    ``1 / (timestep * decimation)`` Hz.

``episode_length_s``
    The episode duration in seconds. The maximum number of policy steps per
    episode is ``ceil(episode_length_s / (timestep * decimation))``.

**Concrete example.** The velocity task uses ``timestep=0.005`` (200 Hz
physics) and ``decimation=4``, giving a policy frequency of 50 Hz. With
``episode_length_s=20.0``, each episode runs for exactly 1000 policy steps.

.. code-block:: python

    physics_dt  = 0.005        # seconds per physics step (200 Hz)
    decimation  = 4            # physics steps per policy step
    step_dt     = 0.005 * 4   # = 0.02 s per policy step (50 Hz)
    episode_len = 20.0 / 0.02  # = 1000 policy steps per episode

To read these values at runtime, use the environment properties:

.. code-block:: python

    env.physics_dt          # = cfg.sim.mujoco.timestep
    env.step_dt             # = cfg.sim.mujoco.timestep * cfg.decimation
    env.max_episode_length  # steps (int)
    env.max_episode_length_s  # seconds (float)

When ``scale_rewards_by_dt=True`` (the default), each reward term is
multiplied by ``step_dt`` before being returned. A reward function that
returns a constant value of 1.0 contributes ``step_dt`` per step and
approximately ``episode_length_s`` over a full episode, regardless of how
``decimation`` and ``timestep`` are set. Changing the simulation frequency
without disabling this scaling leaves reward magnitudes unchanged.


.. _env-config-subclassing:

Subclassing pattern
-------------------

mjlab uses plain dataclass inheritance rather than deeply nested class
hierarchies. To build a task-specific configuration, subclass
``ManagerBasedRlEnvCfg`` and override fields.

The recommended approach is to define the full configuration in a factory
function, then call it from robot-specific configs that override only the
fields that differ. The velocity task uses this pattern: ``make_velocity_env_cfg``
returns a fully assembled ``ManagerBasedRlEnvCfg``, and each robot
configuration calls the factory and patches in robot-specific values such
as the scene, joint name patterns, and action scale.

A condensed version of the factory illustrates the full assembly pattern:

.. code-block:: python

    import math
    from dataclasses import replace

    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp import dr
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.tasks.velocity import mdp
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.terrains.config import ROUGH_TERRAINS_CFG
    from mjlab.viewer import ViewerConfig


    def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:

        observations = {
            "actor": ObservationGroupCfg(
                terms={
                    "base_lin_vel": ObservationTermCfg(
                        func=mdp.builtin_sensor,
                        params={"sensor_name": "robot/imu_lin_vel"},
                    ),
                    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                    "command": ObservationTermCfg(
                        func=mdp.generated_commands,
                        params={"command_name": "twist"},
                    ),
                    # additional terms omitted for brevity
                },
                concatenate_terms=True,
                enable_corruption=True,
            ),
            "critic": ObservationGroupCfg(
                terms={...},
                concatenate_terms=True,
                enable_corruption=False,
            ),
        }

        actions = {
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.5,
                use_default_offset=True,
            )
        }

        commands = {
            "twist": UniformVelocityCommandCfg(
                entity_name="robot",
                resampling_time_range=(3.0, 8.0),
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-1.0, 1.0),
                    ang_vel_z=(-0.5, 0.5),
                    heading=(-math.pi, math.pi),
                ),
            )
        }

        events = {
            "reset_base": EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {},
                },
            ),
            "foot_friction": EventTermCfg(
                mode="startup",
                func=dr.geom_friction,
                params={
                    "asset_cfg": SceneEntityCfg("robot", geom_names=[]),
                    "operation": "abs",
                    "ranges": (0.3, 1.2),
                },
            ),
            "push_robot": EventTermCfg(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(1.0, 3.0),
                params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
            ),
        }

        rewards = {
            "track_linear_velocity": RewardTermCfg(
                func=mdp.track_linear_velocity,
                weight=2.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            ),
            "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
            "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        }

        terminations = {
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "fell_over": TerminationTermCfg(
                func=mdp.bad_orientation,
                params={"limit_angle": math.radians(70.0)},
            ),
        }

        return ManagerBasedRlEnvCfg(
            decimation=4,
            episode_length_s=20.0,
            sim=SimulationCfg(
                nconmax=35,
                njmax=1500,
                mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
            ),
            scene=SceneCfg(
                terrain=TerrainEntityCfg(
                    terrain_type="generator",
                    terrain_generator=replace(ROUGH_TERRAINS_CFG),
                    max_init_terrain_level=5,
                ),
                num_envs=1,
            ),
            observations=observations,
            actions=actions,
            commands=commands,
            events=events,
            rewards=rewards,
            terminations=terminations,
        )

Robot-specific configs call this factory and patch fields using
``dataclasses.replace`` or direct assignment. Common per-robot overrides
include ``scene`` (to add the robot entity and sensors), joint name patterns
inside ``SceneEntityCfg``, action ``scale``, and body names for reward terms.

.. note::

   Isaac Lab uses deeply nested ``__post_init__`` overrides for configuration
   inheritance. mjlab avoids that pattern: each ``ManagerBasedRlEnvCfg`` is a
   flat, inspectable dataclass. A misspelled field name raises a ``TypeError``
   at construction rather than silently creating a new attribute. See
   :ref:`migration_isaac_lab` for a full comparison.


Where to go next
----------------

The remaining pages in the Manager Layer section cover each manager in
detail:

- :ref:`observations`: observation groups, the processing pipeline
  (clip, scale, noise, delay, history), and built-in observation functions.
- :ref:`actions`: action types and how the action manager routes policy
  output to actuators.
- :ref:`rewards`: reward terms and scaling by dt.
- :ref:`terminations`: episode end conditions and the truncation/failure
  distinction.
- :ref:`commands`: command generators and goal-conditioned task setup.
- :ref:`events`: the event manager lifecycle (startup, reset, interval).
- :ref:`domain_randomization`: the full ``dr`` module for domain
  randomization.
- :ref:`curriculum`: difficulty progression based on policy performance.
- :ref:`metrics`: custom per-step metrics logged as episode averages.
