.. _commands:

Commands
========

Commands specify what the policy should achieve at each moment: a target
velocity, a reference trajectory, a goal position. The command manager
generates these signals, resamples them at configurable intervals, and
passes them to the policy through the observation system.


Registration
------------

Commands are registered in ``ManagerBasedRlEnvCfg`` as a dictionary
mapping string names to ``CommandTermCfg`` instances. Unlike the
function-based terms used by other managers, every command term is a
class that inherits from ``CommandTerm``.

The ``resampling_time_range`` field controls how often the command
changes. After each resample the term draws a new timer value uniformly
from the given ``(min, max)`` range in seconds. Commands are also
resampled unconditionally on every episode reset.

.. code-block:: python

    commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-0.5, 0.5),
            ),
        ),
    }

The ``generated_commands`` observation function reads the current
command tensor by name and passes it to the policy:

.. code-block:: python

    ObservationTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "twist"},
    )

If the environment has no commands, the manager no-ops all operations
and returns empty tensors. There is no special handling required.


Included command terms
----------------------

Each task ships with its own command terms tailored to its objective.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Term
     - Description
   * - ``UniformVelocityCommand``
     - Generates planar velocity commands ``[v_x, v_y, omega_z]``
       sampled uniformly from configurable ranges. Supports a standing
       mode (fraction of environments receive zero velocity) and a
       heading mode (yaw rate replaced by a proportional controller
       tracking a sampled heading angle). Used by the velocity task.
   * - ``LiftingCommand``
     - Generates a 3D target position for a manipulated object.
       Supports fixed and dynamic difficulty modes. Tracks metrics
       including position error and episode success rate. Used by the
       manipulation task.
   * - ``MotionCommand``
     - Streams reference joint positions, velocities, and body poses
       from a pre-recorded ``.npz`` motion clip. Supports three
       start-frame sampling modes: ``"start"`` (always frame 0),
       ``"uniform"`` (random), and ``"adaptive"`` (biased toward
       difficult regions). At reset the robot is initialized from the
       sampled frame with optional perturbations. Used by the tracking
       task.

Each term can render debug visualizations in the interactive viewer
when ``debug_vis=True`` is set in the configuration. The image below
shows the ghost visualization from ``MotionCommand``, which renders a
translucent copy of the robot at the reference pose alongside the
actual robot.

.. figure:: _static/ghost_visualization.png
   :align: center
   :width: 100%

   Viser visualization of the commanded reference motion for the G1 tracking task.


Writing custom command terms
-----------------------------

A custom command term is a class inheriting from ``CommandTerm`` paired
with a configuration dataclass inheriting from ``CommandTermCfg``. The
term must implement four methods: ``_resample_command(env_ids)`` to
sample new goals, ``_update_command()`` for per-step updates,
``_update_metrics()`` for logging, and a ``command`` property returning
the current goal tensor. The base class manages the resampling timer
and reset logic automatically.

The configuration must implement a ``build(env)`` method that
constructs the paired term instance.
