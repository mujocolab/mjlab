.. _actuators:

Actuators
=========

Actuators convert high-level commands (position, velocity, effort) into
low-level efforts that drive joints. They are configured through the
``articulation`` field of :ref:`EntityCfg <entity>`. mjlab provides
**built-in** actuators that leverage the physics engine's implicit
integration for best stability, and **explicit** actuators for custom
control laws and actuator dynamics.


Quick start
-----------

Basic PD control with ``BuiltinPositionActuator``, the most common
starting point.

.. code-block:: python

    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

    robot_cfg = EntityCfg(
        spec_fn=lambda: load_robot_spec(),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*_hip_.*", ".*_knee_.*"),
                    stiffness=80.0,
                    damping=10.0,
                    effort_limit=100.0,
                ),
            ),
        ),
    )

Wrap any actuator in ``DelayedActuatorCfg`` to model communication
latency.

.. code-block:: python

    from mjlab.actuator import DelayedActuatorCfg, BuiltinPositionActuatorCfg

    DelayedActuatorCfg(
        base_cfg=BuiltinPositionActuatorCfg(
            target_names_expr=(".*",),
            stiffness=80.0,
            damping=10.0,
        ),
        delay_target="position",
        delay_min_lag=2,  # Minimum 2 physics steps
        delay_max_lag=5,  # Maximum 5 physics steps
    )


Built-in vs explicit actuators
------------------------------

The key design decision when configuring actuators is whether to use
**built-in** or **explicit** types. The difference comes down to how
MuJoCo's integrator handles velocity-dependent forces.

**Built-in actuators** (``BuiltinPositionActuator``,
``BuiltinVelocityActuator``, ``BuiltinMotorActuator``,
``BuiltinMuscleActuator``) create native MuJoCo actuator elements in the
MjSpec. The physics engine computes the control law and integrates
velocity-dependent damping forces implicitly. This provides the best
numerical stability, particularly with high gains or large timesteps.

**Explicit actuators** (``IdealPdActuator``, ``DcMotorActuator``,
``LearnedMlpActuator``) compute torques in user code and forward them
through a ``<motor>`` actuator acting as a passthrough. Because the
integrator cannot account for the velocity derivatives of these
externally computed forces, they are less numerically robust than built-in
types. Use explicit actuators when you need custom control laws or actuator
dynamics that cannot be expressed with built-in types (e.g.,
velocity-dependent torque limits, learned actuator networks).

The two approaches match closely in the linear, unconstrained regime at
small timesteps. At larger timesteps or higher gains, built-in actuators
are more forgiving.

**Integrator choice.** mjlab places damping inside the actuator rather than
in joints. The ``euler`` integrator treats joint damping implicitly but
actuator damping explicitly, limiting stability. The ``implicitfast``
integrator treats all known velocity-dependent forces implicitly, handling
both proportional and damping terms of the actuator without additional cost.

.. note::

     mjlab defaults to ``implicitfast``, as it is MuJoCo's recommended
     integrator and provides superior stability for actuator-side damping.


Actuator types
--------------

All actuator configs share a few common fields inherited from
``ActuatorCfg``:

- ``target_names_expr``: Tuple of regex patterns matched against joint
  names (or tendon/site names when using a different
  ``transmission_type``).
- ``armature``: Reflected rotor inertia added to the target joint.
- ``frictionloss``: Static friction (stiction) modeled as a constraint
  on the target joint. See MuJoCo's
  `frictionloss <https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-frictionloss>`_.

Built-in actuators
^^^^^^^^^^^^^^^^^^

Built-in actuators use MuJoCo's native actuator types via the MjSpec API.

**BuiltinPositionActuator**: Creates ``<position>`` actuators for PD
control.

**BuiltinVelocityActuator**: Creates ``<velocity>`` actuators for velocity
control.

**BuiltinMotorActuator**: Creates ``<motor>`` actuators for direct torque
control.

**BuiltinMuscleActuator**: Creates ``<muscle>`` actuators for
biologically-inspired muscle dynamics with force-length-velocity
characteristics.

.. code-block:: python

    from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg

    # Mobile manipulator: PD for arm joints, velocity control for wheels.
    actuators = (
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"),
            stiffness=100.0,
            damping=10.0,
            effort_limit=150.0,
        ),
        BuiltinVelocityActuatorCfg(
            target_names_expr=(".*_wheel_.*",),
            damping=20.0,
            effort_limit=50.0,
        ),
    )


Explicit actuators
^^^^^^^^^^^^^^^^^^

Explicit actuators compute efforts and forward them to an underlying
``<motor>`` actuator acting as a passthrough. See
`Built-in vs explicit actuators`_ above for stability implications.

**IdealPdActuator**: Implements an ideal PD controller. Computes torques
as ``tau = Kp * pos_error + Kd * vel_error``.

**DcMotorActuator**: Extends ``IdealPdActuator`` with velocity-dependent
torque saturation to model DC motor torque-speed curves (back-EMF
effects). Implements a linear torque-speed curve: maximum torque at zero
velocity, zero torque at maximum velocity.

**LearnedMlpActuator**: Neural network-based actuator that uses a
trained MLP to predict torque outputs from joint state history. Useful
when analytical models cannot capture complex actuator dynamics like
delays, nonlinearities, and friction effects. Inherits DC motor
velocity-based torque limits.

.. code-block:: python

    from mjlab.actuator import IdealPdActuatorCfg, DcMotorActuatorCfg

    # Ideal PD for hips, DC motor model with torque-speed curve for knees.
    actuators = (
        IdealPdActuatorCfg(
            target_names_expr=(".*_hip_.*",),
            stiffness=80.0,
            damping=10.0,
            effort_limit=100.0,
        ),
        DcMotorActuatorCfg(
            target_names_expr=(".*_knee_.*",),
            stiffness=80.0,
            damping=10.0,
            effort_limit=25.0,       # Continuous torque limit
            saturation_effort=50.0,  # Peak torque at stall
            velocity_limit=30.0,     # No-load speed (rad/s)
        ),
    )


XML actuators
^^^^^^^^^^^^^

XML actuators wrap actuators already defined in your robot's XML file. The
config finds existing actuators by matching their ``target`` joint name
against the ``target_names_expr`` patterns. Each joint must have exactly one
matching actuator.

**XmlPositionActuator**: Wraps existing ``<position>`` actuators

**XmlVelocityActuator**: Wraps existing ``<velocity>`` actuators

**XmlMotorActuator**: Wraps existing ``<motor>`` actuators

.. code-block:: python

    from mjlab.actuator import XmlPositionActuatorCfg

    # Robot XML already has:
    # <actuator>
    #   <position name="hip_joint" joint="hip_joint" kp="100"/>
    # </actuator>

    # Wrap existing XML actuators.
    actuators = (
        XmlPositionActuatorCfg(target_names_expr=("hip_joint",)),
    )

Delayed actuator
^^^^^^^^^^^^^^^^

Generic wrapper that adds command delays to any actuator. Useful for
modeling actuator latency and communication delays. The delay operates on
command targets before they reach the actuator's control law.

.. code-block:: python

    from mjlab.actuator import DelayedActuatorCfg, IdealPdActuatorCfg

    # Add 2-5 step delay to position commands.
    actuators = (
        DelayedActuatorCfg(
            base_cfg=IdealPdActuatorCfg(
                target_names_expr=(".*",),
                stiffness=80.0,
                damping=10.0,
            ),
            delay_target="position",     # Delay position commands
            delay_min_lag=2,
            delay_max_lag=5,
            delay_hold_prob=0.3,         # 30% chance to keep previous lag
            delay_update_period=10,      # Update lag every 10 steps
        ),
    )


**Multi-target delays:**

.. code-block:: python

    DelayedActuatorCfg(
        base_cfg=IdealPdActuatorCfg(...),
        delay_target=("position", "velocity", "effort"),
        delay_min_lag=2,
        delay_max_lag=5,
    )

Delays are quantized to physics timesteps. For example, with 500Hz physics
(2ms/step), ``delay_min_lag=2`` represents a 4ms minimum delay.

.. note::

     Each target gets an independent delay buffer with its own lag
     schedule. This provides maximum flexibility for modeling different
     latency characteristics for position, velocity, and effort commands.


Authoring actuator configs
--------------------------

Since actuator parameters are uniform within each config, use separate
actuator configs for joints that need different parameters:

.. code-block:: python

    from mjlab.actuator import BuiltinPositionActuatorCfg

    # G1 humanoid with different gains per joint group.
    G1_ACTUATORS = (
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_.*", "waist_yaw_joint"),
            stiffness=180.0,
            damping=18.0,
            effort_limit=88.0,
            armature=0.0015,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("left_hip_pitch_joint", "right_hip_pitch_joint"),
            stiffness=200.0,
            damping=20.0,
            effort_limit=88.0,
            armature=0.0015,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_knee_joint",),
            stiffness=150.0,
            damping=15.0,
            effort_limit=139.0,
            armature=0.0025,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_ankle_.*",),
            stiffness=40.0,
            damping=5.0,
            effort_limit=25.0,
            armature=0.0008,
        ),
    )

This design choice reflects a deliberate simplification in mjlab: each
``ActuatorCfg`` represents a single actuator type (e.g., a specific
motor/gearbox model) applied uniformly across all joints it drives.
Hardware parameters such as ``armature`` (reflected rotor inertia) and
``gear`` describe properties of the actuator hardware, even though they
are implemented in MuJoCo as joint or actuator fields. In other frameworks
(like Isaac Lab), these fields may accept ``float | dict[str, float]`` to
support per-joint variation. mjlab instead encourages one config per
actuator type or per joint group, keeping the hardware model physically
consistent and explicit. The main trade-off is verbosity in special cases,
such as parallel linkages, where per-joint overrides could have been
convenient, but the benefit is clearer semantics and simpler maintenance.

See :ref:`actions` for how action terms route policy outputs to actuators
(including DifferentialIK for task-space control), and
:ref:`domain_randomization` for randomizing gains and effort limits.


Computing hardware parameters
------------------------------

This section is relevant when configuring actuators from real motor
datasheets. If you are using manually tuned gains, you can skip ahead.

mjlab provides utilities in ``mjlab.utils.actuator`` to compute actuator
parameters from physical motor specifications. This is particularly
useful for computing reflected inertia (``armature``) and deriving
appropriate control gains from hardware datasheets.

**Example: Unitree G1 motor configuration**

.. code-block:: python

    from math import pi

    from mjlab.utils.actuator import (
        reflected_inertia_from_two_stage_planetary,
        ElectricActuator
    )

    # Motor specs from manufacturer datasheet.
    ROTOR_INERTIAS_7520_14 = (
        0.489e-4,  # Motor rotor inertia (kg*m**2)
        0.098e-4,  # Planet carrier inertia
        0.533e-4,  # Output stage inertia
    )
    GEARS_7520_14 = (
        1,            # First stage (motor to planet)
        4.5,          # Second stage (planet to carrier)
        1 + (48/22),  # Third stage (carrier to output)
    )

    # Compute reflected inertia at joint output.
    # J_reflected = J_motor*(N1*N2)**2 + J_carrier*N2**2 + J_output.
    ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
        ROTOR_INERTIAS_7520_14, GEARS_7520_14
    )

    # Create motor spec container.
    ACTUATOR_7520_14 = ElectricActuator(
        reflected_inertia=ARMATURE_7520_14,
        velocity_limit=32.0,   # rad/s at joint
        effort_limit=88.0,     # N*m continuous torque
    )

    # Derive PD gains from natural frequency and damping ratio.
    NATURAL_FREQ = 10 * 2*pi  # 10 Hz bandwidth.
    DAMPING_RATIO = 2.0       # Overdamped, see note below.
    STIFFNESS = ARMATURE_7520_14 * NATURAL_FREQ**2
    DAMPING = 2 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ

    # Use in actuator config.
    from mjlab.actuator import BuiltinPositionActuatorCfg

    actuator = BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_pitch_joint",),
        stiffness=STIFFNESS,
        damping=DAMPING,
        effort_limit=ACTUATOR_7520_14.effort_limit,
        armature=ACTUATOR_7520_14.reflected_inertia,
    )

.. note::

     The example uses ``DAMPING_RATIO = 2.0``
     (overdamped) rather than the critically damped value of 1.0. This is
     because the reflected inertia calculation only accounts for the motor's
     rotor inertia, not the apparent inertia of the links being moved. In
     practice, the total effective inertia at the joint is higher than just
     the reflected motor inertia, so using an overdamped ratio provides
     better stability margins when the true system inertia is
     underestimated.

**Parallel linkage approximation:**

For joints driven by parallel linkages (like the G1's ankles with dual
motors), the effective armature in the nominal configuration can be
approximated as the sum of the individual motor armatures:

.. code-block:: python

    # Two 5020 motors driving ankle through parallel linkage.
    G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
        target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
        stiffness=STIFFNESS_5020 * 2,
        damping=DAMPING_5020 * 2,
        effort_limit=ACTUATOR_5020.effort_limit * 2,
        armature=ACTUATOR_5020.reflected_inertia * 2,
    )


Extending: custom actuators
----------------------------

All actuators implement a unified ``compute()`` interface that receives an
``ActuatorCmd`` (containing position, velocity, and effort targets) and
returns control signals for the low-level MuJoCo actuators driving each
joint.

**Core interface:**

.. code-block:: python

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        """Convert high-level commands to control signals.

        Args:
            cmd: Command containing position_target, velocity_target,
                effort_target (each is a [num_envs, num_targets] tensor
                or None)

        Returns:
            Control signals for this actuator
            ([num_envs, num_targets] tensor)
        """

**Lifecycle hooks:**

- ``edit_spec``: Modify MjSpec before compilation (add actuators, set
  gains)
- ``initialize``: Post-compilation setup (resolve indices, allocate
  buffers)
- ``reset``: Per-environment reset logic
- ``update``: Pre-step updates
- ``compute``: Convert commands to control signals

**Properties:**

- ``target_ids``: Tensor of local target indices controlled by this
  actuator
- ``target_names``: List of target names controlled by this actuator
- ``ctrl_ids``: Tensor of global control input indices for this actuator

``IdealPdActuator`` is the recommended base class for custom explicit
actuators. ``DcMotorActuator`` and ``LearnedMlpActuator`` are both
built on top of it and serve as examples of the extension pattern.
