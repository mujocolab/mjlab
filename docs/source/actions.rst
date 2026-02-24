.. _actions:

Actions
=======

Actions define how the policy controls the simulation. The action
manager receives the policy's output tensor each step, splits it across
registered action terms, and routes each slice to the appropriate
entity's actuators. Each term maps a contiguous segment of the policy
output to a control mode (position, velocity, effort) on a set of
joints, tendons, or sites.

.. code-block:: python

    from mjlab.envs.mdp.actions import JointPositionActionCfg

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),   # regex matching actuator names
            scale=0.5,
            use_default_offset=True,  # action 0 = default pose
        ),
    }


Common parameters
-----------------

All action types share a base set of parameters inherited from
``BaseActionCfg``.

``entity_name`` identifies the scene entity to control. ``actuator_names``
is a tuple of regex patterns matched against actuator (or tendon/site)
names to select the controlled targets.

``scale`` multiplies the raw policy output before any offset is applied.
It accepts a scalar or a dict mapping actuator name patterns to
per-target values. This keeps policy outputs in a normalized range while
mapping to physically meaningful units. ``offset`` is added after
scaling; joint action types also provide ``use_default_offset``, which
automatically loads the entity's default joint positions or velocities
as the offset so that a raw output of zero produces the default pose.

``clip`` optionally clamps the processed action before it reaches the
actuator.

Actions are written to actuator targets on every decimation substep
(physics step), not just once per policy step. This is in contrast to
observation delay, which operates in units of policy steps.


Action types
------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Type
     - Description
   * - ``JointPositionAction``
     - Sets joint position targets. With ``use_default_offset=True``
       (the default), a policy output of zero commands the default pose.
       Encoder bias from ``dr.encoder_bias`` is subtracted automatically
       so that randomized offsets propagate correctly to the control
       command.
   * - ``JointVelocityAction``
     - Sets joint velocity targets. ``use_default_offset=True`` uses the
       default joint velocities (typically zero).
   * - ``JointEffortAction``
     - Sets joint effort (torque) targets directly. No default offset.
   * - ``TendonLengthAction``
     - Sets tendon length targets. Targets are resolved by matching
       ``actuator_names`` against tendon names.
   * - ``TendonVelocityAction``
     - Sets tendon velocity targets.
   * - ``TendonEffortAction``
     - Sets tendon effort targets.
   * - ``SiteEffortAction``
     - Applies forces and torques at named sites. Useful for
       quadrotors and drones where thrust is applied at rotor sites
       rather than through joint actuators.


Task-space actions
------------------

``DifferentialIKAction`` converts Cartesian position and orientation
commands into joint-space position targets via damped least-squares
inverse kinematics. One IK step is executed per decimation substep, so
the end-effector tracks the target continuously across substeps rather
than only at policy frequency.

The action dimension is selected automatically based on configuration:

- ``orientation_weight == 0``: **3D** (position only)
- ``orientation_weight > 0, use_relative_mode=True``: **6D** (delta
  position + delta axis-angle)
- ``orientation_weight > 0, use_relative_mode=False``: **7D** (absolute
  position + quaternion)

All objectives (position, orientation, joint limits, posture) are
stacked into a single DLS system. Setting a weight to zero disables
that objective with no overhead in the solve.

The ``compute_dq()`` method returns joint displacements without writing
to actuator targets, enabling multi-iteration IK in standalone scripts
outside of RL training.


Action dimensions and history
------------------------------

The total action dimension presented to the policy is the sum of each
registered term's ``action_dim``. For joint, tendon, and site actions
this equals the number of matched targets. For ``DifferentialIKAction``
it is 3, 6, or 7 depending on the active objectives.

The action manager tracks the three most recent action vectors:
``action``, ``prev_action``, and ``prev_prev_action``. Observation terms
such as ``last_action`` and reward terms such as ``action_rate_l2`` and
``action_acc_l2`` read from these buffers. Action history is zeroed on
environment reset so that episode boundaries do not leak information.


Multiple action terms
---------------------

An environment can register any number of terms. The action manager
concatenates their dimensions in registration order, splits the
policy's output tensor at the corresponding boundaries, and routes
each slice independently.

.. code-block:: python

    from mjlab.envs.mdp.actions import (
        JointPositionActionCfg,
        JointVelocityActionCfg,
    )

    actions = {
        "arm_joints": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*_arm_.*",),
            scale=0.5,
        ),
        "wheel_joints": JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=(".*_wheel_.*",),
            scale=10.0,
        ),
    }

The policy outputs a tensor whose width equals the total number of
matched targets across all terms. Terms can also target different
entities, for example one term for a robot and another for an object
being manipulated.
