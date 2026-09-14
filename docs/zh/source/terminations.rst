.. _terminations:

Terminations
============

Termination terms define when an episode ends. Each term is a function
that returns a boolean per-environment tensor every step. The
termination manager aggregates all terms and reports the result to the
training framework as either a terminal failure or a truncation.

Each term is registered by name with a ``TerminationTermCfg``. Setting
``time_out=True`` marks the condition as a truncation rather than a
terminal failure. Truncations map to the ``truncated`` signal in the
Gym interface; failures map to ``terminated``. This distinction matters
for value bootstrapping: the agent should estimate future value beyond
a truncation but not beyond a failure.

.. code-block:: python

    from mjlab.envs.mdp import terminations
    from mjlab.managers.termination_manager import TerminationTermCfg

    terminations_cfg = {
        "time_out": TerminationTermCfg(
            func=terminations.time_out, time_out=True,
        ),
        "fallen": TerminationTermCfg(
            func=terminations.bad_orientation,
            params={"limit_angle": 1.0},
        ),
    }


Built-in termination functions
-------------------------------

The functions below are available in ``mjlab.envs.mdp.terminations`` and
are shared across tasks. Individual tasks may define additional
termination functions specific to their objective. All termination
functions return a boolean tensor of shape ``[num_envs]``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Function
     - Description
   * - ``time_out``
     - Returns ``True`` when the episode length reaches
       ``env.max_episode_length``. Register with ``time_out=True`` so
       the manager treats it as a truncation.
   * - ``bad_orientation``
     - Returns ``True`` when the angle between the asset's up axis and
       world up exceeds ``limit_angle`` (radians).
   * - ``root_height_below_minimum``
     - Returns ``True`` when the asset's root link height is below
       ``minimum_height`` (meters).
   * - ``nan_detection``
     - Returns ``True`` when NaN or Inf values appear anywhere in the
       physics state. A safety net to terminate diverged simulations
       cleanly.


Writing custom termination functions
-------------------------------------

Custom termination functions follow the same patterns as reward
functions. A plain function accepts ``env`` and returns a boolean
``[num_envs]`` tensor. See :ref:`env-config-term-pattern` for the
general pattern.
