.. _rewards:

Rewards
=======

Rewards are the training signal that shapes policy behavior. Each reward
term is a function that returns a per-environment scalar every step. The
reward manager computes a weighted sum of all terms and returns it to
the training framework.

Each term is registered by name with a ``RewardTermCfg`` that carries
the callable and a ``weight``. Negative weights produce penalties.
Additional keyword arguments are supplied through ``params``.

.. code-block:: python

    from mjlab.envs.mdp import rewards
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    rewards_cfg = {
        "alive": RewardTermCfg(func=rewards.is_alive, weight=1.0),
        "joint_torques": RewardTermCfg(
            func=rewards.joint_torques_l2,
            weight=-1e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
    }


Built-in reward functions
-------------------------

The functions below are available in ``mjlab.envs.mdp.rewards`` and are
shared across tasks. Individual tasks also define their own reward
functions tailored to the task objective (e.g. velocity tracking for
locomotion). All reward functions return a tensor of shape
``[num_envs]``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Function
     - Description
   * - ``is_alive``
     - Returns ``1.0`` for environments that have not terminated this
       step. Use with a positive weight as a survival bonus.
   * - ``is_terminated``
     - Returns ``1.0`` for environments that terminated due to a
       non-timeout condition. Use with a negative weight to penalize
       failure.
   * - ``joint_torques_l2``
     - Sum of squared actuator forces. Penalizes energy-intensive
       actions.
   * - ``joint_vel_l2``
     - Sum of squared joint velocities.
   * - ``joint_acc_l2``
     - Sum of squared joint accelerations.
   * - ``action_rate_l2``
     - Sum of squared differences between the current and previous
       action. Penalizes rapid changes in the policy output.
   * - ``action_acc_l2``
     - Sum of squared second-order action differences. Penalizes
       high-frequency jitter in the action signal.
   * - ``joint_pos_limits``
     - Penalty for joint positions exceeding the soft limits. Zero
       when all joints are within limits.
   * - ``posture`` *(class)*
     - Exponential kernel measuring deviation from the default joint
       positions: ``exp(-mean(error^2 / std^2))``.
   * - ``electrical_power_cost`` *(class)*
     - Sum of positive mechanical power consumed by actuators.
       Regenerative power is not penalized.
   * - ``flat_orientation_l2``
     - Sum of squares of the x and y components of the projected
       gravity vector in the base frame. Zero when perfectly upright.


Reward scaling by dt
--------------------

``ManagerBasedRlEnvCfg.scale_rewards_by_dt`` is ``True`` by default.
When enabled, the reward manager multiplies each term by the environment
step duration before accumulating it. This makes episodic reward totals
invariant to simulation frequency: a task running at 50 Hz produces the
same expected episode return as the same task at 200 Hz, because each
step contributes proportionally less to the total.

Per-term episodic sums are logged as ``Episode_Reward/<term_name>`` and
are always divided by the episode duration, giving a reward rate that is
comparable across runs with different episode lengths.


Writing custom reward functions
-------------------------------

A reward function accepts ``env`` as its first argument and returns a
``[num_envs]`` tensor. Additional parameters are declared as function
arguments and supplied via ``RewardTermCfg(params={...})``. When a term
needs to cache setup work or maintain per-episode state, implement it as
a class. See :ref:`env-config-term-pattern` for the general pattern.
