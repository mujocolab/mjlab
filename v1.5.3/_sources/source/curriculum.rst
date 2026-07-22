.. _curriculum:

Curriculum
==========

The curriculum manager adjusts training conditions based on policy
performance. Training begins with an easier problem and difficulty
increases as the policy demonstrates it can handle the current
conditions. Common uses include advancing robots to harder terrain,
widening command velocity ranges, and ramping reward penalty weights
over the course of training.

Curriculum terms are called at each environment reset. Each term
receives the environment and the set of resetting environment IDs,
examines some performance signal, and applies changes to environment
parameters directly.

.. code-block:: python

    from mjlab.managers.curriculum_manager import CurriculumTermCfg

    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "twist"},
        ),
    }

The return value of a curriculum function is logged under
``Curriculum/<term_name>`` in the training metrics.


Built-in curriculum functions
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Function
     - Description
   * - ``terrain_levels_vel``
     - Measures how far each robot traveled during the episode. Robots
       that covered enough distance move up one difficulty row in the
       terrain grid; those that fell short move down. See the terrain
       curriculum section below.
   * - ``commands_vel``
     - Widens velocity command ranges based on training step count.
       Each stage specifies a step threshold and the new ranges to
       apply once that threshold is exceeded.
   * - ``reward_curriculum``
     - Adjusts a reward term's weight and/or params according to
       training step thresholds. Replaces the older ``reward_weight``
       function and also supports modifying reward function parameters.
   * - ``termination_curriculum``
     - Adjusts a termination term's params according to training step
       thresholds. Useful for gradually tightening termination
       conditions (e.g. energy limits) as training progresses.


Reward curriculum
-----------------

``reward_curriculum`` schedules changes to a reward term's weight or
keyword arguments as training progresses. Each stage specifies a
``step`` threshold and an optional ``weight`` or ``params`` update.
Stages are evaluated in order, and each one whose threshold has been
reached is applied.

**Ramping a penalty weight**

A common pattern is to introduce a penalty term at low weight early in
training and increase it once the policy has learned the basics:

.. code-block:: python

    from mjlab.managers.curriculum_manager import CurriculumTermCfg

    curriculum = {
        "joint_vel_hinge_weight": CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={
                "reward_name": "joint_vel_hinge",
                "stages": [
                    {"step": 0, "weight": -0.01},
                    {"step": 12000, "weight": -0.1},
                    {"step": 24000, "weight": -1.0},
                ],
            },
        ),
    }

**Adjusting reward parameters**

You can also change the parameters passed to the reward function. For
example, tightening a tracking tolerance as training progresses:

.. code-block:: python

    curriculum = {
        "track_lin_vel_tighten": CurriculumTermCfg(
            func=mdp.reward_curriculum,
            params={
                "reward_name": "track_linear_velocity",
                "stages": [
                    {"step": 0, "params": {"std": 0.5}},
                    {"step": 20000, "params": {"std": 0.3}},
                    {"step": 50000, "params": {"std": 0.1}},
                ],
            },
        ),
    }

**Combining weight and params**

A single stage can update both weight and params at once:

.. code-block:: python

    {"step": 24000, "weight": -1.0, "params": {"max_vel": 1.0}}


Termination curriculum
----------------------

``termination_curriculum`` schedules changes to a termination term's
parameters as training progresses. This is useful for gradually
tightening termination conditions once the policy has learned basic
behaviors.

**Tightening an energy limit**

Start with a permissive energy threshold and reduce it over training:

.. code-block:: python

    from mjlab.managers.curriculum_manager import CurriculumTermCfg

    curriculum = {
        "energy_threshold": CurriculumTermCfg(
            func=mdp.termination_curriculum,
            params={
                "termination_name": "energy",
                "stages": [
                    {"step": 12000, "params": {"threshold": 1000.0}},
                    {"step": 24000, "params": {"threshold": 700.0}},
                    {"step": 36000, "params": {"threshold": 400.0}},
                ],
            },
        ),
    }

The ``time_out`` field on ``TerminationTermCfg`` can also be toggled
via stages if needed, though this is uncommon in practice.


Terrain curriculum
------------------

The terrain grid used with procedural terrain is a
``num_rows x num_cols`` matrix of patches. Columns represent terrain
type variants; rows represent difficulty levels, with row 0 being the
easiest and row ``num_rows - 1`` the hardest. When
``TerrainGeneratorCfg.curriculum=True``, each column is assigned exactly
one terrain type so that difficulty increases monotonically along rows.

At environment construction each environment is assigned a random
starting row within ``[0, max_init_terrain_level]``. The
``terrain_levels_vel`` curriculum term promotes or demotes environments
on each reset based on distance traveled during the episode.
Environments that reach the maximum level are randomly reassigned to
any row, maintaining coverage across all difficulty levels. See
:ref:`terrain` for details on configuring the terrain grid itself.


Writing custom curriculum functions
------------------------------------

A curriculum function accepts ``env`` and ``env_ids``, applies
parameter changes, and returns a value to log (a scalar tensor, a dict
of tensors, or ``None``). A typical implementation reads a performance
metric, decides whether to increase or decrease difficulty, mutates the
relevant configuration in place, and returns the current difficulty
level. See :ref:`env-config-term-pattern` for the general pattern.
