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
   * - ``reward_weight``
     - Ramps a reward term's weight up or down according to training
       step thresholds. Useful for introducing penalty terms gradually
       after the policy has learned basic locomotion.


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
