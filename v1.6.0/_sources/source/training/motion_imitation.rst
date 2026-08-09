.. _motion-imitation:

Motion Imitation
================

mjlab can train humanoid policies to imitate reference motions. This page
covers motion data preprocessing and training.

WandB registry setup
--------------------

mjlab uses `Weights & Biases <https://wandb.ai/>`_ to store and load
reference motions. Before preprocessing any motions, create a WandB registry
by following the
`BeyondMimic instructions <https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup>`_
(only the registry creation step; skip the ``csv_to_npz.py`` command shown
there).

Motion preprocessing
--------------------

Reference motions are retargeted CSV files in Unitree's generalized
coordinate convention (base position, base quaternion in xyzw, then joint
angles).

Convert a CSV to the NPZ format mjlab expects:

.. code-block:: bash

   MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
       --input-file <PATH_TO_CSV> \
       --output-name <MOTION_NAME> \
       --input-fps 30 \
       --output-fps 50 \
       --render True

The script plays the motion through MuJoCo Warp, computes forward kinematics
for every body, and uploads the resulting NPZ to your WandB registry.

.. warning::

   You **must** use mjlab's converter (``mjlab.scripts.csv_to_npz``).
   Converters from other frameworks such as IsaacLab produce NPZ files with
   incompatible body orderings. The NPZ stores precomputed body positions and
   quaternions indexed by body number, and different physics engines assign
   body indices differently (MuJoCo uses depth first traversal, PhysX uses
   breadth first). A mismatched NPZ will map tracking targets to the wrong
   bodies and training will not converge.

Training
--------

.. code-block:: bash

   uv run train Mjlab-Tracking-Flat-Unitree-G1 \
       --registry-name your-org/motions/motion-name \
       --env.scene.num-envs 4096

Evaluation
----------

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1 \
       --wandb-run-path your-org/mjlab/run-id
