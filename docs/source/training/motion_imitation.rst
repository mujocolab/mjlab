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

The script resamples the motion to the output frame rate and uploads the
resulting NPZ to your WandB registry.

.. warning::

   You **must** use mjlab's converter (``mjlab.scripts.csv_to_npz``). The NPZ
   holds ``root_pos_w``, ``root_quat_w`` (wxyz), ``root_lin_vel_w``,
   ``root_ang_vel_w``, ``joint_pos``, ``joint_vel``, ``joint_names`` and
   ``fps``, and the tracking task computes the tracked bodies from them with
   forward kinematics. Loading raises unless ``joint_names`` equal the robot's
   joint names, so NPZ files from other frameworks do not load. The robot needs
   a free-joint root, one degree of freedom per joint and only hinge joints on
   the paths to the tracked bodies.

Training
--------

.. code-block:: bash

   uv run train Mjlab-Tracking-Flat-Unitree-G1 \
       --registry-name your-org/motions/motion-name \
       --env.scene.num-envs 4096

Multiple clips
--------------

``MotionCommandCfg.motion_file`` takes one NPZ file, a list of NPZ files, a glob
or a packed directory. On the command line, ``train`` and ``play`` take a file,
a glob or a packed directory. Each episode tracks one clip and ends as a
time-out on that clip's last frame. All clips share one ``fps``. With several
clips, the exported ONNX file holds the policy without the motion and maps
``obs`` to ``actions``.

``csv_to_npz`` writes each converted clip to ``/tmp/motion.npz``. Copy it to
``<MOTION_DIR>/<CLIP_NAME>.npz``; the file name is the clip's name.

.. code-block:: bash

   uv run train Mjlab-Tracking-Flat-Unitree-G1 \
       --env.commands.motion.motion-file '<MOTION_DIR>/*.npz' \
       --env.scene.num-envs 4096

A run trained from local files links no motion artifact, so ``play`` needs the
motions as well:

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1 \
       --wandb-run-path your-org/mjlab/run-id \
       --motion-file '<MOTION_DIR>/*.npz'

A glob opens every clip at startup. For a dataset of many clips, pack them once
into a directory and point ``motion_file`` at it:

.. code-block:: bash

   uv run -m mjlab.scripts.pack_motions \
       --input '<MOTION_DIR>/*.npz' \
       --output <PACKED_DIR>

Evaluation
----------

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1 \
       --wandb-run-path your-org/mjlab/run-id
