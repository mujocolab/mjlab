.. _motion-imitation:

运动模仿
========

mjlab 可以训练人形机器人策略模仿参考运动。本页介绍运动数据预处理与
训练。

WandB registry 准备
-------------------

mjlab 使用 `Weights & Biases <https://wandb.ai/>`_ 存储和加载参考运动。
预处理任何运动之前，请按照
`BeyondMimic 说明 <https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup>`_
创建 WandB registry（只需完成创建 registry 这一步；跳过其中的
``csv_to_npz.py`` 命令）。

运动预处理
----------

参考运动是按 Unitree 广义坐标约定重定向的 CSV 文件（基座位置、基座
四元数 xyzw，然后是关节角度）。

把 CSV 转换为 mjlab 期望的 NPZ 格式：

.. code-block:: bash

   MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
       --input-file <PATH_TO_CSV> \
       --output-name <MOTION_NAME> \
       --input-fps 30 \
       --output-fps 50 \
       --render True

该脚本在 MuJoCo Warp 中回放运动、为每个 body 计算正向运动学，并把
得到的 NPZ 上传到你的 WandB registry。

.. warning::

   你 **必须** 使用 mjlab 的转换器（``mjlab.scripts.csv_to_npz``）。
   IsaacLab 等其他框架的转换器产出的 NPZ 文件 body 顺序不兼容。NPZ
   按身体编号存储预计算的 body 位置和四元数，不同物理引擎分配 body
   索引的方式不同（MuJoCo 用深度优先遍历，PhysX 用广度优先）。不匹配的
   NPZ 会把跟踪目标映射到错误的 body 上，训练不会收敛。

训练
----

.. code-block:: bash

   uv run train Mjlab-Tracking-Flat-Unitree-G1 \
       --registry-name your-org/motions/motion-name \
       --env.scene.num-envs 4096

评估
----

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1 \
       --wandb-run-path your-org/mjlab/run-id
