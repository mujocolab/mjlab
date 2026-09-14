欢迎使用 mjlab！
================

.. figure:: source/_static/mjlab-banner.jpg
   :width: 100%
   :alt: mjlab

mjlab 是一个轻量级的开源机器人学习框架，它将 GPU 加速仿真与可组合的
环境结合在一起，并把上手成本降到最低。它采用
`Isaac Lab <https://github.com/isaac-sim/IsaacLab>`_ 提出的基于管理器的
API——用户以模块化积木的方式组合观测、奖励和事件——并搭配
`MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`_ 实现
GPU 加速物理。最终成果是一个可以用一条命令安装、依赖精简、并且能直接
访问 `MuJoCo <https://github.com/google-deepmind/mujoco>`_ 原生数据结构
的框架。

**核心特性：**

- **可组合的环境：** 用户把观测、奖励、终止等 MDP 项定义为模块化的
  构建积木
- **极简依赖：** 通过 ``uv`` 一条命令安装，启动延迟低
- **直接访问 MuJoCo 数据结构：** 原生 ``MjModel``/``MjData`` 访问，
  没有任何转换层
- **PyTorch 原生：** 观测、奖励和动作都是 PyTorch 张量，底层由零拷贝
  GPU 显存共享支撑

想进一步了解 mjlab 背后的设计决策，见 :doc:`source/motivation`。

**立即尝鲜**（无需安装）：

.. code-block:: bash

   uvx --from mjlab --refresh demo

目录
----

.. toctree::
   :maxdepth: 1
   :caption: 用户指南

   source/installation
   source/tutorials
   source/contributing

.. toctree::
   :maxdepth: 1
   :caption: 核心概念

   source/architecture_overview
   source/entity/index
   source/actuators
   source/sensors/index
   source/scene
   source/terrain

.. toctree::
   :maxdepth: 1
   :caption: 管理器层

   source/environment_config
   source/observations
   source/actions
   source/rewards
   source/terminations
   source/commands
   source/events
   source/randomization
   source/curriculum
   source/metrics
   source/recorders

.. toctree::
   :maxdepth: 1
   :caption: 训练与调试

   source/training/rsl_rl
   source/viewers
   source/training/distributed_training
   source/training/cloud
   source/debugging/nan_guard
   source/debugging/export_scene

.. toctree::
   :maxdepth: 2
   :caption: API 参考

   source/api/index

.. toctree::
   :maxdepth: 1
   :caption: 延伸阅读

   source/motivation
   source/migration_isaac_lab
   source/faq
   source/research
   source/changelog

许可证与引用
------------

mjlab 基于 Apache License 2.0 许可证发布。
详情请参阅 `LICENSE 文件 <https://github.com/mujocolab/mjlab/blob/main/LICENSE/>`_。

如果你在研究中使用了 mjlab，我们恳请引用：

.. code-block:: bibtex

    @article{Zakka_mjlab_A_Lightweight_2026,
        author = {Zakka, Kevin and Liao, Qiayuan and Yi, Brent and Le Lay, Louis and Sreenath, Koushil and Abbeel, Pieter},
        title = {{mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning}},
        url = {https://arxiv.org/abs/2601.22074},
        year = {2026}
    }

致谢
----

mjlab 的诞生离不开 Isaac Lab 团队的出色工作——mjlab 正是构建在他们的
API 设计与抽象之上的。

同样感谢 MuJoCo Warp 团队——尤其是 Erik Frey 和 Taylor Howell——他们
无数次解答我们的问题、提供有价值的反馈，并根据我们的需求实现了诸多特性。
