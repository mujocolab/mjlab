.. _export-scene:

导出场景
========

``export-scene`` 脚本把完整场景（XML 与网格资产）写到一个目录，便于
检视、分享或在独立 MuJoCo 中加载。

快速上手
--------

.. code-block:: bash

    # Export a built-in entity by alias.
    uv run export-scene g1 --output-dir /tmp/g1

    # Export a registered task scene.
    uv run export-scene Mjlab-Velocity-Flat-Unitree-Go1 --output-dir /tmp/task

    # Export as a zip archive.
    uv run export-scene yam --output-dir /tmp/yam --zip True

    # Export a custom entity via import path.
    uv run export-scene my_pkg.robots:get_my_robot_cfg --output-dir /tmp/custom

输出目录包含一个 ``scene.xml`` 和一个 ``assets/`` 子目录（含所有被
引用的网格文件）。XML 可以直接用 ``mujoco.MjModel.from_xml_path()``
加载，或拖进
`simulate 查看器 <https://mujoco.readthedocs.io/en/stable/programming/samples.html#sasimulate>`_。

目标解析
--------

位置参数 ``target`` 按以下顺序解析：

1. **任务 ID**：对照任务注册表检查（``import mjlab.tasks``）。
2. **实体别名**：内置简写之一（``g1``、``go1``、``yam``）。
3. **导入路径**：形如 ``module:attribute`` 的字符串，指向任何返回
   ``EntityCfg`` 的可调用对象。

都不匹配时，脚本会打印可用的任务 ID 和别名。

选项
----

``--output-dir DIR`` *（默认 "export"）*
    目标目录。每次导出前清空，防止残留旧资产。

``--zip True`` *（默认 False）*
    把输出压缩为 ``.zip`` 归档并删除目录。
