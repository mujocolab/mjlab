.. _installation:

安装指南
========

本指南介绍几种不同的安装路径，你可以选择最适合自己用例的一种。

.. contents::
   :local:
   :depth: 1

.. note::

    **系统要求**

    - **训练**：Linux + NVIDIA GPU（推荐 CUDA 12.4+）
    - **评估**：Linux、macOS 或 Windows (WSL)
    - **Python**：3.10 或更高版本

    有关支持范围的更多细节见 :ref:`faq`。


如何选择安装方式？
------------------

选择与你的使用方式最匹配的卡片。

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 方式 1 - 将 mjlab 作为依赖使用 (uv)
      :link: install-uv-dependency
      :link-type: ref

      你在自己的、由 ``uv`` 管理的项目中 **将 mjlab 作为依赖使用**。
      **（推荐大多数用户选择）**

   .. grid-item-card:: 方式 2 - 开发 / 贡献 (uv)
      :link: install-uv-develop
      :link-type: ref

      你直接在 mjlab 仓库内 **试用 mjlab** 或 **为 mjlab 本身做贡献**，
      环境由 ``uv`` 管理。

   .. grid-item-card:: 方式 3 - 传统 pip / venv / conda
      :link: install-pip
      :link-type: ref

      你使用 **传统工具** （``pip`` / ``venv`` / ``conda``）， **不使用 uv**。

   .. grid-item-card:: 方式 4 - Docker / 集群
      :link: install-docker
      :link-type: ref

      你在 **容器或集群中运行**，倾向于 **基于 Docker** 的部署方式。


.. _install-uv-dependency:

方式 1 - 将 mjlab 作为依赖使用 (uv)
-----------------------------------

这是我们推荐的使用 ``mjlab`` 的方式。你拥有自己的项目，希望用 ``uv``
把 ``mjlab`` 作为依赖引入。

1. 安装 uv
^^^^^^^^^^

如果尚未安装 ``uv``，请运行：

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

2. 初始化你的项目
^^^^^^^^^^^^^^^^^

初始化一个受管理的 Python 项目：

.. code-block:: bash

   # Create a new package-based project
   uv init --package my_mjlab_project
   cd my_mjlab_project

3. 添加 mjlab 依赖
^^^^^^^^^^^^^^^^^^

将 ``mjlab`` 添加为依赖有几种方式。我们推荐使用 PyPI 上的最新稳定版本；
如果需要最新特性，可以使用 GitHub 直装；如果需要使用你本地开发的特性，
则用本地可编辑安装。这几种方式可以互换，随时切换。

.. tab-set::

   .. tab-item:: PyPI

      进入你的项目后，安装 PyPI 上的最新版本：

      .. code:: bash

         uv add mjlab

   .. tab-item:: Source

      进入你的项目后，无需克隆即可从 GitHub 直接安装：

      .. code:: bash

         uv add "mjlab @ git+https://github.com/mujocolab/mjlab"

   .. tab-item:: Local

      克隆仓库：

      .. code:: bash

         git clone https://github.com/mujocolab/mjlab.git

      进入你的项目后，将其添加为可编辑依赖：

      .. code:: bash

         uv add --editable /path/to/cloned/mjlab

.. tip::

   想了解如何组织"自定义机器人 + 现有 ``mjlab`` 任务"的项目结构，
   可以参考
   `ANYmal C Velocity Tracking <https://github.com/mujocolab/anymal_c_velocity>`_
   仓库中的完整示例。

验证
^^^^

安装完成后，运行 demo 验证 ``mjlab`` 是否正常工作：

.. code-block:: bash

   uv run demo


.. _install-uv-develop:

方式 2 - 开发 / 贡献 (uv)
-------------------------

此方式适用于开发 ``mjlab`` 本身或为项目做贡献。

.. code:: bash

   git clone https://github.com/mujocolab/mjlab.git && cd mjlab
   uv sync

验证
^^^^

安装完成后，运行 demo 验证 ``mjlab`` 是否正常工作：

.. code-block:: bash

   uv run demo


.. _install-pip:

方式 3 - 传统 pip / venv / conda
--------------------------------

激活你的虚拟环境（``venv``、``conda`` 等），然后安装：

.. code:: bash

   pip install mjlab


验证
^^^^

安装完成后，运行 demo 验证 ``mjlab`` 是否正常工作：

.. code-block:: bash

   demo


.. _install-docker:

方式 4 - Docker / 集群
----------------------

前置条件：

- 安装 Docker：`Docker 安装指南 <https://docs.docker.com/engine/install/>`_。
- 为你的系统安装合适的 NVIDIA 驱动以及
  `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_。

  - 务必按 NVIDIA 安装指南中 Docker 配置一节的说明注册容器运行时并重启。

.. tab-set::

   .. tab-item:: 预构建镜像（推荐）

      从 GitHub Container Registry 拉取并运行最新镜像：

      .. code-block:: bash

         docker run --rm --runtime=nvidia --gpus all \
           ghcr.io/mujocolab/mjlab uv run demo

      该镜像在每次 push 到 ``main`` 时重新构建。

   .. tab-item:: 本地构建

      从源码构建并运行：

      .. code-block:: bash

         ./scripts/run_docker.sh uv run demo


遇到问题了？
------------

1. **查阅 FAQ**

    常见安装与运行问题请先查阅 mjlab 的 :ref:`faq`

2. **仍然没有解决？**

    在 GitHub 上提交 issue：https://github.com/mujocolab/mjlab/issues
