.. _installation:

Installation Guide
==================

This guide presents different installation paths so you can
choose the one that best fits your use case.

.. contents::
   :local:
   :depth: 1

.. note::

    **System Requirements**

    - **Operating System**: Linux recommended
    - **Python**: 3.10 or higher
    - **GPU**: NVIDIA GPU
    - **CUDA version**: CUDA 12.4+ Recommended

    See :ref:`faq` for more details on what is exactly supported.


How to choose an installation method?
-------------------------------------

Select the card that best matches how you plan to use ``mjlab``.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Method 1 - Use mjlab as a dependency (uv)
      :link: install-uv-dependency
      :link-type: ref

      You are **using mjlab as a dependency** in your own project managed by ``uv``. **(Recommended for most users)**

   .. grid-item-card:: Method 2 - Develop / contribute (uv)
      :link: install-uv-develop
      :link-type: ref

      You are **trying mjlab** or **contributing to mjlab itself** directly from inside the mjlab repository, with ``uv`` managing the environment.

   .. grid-item-card:: Method 3 - Classic pip / venv / conda
      :link: install-pip
      :link-type: ref

      You are using **classic tools** (``pip`` / ``venv`` / ``conda``) and **do not use uv**.

   .. grid-item-card:: Method 4 - Docker / clusters
      :link: install-docker
      :link-type: ref

      You are **running in containers or on clusters** and prefer a **Docker-based** setup.


.. _install-uv-dependency:

Method 1 - Use mjlab as a dependency (uv)
-----------------------------------------

This is our recommended way to use ``mjlab``. You have
your own project and want to use ``mjlab`` as a dependency
using ``uv``.

1. Install uv
^^^^^^^^^^^^^

If you do not have ``uv`` installed, run:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

2. Initialize your project
^^^^^^^^^^^^^^^^^^^^^^^^^^

Initialize a managed Python project:

.. code-block:: bash

   # Create a new package-based project
   uv init --package my_mjlab_project
   cd my_mjlab_project

3. Add mjlab dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^

There are different options to add ``mjlab`` as a dependency.
We recommend using the latest stable version from PyPI. If you need
the latest features, use the direct GitHub installation. Finally, if you
need to use a feature you have developed locally, use the local editable
install. These options are interchangeable: you can switch at any time.

.. tab-set::

   .. tab-item:: PyPI

      Once in your project, install the latest snapshot from PyPI:

      .. code:: bash

         uv add mjlab

   .. tab-item:: Source

      Once in your project, install directly from GitHub without cloning:

      .. code:: bash

         uv add "mjlab @ git+https://github.com/mujocolab/mjlab"

   .. tab-item:: Local

      Clone the repository:

      .. code:: bash

         git clone https://github.com/mujocolab/mjlab.git

      Once in your project, add it as an editable dependency:

      .. code:: bash

         uv add --editable /path/to/cloned/mjlab

.. tip::

   For a complete example of how to structure a project that integrates a custom robot
   with an existing ``mjlab`` task, check out the
   `ANYmal C Velocity Tracking <https://github.com/mujocolab/anymal_c_velocity>`_ repository.

Verification
^^^^^^^^^^^^

After installation, verify that ``mjlab`` is working by running the demo:

.. code-block:: bash

   uv run demo


.. _install-uv-develop:

Method 2 - Develop / contribute (uv)
------------------------------------

This method is for developing ``mjlab`` itself or contributing to the project.

.. code:: bash

   git clone https://github.com/mujocolab/mjlab.git
   cd mjlab
   uv sync

Verification
^^^^^^^^^^^^

After installation, verify that ``mjlab`` is working by running the demo:

.. code-block:: bash

   uv run demo


.. _install-pip:

Method 3 - Classic pip / venv / conda
-------------------------------------

While ``mjlab`` is designed to work with `uv <https://docs.astral.sh/uv/>`_, you can
also use it with any pip-based virtual environment (``venv``, ``conda``, ``virtualenv``, etc.).

Create and activate your virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: venv

      Using ``venv`` (standard library):

      .. code:: bash

         python -m venv mjlab-env
         source mjlab-env/bin/activate

   .. tab-item:: conda

      Using ``conda``:

      .. code:: bash

         conda create -n mjlab python=3.13
         conda activate mjlab


Install mjlab and dependencies via pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: PyPI

      From PyPI:

      .. code:: bash

         pip install mjlab

   .. tab-item:: Source

      From Source:

      .. code:: bash

         git clone https://github.com/mujocolab/mjlab.git
         cd mjlab
         pip install -e .


Verification
^^^^^^^^^^^^

After installation, verify that ``mjlab`` is working by running the demo:

.. code-block:: bash

   demo


.. _install-docker:

Method 4 - Docker / clusters
----------------------------

This method is recommended if you prefer running ``mjlab`` in containers (for example on
servers or clusters).


Prerequisites
^^^^^^^^^^^^^

- Install Docker: `Docker installation guide <https://docs.docker.com/engine/install/>`_.
- Install an appropriate NVIDIA driver for your system and the
  `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_.

  - Be sure to register the container runtime with Docker and restart, as described in
    the Docker configuration section of the NVIDIA install guide.


Build the Docker image
^^^^^^^^^^^^^^^^^^^^^^

From the root of the repository:

.. code-block:: bash

   make docker-build


Run mjlab in Docker
^^^^^^^^^^^^^^^^^^^

Use the included helper script to run an ``mjlab`` Docker container with useful arguments preconfigured:

.. code-block:: bash

   ./scripts/run_docker.sh

Examples:

- Demo with viewer:

  .. code-block:: bash

     ./scripts/run_docker.sh uv run demo

- Training example:

  .. code-block:: bash

     ./scripts/run_docker.sh uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096


Having some troubles?
---------------------

1. **Check the FAQ**

    Consult the mjlab :ref:`faq` for answers to common installation and runtime issues

2. **Still stuck?**

    Open an issue on GitHub: https://github.com/mujocolab/mjlab/issues
