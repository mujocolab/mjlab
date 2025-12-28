Welcome to mjlab!
=================

.. figure:: source/_static/mjlab-banner.jpg
   :width: 100%
   :alt: mjlab

What is mjlab?
==============

**mjlab = Isaac Lab's API + MuJoCo's simplicity + GPU acceleration**

We took Isaac Lab's proven manager-based architecture and RL abstractions,
then built them directly on MuJoCo Warp. No translation layers, no Omniverse
overhead. Just fast, transparent physics.

You can try mjlab *without installing anything* by using `uvx`:

.. code-block:: bash

   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Run the mjlab demo (no local installation needed)
   uvx --from mjlab \
       --with "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@f2f795796fc433adf8e235f01fae3747585ae5db" \
       demo

If this runs, your setup is compatible with mjlab *for evaluation*.

License & citation
==================

mjlab is licensed under the Apache License, Version 2.0.
Please refer to the `LICENSE file <https://github.com/mujocolab/mjlab/blob/main/LICENSE/>`_ for details.

If you use mjlab in your research, we would appreciate a citation:

.. code-block:: bibtex

    @software{Zakka_Mjlab_Isaac_Lab_2025,
        author = {Zakka, Kevin and Yi, Brent and Liao, Qiayuan and Le Lay, Louis},
        license = {Apache-2.0},
        month = sep,
        title = {{mjlab: Isaac Lab API, powered by MuJoCo-Warp, for RL and robotics research.}},
        url = {https://github.com/mujocolab/mjlab},
        version = {1.0.0},
        year = {2025}
    }

Acknowledgments
===============

mjlab would not exist without the excellent work of the Isaac Lab team, whose API design
and abstractions mjlab builds upon.

Thanks also to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features based
on our requests countless times.

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :titlesonly:

   source/getting_started/installation
   source/getting_started/motivation
   source/getting_started/walkthrough/index

.. toctree::
   :maxdepth: 2
   :caption: Architecture
   :titlesonly:

   source/architecture/manager_based_env
   source/architecture/scene
   source/architecture/control_flow

.. toctree::
   :maxdepth: 2
   :caption: Components
   :titlesonly:

   source/components/entities
   source/components/actuators
   source/components/sensors
   source/components/terrains

.. toctree::
   :maxdepth: 2
   :caption: Environment Guide
   :titlesonly:

   source/environment_guide/observations
   source/environment_guide/domain_randomization

.. toctree::
   :maxdepth: 2
   :caption: Features
   :titlesonly:

   source/features/configuration
   source/features/distributed_training
   source/features/nan_guard

.. toctree::
   :maxdepth: 1
   :caption: API
   :titlesonly:

   source/api/index

.. toctree::
   :maxdepth: 1
   :caption: References
   :titlesonly:

   source/references/changelog
   source/references/contributing
   source/references/faq
   source/references/migration_isaac_lab
