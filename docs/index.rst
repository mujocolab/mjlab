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
       --with "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@9fc294d86955a303619a254cefae809a41adb274" \
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
        version = {0.1.0},
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
   :maxdepth: 1
   :caption: Getting Started

   source/installation
   source/migration_isaac_lab

.. toctree::
   :maxdepth: 1
   :caption: About the Project

   source/motivation
   source/faq

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   source/api/index

.. toctree::
   :maxdepth: 1
   :caption: Core Concepts

   source/randomization
   source/nan_guard
   source/observation
   source/actuators
   source/sensors
   source/distributed_training
