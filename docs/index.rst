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
   uvx --from mjlab demo

If this runs, your setup is compatible with mjlab *for evaluation*.

License & citation
==================

mjlab is licensed under the Apache License, Version 2.0.
Please refer to the `LICENSE file <https://github.com/mujocolab/mjlab/blob/main/LICENSE/>`_ for details.

If you use mjlab in your research, we would appreciate a citation:

.. code-block:: bibtex

    @article{Zakka_mjlab_A_Lightweight_2026,
        author = {Zakka, Kevin and Liao, Qiayuan and Yi, Brent and Le Lay, Louis and Sreenath, Koushil and Abbeel, Pieter},
        title = {{mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning}},
        url = {https://arxiv.org/abs/2601.22074},
        year = {2026}
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
   source/changelog

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
   source/raycast_sensor
   source/distributed_training
