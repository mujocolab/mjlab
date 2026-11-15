.. _export-scene:

Export Scene
============

The ``export-scene`` script writes a complete scene (XML and mesh assets) to a
directory for inspection, sharing, or loading in standalone MuJoCo.

Quick start
-----------

.. code-block:: bash

    # Export a built-in entity by alias.
    uv run export-scene g1 --output-dir /tmp/g1

    # Export a registered task scene.
    uv run export-scene Mjlab-Velocity-Flat-Unitree-Go1 --output-dir /tmp/task

    # Export as a zip archive.
    uv run export-scene yam --output-dir /tmp/yam --zip True

    # Export a custom entity via import path.
    uv run export-scene my_pkg.robots:get_my_robot_cfg --output-dir /tmp/custom

The output directory contains a ``scene.xml`` and an ``assets/`` subdirectory
with all referenced mesh files. The XML can be loaded directly with
``mujoco.MjModel.from_xml_path()`` or dropped into the
`simulate viewer <https://mujoco.readthedocs.io/en/stable/programming/samples.html#sasimulate>`_.

Target resolution
-----------------

The positional ``target`` argument is resolved in order:

1. **Task ID**: checked against the task registry (``import mjlab.tasks``).
2. **Entity alias**: one of the built-in shorthands (``g1``, ``go1``, ``yam``).
3. **Import path**: a ``module:attribute`` string pointing to any callable
   that returns an ``EntityCfg``.

If none match, the script prints available task IDs and aliases.

Options
-------

``--output-dir DIR`` *(default: "export")*
    Destination directory. Cleaned before each export to prevent stale assets.

``--zip True`` *(default: False)*
    Compress the output into a ``.zip`` archive and remove the directory.
