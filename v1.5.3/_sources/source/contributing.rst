Contributing
============

Bug fixes and documentation improvements are always welcome.

.. important::

   For new features, please
   `open an issue <https://github.com/mujocolab/mjlab/issues>`_ first so
   we can discuss whether it fits the project scope.


Development setup
-----------------

Clone the repository and sync dependencies:

.. code-block:: bash

   git clone https://github.com/mujocolab/mjlab.git && cd mjlab
   uv sync

Install pre-commit hooks to catch formatting and lint issues before each
commit:

.. code-block:: bash

   uvx pre-commit install


Common commands
---------------

The ``Makefile`` provides shortcuts for the most common development tasks:

.. code-block:: bash

   make format      # Format code and fix lint errors (ruff)
   make type        # Type check (ty + pyright)
   make check       # Format + type check
   make test-fast   # Run tests, excluding slow ones
   make test        # Run the full test suite
   make test-all    # Format + type check + full test suite

You can also run individual tests for faster iteration:

.. code-block:: bash

   uv run pytest tests/test_rewards.py

Type checking (``make type``) is required. PRs that do not pass will be
blocked.


Building the docs
-----------------

Build the documentation locally:

.. code-block:: bash

   make docs

The HTML output is written to ``docs/_build/``. For live reload during
editing:

.. code-block:: bash

   make docs-watch


Submitting a pull request
-------------------------

1. Fork the repository and create a feature branch.
2. Make your changes.
3. Run ``make test-all`` to verify formatting, type checking, and tests
   pass.
4. Add an entry to the "Upcoming version" section in
   ``docs/source/changelog.rst`` under the appropriate category
   (Added / Changed / Fixed), following
   `Keep a Changelog <https://keepachangelog.com/>`_ conventions.
5. Submit a pull request.


Development with Claude Code
----------------------------

The repository includes a ``CLAUDE.md`` file at the project root. This file
defines development conventions, style guidelines, and common commands for
`Claude Code <https://claude.com/claude-code>`_. It is also a useful
reference for human contributors since it captures the same rules enforced
in CI.

The project also includes shared commands in ``.claude/commands/``.
Any contributor with Claude Code installed can invoke them as slash commands.

``/update-mjwarp <commit-hash>``
   Update the ``mujoco-warp`` dependency to a specific commit. This edits
   ``pyproject.toml``, runs ``uv lock``, and opens a PR in one step.

   .. code-block:: text

      /update-mjwarp e28c6038cdf8a353b4146974e4cf37e74dda809a

``/commit-push-pr``
   Stage current changes, commit, push, and open a PR.

