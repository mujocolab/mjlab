Contributing
============

See `CONTRIBUTING.md <https://github.com/mujocolab/mjlab/blob/main/CONTRIBUTING.md>`_
for the general contribution workflow, including forking, testing, and changelog
conventions.

CLAUDE.md
---------

The repository includes a ``CLAUDE.md`` file at the project root. This file
defines development conventions, style guidelines, and common commands for
`Claude Code <https://claude.com/claude-code>`_. It is also a useful reference
for human contributors since it captures the same rules enforced in CI.

Claude Code Commands
--------------------

The project includes shared Claude Code commands in ``.claude/commands/``.
Any contributor with Claude Code installed can invoke them as slash commands.

``/update-mjwarp <commit-hash>``
   Update the ``mujoco-warp`` dependency to a specific commit. This edits
   ``pyproject.toml``, runs ``uv lock``, and opens a PR in one step.

   .. code-block:: text

      /update-mjwarp e28c6038cdf8a353b4146974e4cf37e74dda809a

``/commit-push-pr``
   Stage current changes, commit, push, and open a PR.
