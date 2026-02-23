---
allowed-tools: Bash(uv lock), Bash(git checkout:*), Bash(git add:*), Bash(git status:*), Bash(git push:*), Bash(git commit:*), Bash(gh pr create:*), Edit, Read
description: Update the mujoco-warp dependency to a given commit
---

Update the mujoco-warp dependency to commit $ARGUMENTS.

Steps:
1. Read `pyproject.toml` and find the `mujoco-warp` line under `[tool.uv.sources]`.
2. Use Edit to replace the current `rev = "..."` value with `rev = "$ARGUMENTS"` on that line.
3. Run `uv lock` to regenerate the lockfile.
4. Create and switch to a new branch named `update-mjwarp/<first-8-chars-of-hash>` (e.g. `update-mjwarp/e28c6038`).
5. Stage `pyproject.toml` and `uv.lock`, then commit with message: `Update mujoco-warp to <first-8-chars-of-hash>`.
6. Push the branch and open a PR with title `Update mujoco-warp to <first-8-chars-of-hash>`.

Important:
- The commit hash is required. If `$ARGUMENTS` is empty, ask the user for a commit hash.
- Do NOT modify anything else in `pyproject.toml`.
