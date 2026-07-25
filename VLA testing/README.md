# VLA testing

This folder contains isolated experiments for connecting open-source robot
environments and VLA policies to mjlab.

## Current work

- Added a source snapshot of Meta-World under `environments/metaworld`.
- Pinned the snapshot to commit
  `f571cd00d85af4dc1264a31ed85407ec23495d89`.
- Added a continuously rendered launcher for Meta-World's `pick-place-v3` task.
- Added a custom mjlab scene with the Go2+D1 robot, a low work table, three
  movable objects, a target tray, and navigation obstacles.
- Added robot-mounted ego and wrist cameras. Both produce 256x256 RGB and depth;
  SmolVLA will consume RGB, while depth is retained for expert demonstrations,
  debugging, and future perception work.
- The next step is to inspect and port the task scene into mjlab, replacing the
  original arm with the existing Go2+D1 robot.

Meta-World is maintained by the Farama Foundation and its downloaded source
retains the upstream MIT license.

## Start

From the mjlab repository root, run:

```powershell
uv run --no-project --with-editable ".\VLA testing\environments\metaworld" python ".\VLA testing\start_metaworld.py"
```

Meta-World does not officially support Windows. If the viewer fails on Windows,
run the same command in Linux or WSL with GPU/GUI support.

To start the custom Go2+D1 scene from the repository root:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"
uv run python ".\VLA testing\start_custom_scene.py"
```

This scene currently validates MuJoCo-Warp composition and physics. The robot is
held near its default standing pose; arm control and grasping come next. The
first run creates a separate Windows environment and can take several minutes
while the GPU dependencies are installed.
