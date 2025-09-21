import re
import shutil
import subprocess
from pathlib import Path


def sh(cmd: list[str], cwd: Path | None = None) -> None:
  """Run a shell command."""
  subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def copy_dir(src: Path, dst: Path, *, overwrite: bool = True) -> None:
  """Copy entire directory src to dst."""
  if not src.is_dir():
    raise ValueError(f"Source is not a directory: {src}")
  if dst.exists():
    if not overwrite:
      raise FileExistsError(f"Destination already exists: {dst}")
    shutil.rmtree(dst)
  shutil.copytree(src, dst)


def resolve_name(dist_name: str) -> str:
  """Make a safe name from a given string."""
  name = re.sub(r"[^a-z0-9_]", "_", dist_name.lower())
  return f"_{name}" if name and name[0].isdigit() else name


def replace_line(path: Path, line_no: int, new: str | None) -> None:
  """Replace line in file."""
  lines = path.read_text(encoding="utf-8").splitlines()
  if not (1 <= line_no <= len(lines)):
    raise IndexError(f"Line {line_no} out of range (file has {len(lines)} lines)")
  if new is None:
    del lines[line_no - 1]
  else:
    lines[line_no - 1] = new
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_or_copy(
  path: Path, *, content: str | None = None, src: Path | None = None, force: bool = True
) -> None:
  """Write `content` to `path` or copy from `src`."""
  path.parent.mkdir(parents=True, exist_ok=True)
  if path.exists() and not force:
    print(f"[skip] {path} exists")
    return
  if src is not None:
    shutil.copy(src, path)
  else:
    path.write_text(content or "", encoding="utf-8")


def main():
  project_name = resolve_name(input("Name of project: "))
  print(f"[INFO] Creating {project_name}")

  # TODO add the name in majuscule
  # TODO propose manager based env/direct (when available)
  # TODO propose path of project

  # Roots
  mjlab_root = Path(__file__).resolve().parents[2]
  mjlab_content = mjlab_root / "src" / "mjlab"
  templates = mjlab_root / "scripts" / "generator" / "templates"
  core = templates / "core"

  project_root = mjlab_root.parent / project_name
  project_content = project_root / "src" / project_name

  # Init project + add local mjlab
  sh(["uv", "init", "--package", project_name], cwd=mjlab_root.parent)
  sh(["uv", "add", "../mjlab"], cwd=project_root)

  # Core files
  write_or_copy(project_root / "README.md", content=f"###{project_name}")
  write_or_copy(project_content / "__init__.py", src=core / "__init__.py")

  # Robots
  write_or_copy(
    project_content / "robots" / "__init__.py", src=templates / "robots" / "__init__.py"
  )
  copy_dir(
    mjlab_content / "asset_zoo" / "robots" / "unitree_go1",
    project_content / "robots" / "unitree_go1",
  )

  # Tasks
  write_or_copy(
    project_content / "tasks" / "__init__.py",
    src=mjlab_content / "tasks" / "__init__.py",
  )
  copy_dir(
    mjlab_content / "tasks" / "locomotion" / "velocity" / "config" / "go1",
    project_content / "tasks" / "go1_locomotion",
  )

  # Scripts
  write_or_copy(
    project_root / "scripts" / "list_envs.py",
    src=mjlab_root / "scripts" / "list_envs.py",
  )
  copy_dir(mjlab_root / "scripts" / "velocity" / "rl", project_root / "scripts" / "rl")

  # modifying scriptsss
  replace_line(
    project_root / "scripts" / "list_envs.py",
    6,
    f"import {project_name}.tasks  # noqa: F401 to register environments",
  )
  replace_line(
    project_root / "scripts" / "rl" / "train.py",
    12,
    f"import {project_name}.tasks  # noqa: F401",
  )

  # modifying core
  replace_line(
    project_content / "__init__.py",
    3,
    f"{project_name}_SRC_PATH: Path = Path(__file__).parent",
  )

  # modifying robots
  replace_line(
    project_content / "robots" / "unitree_go1" / "go1_constants.py",
    7,
    f"from {project_name} import {project_name}_SRC_PATH",
  )
  replace_line(
    project_content / "robots" / "unitree_go1" / "go1_constants.py",
    18,
    f'  {project_name}_SRC_PATH / "robots" / "unitree_go1" / "xmls" / "go1.xml"',
  )

  # modifying tasks
  replace_line(
    project_content / "tasks" / "go1_locomotion" / "rough_env_cfg.py",
    3,
    f"from {project_name}.robots.unitree_go1.go1_constants import (",
  )
  replace_line(
    project_content / "tasks" / "go1_locomotion" / "flat_env_cfg.py",
    3,
    f"from {project_name}.tasks.go1_locomotion.rough_env_cfg import (",
  )

  # replaces Mjlab by template in task id to not get override by mjlab task
  replace_line(
    project_root / "scripts" / "list_envs.py",
    11,
    f'  prefix_substring = "{project_name}-"',
  )
  replace_line(
    project_root / "scripts" / "rl" / "train.py",
    98,
    f'  task_prefix = "{project_name}-Velocity-"',
  )
  replace_line(
    project_content / "tasks" / "go1_locomotion" / "__init__.py",
    4,
    f'  id="{project_name}-Velocity-Rough-Unitree-Go1",',
  )
  replace_line(
    project_content / "tasks" / "go1_locomotion" / "__init__.py",
    14,
    f'  id="{project_name}-Velocity-Rough-Unitree-Go1-Play",',
  )
  replace_line(
    project_content / "tasks" / "go1_locomotion" / "__init__.py",
    24,
    f'  id="{project_name}-Velocity-Flat-Unitree-Go1",',
  )
  replace_line(
    project_content / "tasks" / "go1_locomotion" / "__init__.py",
    34,
    f'  id="{project_name}-Velocity-Flat-Unitree-Go1-Play",',
  )


if __name__ == "__main__":
  main()
