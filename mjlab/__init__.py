"""mjlab."""

import subprocess
import sys
from typing import Any, Dict, Union

import tqdm
from pathlib import Path

MJLAB_SRC_PATH = Path(__file__).parent
MJLAB_ROOT_PATH = MJLAB_SRC_PATH.parent
EXTERNAL_DEPS_PATH = MJLAB_SRC_PATH / "external_deps"
MENAGERIE_PATH = EXTERNAL_DEPS_PATH / "mujoco_menagerie"
MENAGERIE_COMMIT_SHA = "14ceccf557cc47240202f2354d684eca58ff8de4"

MOTION_DATA_DIR = MJLAB_SRC_PATH / "envs" / "motion_imitation" / "data"
LAFAN1_DATA_DIR = MOTION_DATA_DIR / "lafan1"
PROCESSED_DATA_DIR = MOTION_DATA_DIR / "processed"


def update_assets(
  assets: Dict[str, Any],
  path: Union[str, Path],
  glob: str = "*",
  recursive: bool = False,
):
  """Update the assets dictionary with the contents of the given path.

  Args:
    assets: The dictionary to update.
    path: The path to the directory to update.
    glob: The glob pattern to use to find files.
    recursive: Whether to recursively update the assets dictionary.
  """
  for f in Path(path).glob(glob):
    if f.is_file():
      assets[f.name] = f.read_bytes()
    elif f.is_dir() and recursive:
      update_assets(assets, f, glob, recursive)


def _clone_with_progress(repo_url: str, target_path: str, commit_sha: str) -> None:
  """Clone a git repo with progress bar."""
  process = subprocess.Popen(
    ["git", "clone", "--progress", repo_url, target_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
  )

  with tqdm.tqdm(
    desc="Cloning mujoco_menagerie",
    bar_format="{desc}: {bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
  ) as pbar:
    pbar.total = 100  # Set to 100 for percentage-based progress.
    current = 0
    while True:
      # Read output line by line.
      output = process.stderr.readline()  # type: ignore
      if not output and process.poll() is not None:
        break
      if output:
        if "Receiving objects:" in output:
          try:
            percent = int(output.split("%")[0].split(":")[-1].strip())
            if percent > current:
              pbar.update(percent - current)
              current = percent
          except (ValueError, IndexError):
            pass

    # Ensure the progress bar reaches 100%.
    if current < 100:
      pbar.update(100 - current)

  if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, ["git", "clone"])

  # Checkout specific commit.
  print(f"Checking out commit {commit_sha}")
  subprocess.run(
    ["git", "-C", target_path, "checkout", commit_sha],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )


def _ensure_menagerie_exists() -> None:
  """Ensure mujoco_menagerie exists, downloading it if necessary."""
  if not MENAGERIE_PATH.exists():
    print("mujoco_menagerie not found. Downloading...")

    # Create external deps directory if it doesn't exist
    EXTERNAL_DEPS_PATH.mkdir(exist_ok=True, parents=True)

    try:
      _clone_with_progress(
        "https://github.com/deepmind/mujoco_menagerie.git",
        str(MENAGERIE_PATH),
        MENAGERIE_COMMIT_SHA,
      )
      print("Successfully downloaded mujoco_menagerie")
    except subprocess.CalledProcessError as e:
      print(f"Error downloading mujoco_menagerie: {e}", file=sys.stderr)
      raise


_ensure_menagerie_exists()


__all__ = (
  "MJLAB_SRC_PATH",
  "MJLAB_ROOT_PATH",
  "EXTERNAL_DEPS_PATH",
  "MENAGERIE_PATH",
)
