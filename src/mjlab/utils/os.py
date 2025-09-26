from pathlib import Path
from typing import Any, Dict, Union

import yaml


def update_assets(
  assets: Dict[str, Any],
  path: Union[str, Path],
  meshdir: str | None = None,
  glob: str = "*",
  recursive: bool = False,
):
  """Update assets dictionary with files from a directory.

  This function reads files from a directory and adds them to an assets dictionary,
  with keys formatted to include the meshdir prefix when specified.

  Args:
    assets: Dictionary to update with file contents. Keys are asset paths, values are
      file contents as bytes.
    path: Path to directory containing asset files.
    meshdir: Optional mesh directory prefix, typically `spec.meshdir`. If provided,
      will be prepended to asset keys (e.g., "mesh.obj" becomes "custom_dir/mesh.obj").
    glob: Glob pattern for file matching. Defaults to "*" (all files).
    recursive: If True, recursively search subdirectories.
  """
  for f in Path(path).glob(glob):
    if f.is_file():
      asset_key = f"{meshdir}/{f.name}" if meshdir else f.name
      assets[asset_key] = f.read_bytes()
    elif f.is_dir() and recursive:
      update_assets(assets, f, meshdir, glob, recursive)


def dump_yaml(filename: Path, data: Dict, sort_keys: bool = False) -> None:
  """Saves data to a YAML file.

  Args:
      filename: The path to the YAML file.
      data: The data to save. Must be a dictionary.
      sort_keys: Whether to sort the keys in the YAML file.
  """
  if not filename.suffix:
    filename = filename.with_suffix(".yaml")
  filename.parent.mkdir(parents=True, exist_ok=True)
  with open(filename, "w") as f:
    yaml.dump(data, f, sort_keys=sort_keys)


def get_wandb_checkpoint_path(log_path: Path, run_path: Path) -> Path:
  import wandb

  api = wandb.Api()
  wandb_run = api.run(str(run_path))
  run_id = wandb_run.id  # Get the unique run ID

  files = [file.name for file in wandb_run.files() if "model" in file.name]
  checkpoint_file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

  # Use run-specific directory.
  download_dir = log_path / "wandb_checkpoints" / run_id
  checkpoint_path = download_dir / checkpoint_file

  # If it exists, don't download it again.
  if checkpoint_path.exists():
    print(f"[INFO]: Using cached checkpoint {checkpoint_file} for run {run_id}")
    return checkpoint_path

  wandb_file = wandb_run.file(str(checkpoint_file))
  wandb_file.download(str(download_dir), replace=True)
  return checkpoint_path
