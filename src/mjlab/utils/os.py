import re
from dataclasses import asdict, is_dataclass
from inspect import isclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
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


def _to_yaml_friendly(x: Any) -> Any:
  # Dataclass instances → dict.
  if is_dataclass(x) and not isclass(x):
    x = asdict(x)

  # Mappings.
  if isinstance(x, dict):
    return {k: _to_yaml_friendly(v) for k, v in x.items()}

  # Sequences (normalize tuples/sets to lists).
  if isinstance(x, (list, tuple, set)):
    return [_to_yaml_friendly(v) for v in x]

  # NumPy arrays & scalars.
  if isinstance(x, np.ndarray):
    return x.tolist()
  if isinstance(x, (np.floating, np.integer, np.bool_)):
    return x.item()

  # Torch tensors (handle scalars cleanly).
  if torch.is_tensor(x):
    x = x.detach().cpu()
    return x.item() if x.dim() == 0 else x.tolist()

  # Bytes → str (fallback to hex).
  if isinstance(x, (bytes, bytearray)):
    try:
      return x.decode("utf-8")
    except Exception:
      return x.hex()

  # For anything else, try to use it as-is if it's a basic type,
  # otherwise convert to string representation.
  if isinstance(x, (str, int, float, bool, type(None))):
    return x

  # Fallback: convert to string for anything we can't serialize
  # (enums, slices, functions, custom objects, etc).
  return str(x)


def dump_yaml(filename: Path, data: Dict, sort_keys: bool = False) -> None:
  """Write a human-readable YAML file.

  Args:
    filename: Destination path (".yaml" added if missing).
    data: Mapping to serialize.
    sort_keys: Whether to sort mapping keys in the output.
  """
  if not filename.suffix:
    filename = filename.with_suffix(".yaml")
  filename.parent.mkdir(parents=True, exist_ok=True)
  safe = _to_yaml_friendly(data)
  with open(filename, "w") as f:
    yaml.safe_dump(
      safe, f, sort_keys=sort_keys, allow_unicode=True, default_flow_style=False
    )


def get_checkpoint_path(
  log_path: Path,
  run_dir: str = ".*",
  checkpoint: str = ".*",
  sort_alpha: bool = True,
) -> Path:
  """Get path to model checkpoint in input directory.

  The checkpoint file is resolved as: `<log_path>/<run_dir>/<checkpoint>`.

  If `run_dir` and `checkpoint` are regex expressions, then the most recent
  (highest alphabetical order) run and checkpoint are selected. To disable this
  behavior, set `sort_alpha` to `False`.
  """
  runs = [
    log_path / run.name
    for run in log_path.iterdir()
    if run.is_dir() and re.match(run_dir, run.name)
  ]
  if sort_alpha:
    runs.sort()
  else:
    runs = sorted(runs, key=lambda p: p.stat().st_mtime)
  run_path = runs[-1]

  model_checkpoints = [
    f.name for f in run_path.iterdir() if re.match(checkpoint, f.name)
  ]
  if len(model_checkpoints) == 0:
    raise ValueError(f"No checkpoint found in {run_path} matching {checkpoint}")
  model_checkpoints.sort(key=lambda m: f"{m:0>15}")
  checkpoint_file = model_checkpoints[-1]
  return run_path / checkpoint_file


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
