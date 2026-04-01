import re
from pathlib import Path
from typing import Dict

import yaml


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
  if not log_path.exists():
    raise ValueError(f"Log path does not exist: {log_path}")
  # Exclude wandb_checkpoints directory which is used for caching downloaded checkpoints.
  runs = [
    log_path / run.name
    for run in log_path.iterdir()
    if run.is_dir() and run.name != "wandb_checkpoints" and re.match(run_dir, run.name)
  ]
  if len(runs) == 0:
    raise ValueError(f"No run directories found in {log_path} matching '{run_dir}'")
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


def get_wandb_checkpoint_path(
  log_path: Path, run_path: Path, checkpoint_name: str | None = None
) -> tuple[Path, bool]:
  """Get checkpoint path from wandb, downloading if needed.

  Returns:
    Tuple of (checkpoint_path, was_cached)
  """
  import wandb

  # Extract run_id from path (e.g., "entity/project/run_id" -> "run_id").
  run_id = str(run_path).split("/")[-1]
  download_dir = log_path / "wandb_checkpoints" / run_id

  # Query wandb API to find the latest checkpoint.
  api = wandb.Api()
  wandb_run = api.run(str(run_path))
  files = [
    file.name for file in wandb_run.files() if re.match(r"^model_\d+\.pt$", file.name)
  ]
  if checkpoint_name is None:
    checkpoint_file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
  else:
    if checkpoint_name not in files:
      raise ValueError(
        f"Checkpoint '{checkpoint_name}' not found in run {run_path}."
        f" Available: {files}"
      )
    checkpoint_file = checkpoint_name

  checkpoint_path = download_dir / checkpoint_file

  # If this checkpoint is not cached locally, download it.
  was_cached = checkpoint_path.exists()
  if not was_cached:
    download_dir.mkdir(parents=True, exist_ok=True)
    wandb_file = wandb_run.file(str(checkpoint_file))
    wandb_file.download(str(download_dir), replace=True)

  return checkpoint_path, was_cached
