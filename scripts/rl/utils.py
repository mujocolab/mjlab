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
  files = [file.name for file in wandb_run.files() if "model" in file.name]
  checkpoint_file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
  # If it exists, don't download it again.
  if (log_path / "temp" / checkpoint_file).exists():
    print(f"[INFO]: Checkpoint {checkpoint_file} already exists in {log_path / 'temp'}")
    return log_path / "temp" / checkpoint_file
  wandb_file = wandb_run.file(str(checkpoint_file))
  wandb_file.download(f"{log_path}/temp", replace=True)
  return log_path / "temp" / checkpoint_file
