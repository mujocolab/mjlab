"""Sources that list and fetch the checkpoints of a training run."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rsl_rl.utils import resolve_callable


@dataclass(frozen=True)
class CheckpointInfo:
  name: str
  updated_at: float
  """Last modification time as a POSIX timestamp."""

  @property
  def step(self) -> int:
    """Training iteration parsed from ``model_<step>.pt``, or 0 if absent."""
    match = re.match(r"^model_(\d+)\.pt$", self.name)
    return int(match.group(1)) if match else 0


class CheckpointSource(ABC):
  """Lists and fetches the checkpoints of a training run.

  Built as ``cls(run, cache_root)``, where ``cache_root`` is the experiment log
  directory.
  """

  label: str = "Remote"

  def __init__(self, run: str, cache_root: Path) -> None:
    self.run = run
    self.cache_root = cache_root
    self.run_name: str | None = None
    self.run_url: str | None = None
    self.run_status: str | None = None

  @abstractmethod
  def list_checkpoints(self) -> list[CheckpointInfo]:
    """List the ``.pt`` checkpoints of the run."""

  @abstractmethod
  def download(self, name: str) -> tuple[Path, bool]:
    """Return a local path to checkpoint ``name`` and whether it was cached."""

  def resolve(self, name: str | None = None) -> tuple[Path, bool]:
    """Fetch checkpoint ``name``, or the latest ``model_<step>.pt`` if None."""
    available = [
      c for c in self.list_checkpoints() if re.match(r"^model_\d+\.pt$", c.name)
    ]
    if not available:
      raise ValueError(f"No model_<step>.pt checkpoint found in run {self.run}.")
    if name is None:
      return self.download(max(available, key=lambda c: c.step).name)
    names = [c.name for c in available]
    if name not in names:
      raise ValueError(
        f"Checkpoint '{name}' not found in run {self.run}. Available: {names}"
      )
    return self.download(name)


class LocalCheckpointSource(CheckpointSource):
  """Checkpoints in a local run directory, given as ``run``."""

  label = "Local"

  def list_checkpoints(self) -> list[CheckpointInfo]:
    return [
      CheckpointInfo(f.name, f.stat().st_mtime) for f in Path(self.run).glob("*.pt")
    ]

  def download(self, name: str) -> tuple[Path, bool]:
    return Path(self.run) / name, True


class WandbCheckpointSource(CheckpointSource):
  """Checkpoints uploaded to a W&B run, given as ``entity/project/run_id``."""

  label = "W&B"

  def __init__(self, run: str, cache_root: Path) -> None:
    super().__init__(run, cache_root)
    import wandb

    self.wandb_run = wandb.Api().run(run)
    self.run_name = _parse_wandb_dt(self.wandb_run.created_at).strftime(
      "%Y-%m-%d_%H-%M-%S"
    )
    self.run_url = self.wandb_run.url
    self.run_status = self.wandb_run.state
    self._cache_dir = cache_root / "wandb_checkpoints" / run.split("/")[-1]

  def list_checkpoints(self) -> list[CheckpointInfo]:
    self.wandb_run.load()
    return [
      CheckpointInfo(f.name, _parse_wandb_dt(f.updated_at).timestamp())
      for f in self.wandb_run.files(pattern="%.pt")
      if f.name.endswith(".pt")
    ]

  def download(self, name: str) -> tuple[Path, bool]:
    path = self._cache_dir / name
    if path.exists():
      return path, True
    self._cache_dir.mkdir(parents=True, exist_ok=True)
    self.wandb_run.file(name).download(str(self._cache_dir), replace=True)
    return path, False


def make_checkpoint_source(source: str, run: str, cache_root: Path) -> CheckpointSource:
  """Build ``source``, either ``"wandb"`` or a ``CheckpointSource`` import path."""
  if source == "wandb":
    return WandbCheckpointSource(run, cache_root)
  return resolve_callable(source)(run, cache_root)


def _parse_wandb_dt(value: str | datetime) -> datetime:
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value
