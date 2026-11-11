"""Tests for checkpoint sources."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import wandb

from mjlab.utils.checkpoint_source import (
  CheckpointInfo,
  CheckpointSource,
  LocalCheckpointSource,
  WandbCheckpointSource,
  make_checkpoint_source,
)


class _StaticCheckpointSource(CheckpointSource):
  """Custom source resolved by import path in the tests below."""

  def list_checkpoints(self) -> list[CheckpointInfo]:
    return [CheckpointInfo("model_10.pt", 0.0), CheckpointInfo("model_5.pt", 0.0)]

  def download(self, name: str) -> tuple[Path, bool]:
    return self.cache_root / name, False


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
  run_dir = tmp_path / "2025-01-01_00-00-00"
  run_dir.mkdir()
  for name in ("model_50.pt", "model_100.pt", "model_9.pt", "other.pt"):
    (run_dir / name).touch()
  return run_dir


def test_local_source_resolves_latest_and_named(run_dir):
  source = LocalCheckpointSource(str(run_dir), run_dir.parent)
  assert source.resolve() == (run_dir / "model_100.pt", True)
  assert source.resolve("model_9.pt") == (run_dir / "model_9.pt", True)
  with pytest.raises(ValueError, match="not found"):
    source.resolve("model_7.pt")


def test_make_checkpoint_source_from_import_path(tmp_path):
  source = make_checkpoint_source(
    f"{__name__}:_StaticCheckpointSource", "run-id", tmp_path
  )
  assert isinstance(source, _StaticCheckpointSource)
  assert source.resolve() == (tmp_path / "model_10.pt", False)


def test_wandb_source_downloads_latest_once(monkeypatch, tmp_path):
  files = [MagicMock(updated_at="2025-01-01T00:00:00Z") for _ in range(2)]
  files[0].name, files[1].name = "model_100.pt", "model_2000.pt"
  run = MagicMock(created_at="2025-01-01T00:00:00Z")
  run.files.return_value = files
  run.file.return_value.download.side_effect = lambda root, replace: (
    Path(root) / "model_2000.pt"
  ).touch()
  monkeypatch.setattr(wandb, "Api", lambda: MagicMock(run=lambda path: run))

  source = WandbCheckpointSource("entity/project/abc123", tmp_path)
  path = tmp_path / "wandb_checkpoints" / "abc123" / "model_2000.pt"
  assert source.resolve() == (path, False)
  assert source.resolve() == (path, True)
  run.file.assert_called_once_with("model_2000.pt")
