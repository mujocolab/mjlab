"""Publish checkpoints without exposing partially written files to readers."""

import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def atomic_output(path: str | Path) -> Iterator[Path]:
  """Write beside the destination, then atomically replace it on success.

  Temporary names do not match the default ``model_*.pt`` checkpoint selector.
  Windows readers can briefly deny replacement; permanent errors still propagate.
  Atomic replacement protects readers and the previous file, not against disk loss.
  """
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(
    prefix="checkpoint-", suffix=".tmp", dir=destination.parent
  )
  os.close(fd)
  temporary_path = Path(temporary)
  try:
    yield temporary_path
    with temporary_path.open("r+b") as stream:
      os.fsync(stream.fileno())
    for attempt in range(5):
      try:
        os.replace(temporary_path, destination)
        break
      except PermissionError:
        if attempt == 4:
          raise
        time.sleep(0.05)
  finally:
    temporary_path.unlink(missing_ok=True)


def save_checkpoint(state: dict, path: str | Path) -> None:
  """Serialize before making the complete checkpoint visible at ``path``."""
  with atomic_output(path) as temporary:
    torch.save(state, temporary)
