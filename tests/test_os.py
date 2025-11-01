"""Tests for mjlab.utils.os."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from mjlab.utils.os import get_wandb_checkpoint_path


def test_get_wandb_checkpoint_path_cached():
  """Cached checkpoints return without wandb API calls."""
  with tempfile.TemporaryDirectory() as tmpdir:
    log_path = Path(tmpdir)
    run_path = Path("entity/project/test_run_123")

    download_dir = log_path / "wandb_checkpoints" / "test_run_123"
    download_dir.mkdir(parents=True)

    checkpoint_file = "model_5000.pt"
    checkpoint_path = download_dir / checkpoint_file
    checkpoint_path.write_text("fake checkpoint")

    metadata_file = download_dir / ".checkpoint_metadata.json"
    with open(metadata_file, "w") as f:
      json.dump({"checkpoint_file": checkpoint_file}, f)

    with patch("wandb.Api") as mock_api:
      result_path, was_cached = get_wandb_checkpoint_path(log_path, run_path)

      mock_api.assert_not_called()
      assert result_path == checkpoint_path
      assert was_cached is True
