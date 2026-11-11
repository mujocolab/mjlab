from pathlib import Path

from mjlab.utils.os import get_checkpoint_path


def test_latest_checkpoint_selection(tmp_path: Path):
  for path in (
    "2026-11-09_12-00-00/model_5000.pt",
    "2026-11-10_12-00-00/model_900.pt",
    "2026-11-10_12-00-00/model_1000.pt",
    "wandb_checkpoints/model_9000.pt",
  ):
    (tmp_path / path).parent.mkdir(exist_ok=True)
    (tmp_path / path).touch()

  path = get_checkpoint_path(tmp_path, ".*", "model_.*.pt")
  assert path == tmp_path / "2026-11-10_12-00-00" / "model_1000.pt"
