"""Tests for wandb utilities."""

import wandb

from mjlab.utils.wandb import add_wandb_tags


def test_add_wandb_tags_survives_prior_setup(monkeypatch, tmp_path):
  """Tags reach the run even when wandb.setup() ran before they were added."""
  monkeypatch.chdir(tmp_path)
  monkeypatch.setenv("WANDB_MODE", "offline")
  wandb.setup()
  add_wandb_tags(("nightly",))
  run = wandb.init(project="test")
  try:
    assert "nightly" in (run.tags or ())
  finally:
    run.finish()
