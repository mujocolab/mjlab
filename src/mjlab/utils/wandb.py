"""WandB utilities."""

from __future__ import annotations

import os
from typing import Sequence


def add_wandb_tags(tags: Sequence[str]) -> None:
  """Add tags to the current wandb run.

  Note: This function stores tags in wandb.config._wandb_tags if the run is not yet
  initialized, allowing them to be retrieved later. If the run is already initialized,
  tags are added directly.
  """
  if not tags:
    return

  try:
    import wandb

    if wandb.run is not None:
      existing_tags = list(wandb.run.tags) if wandb.run.tags else []
      new_tags = list(set(existing_tags + list(tags)))
      wandb.run.tags = new_tags
    else:
      # Store tags to be added when run is initialized.
      # This is a workaround for lazy wandb initialization in rsl_rl 3.1.0.
      current_tags = os.environ.get("WANDB_TAGS", "")
      all_tags = set(current_tags.split(",") if current_tags else [])
      all_tags.update(tags)
      os.environ["WANDB_TAGS"] = ",".join(sorted(all_tags))
  except ImportError:
    pass
