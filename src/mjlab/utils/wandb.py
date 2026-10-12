"""WandB utilities."""

from __future__ import annotations

from typing import Sequence


def add_wandb_tags(tags: Sequence[str]) -> None:
  """Add tags to the current wandb run.

  If the run is not initialized yet (rsl_rl creates it lazily in ``learn()``), the
  tags are added to wandb's global settings, which ``wandb.init`` copies. Setting the
  ``WANDB_TAGS`` env var instead is silently ignored once ``wandb.setup()`` has run.
  """
  if not tags:
    return

  try:
    import wandb
  except ImportError:
    return

  if wandb.run is not None:
    wandb.run.tags = tuple(set(wandb.run.tags or ()) | set(tags))
  else:
    settings = wandb.setup().settings
    settings.run_tags = tuple(sorted(set(settings.run_tags or ()) | set(tags)))
