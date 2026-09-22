"""Helpers for translating mjlab RL configs into rsl-rl constructor kwargs."""

from typing import Any


def clean_model_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
  """Return a copy of a model config dict with unset optional keys removed.

  rsl-rl model constructors do not accept ``None`` for these options, so they
  must be dropped entirely when unset.
  """
  cfg = dict(cfg)
  for opt in ("cnn_cfg", "distribution_cfg"):
    if cfg.get(opt, "unset") is None:
      cfg.pop(opt)
  if cfg.get("rnn_type", "unset") is None:
    for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      cfg.pop(opt, None)
  return cfg
