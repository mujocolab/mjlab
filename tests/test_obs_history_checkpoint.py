"""Tests for observation history round-trip through checkpoints.

Covers:
  - MjlabOnPolicyRunner.save() serialises obs history config into the checkpoint.
  - _apply_obs_history_from_checkpoint() restores it into env_cfg on play.
"""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import torch

from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.scripts.play import _apply_obs_history_from_checkpoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(obs_history_cfg: dict | None, tmp_path) -> str:
  """Write a minimal checkpoint dict to a temp file and return its path."""
  env_state: dict[str, Any] = {"common_step_counter": 0}
  if obs_history_cfg is not None:
    env_state["obs_history_cfg"] = obs_history_cfg
  ckpt = {
    "iter": 0,
    "infos": {"env_state": env_state},
    # Minimal placeholders so torch.load succeeds.
    "actor_state_dict": {},
    "critic_state_dict": {},
  }
  path = str(tmp_path / "model_1000.pt")
  torch.save(ckpt, path)
  return path


def _make_env_cfg(group_cfgs: dict[str, ObservationGroupCfg]):
  """Return a minimal mock env_cfg whose .observations maps to group_cfgs."""

  @dataclass
  class _EnvCfg:
    observations: dict = field(default_factory=dict)

  cfg = _EnvCfg(observations=group_cfgs)
  return cfg


def _simple_obs_func(env):
  return torch.zeros((env.num_envs, 3), device="cpu")


# ---------------------------------------------------------------------------
# _apply_obs_history_from_checkpoint — happy-path tests
# ---------------------------------------------------------------------------


def test_apply_group_level_history(tmp_path):
  """Group-level history_length is applied to the ObservationGroupCfg."""
  obs_history_cfg = {
    "actor": {
      "history_length": 5,
      "flatten_history_dim": False,
      "terms": {
        "obs1": {"history_length": 5, "flatten_history_dim": False},
      },
    }
  }
  path = _make_checkpoint(obs_history_cfg, tmp_path)

  env_cfg = _make_env_cfg(
    {
      "actor": ObservationGroupCfg(
        terms={"obs1": ObservationTermCfg(func=_simple_obs_func, params={})},
      )
    }
  )

  # Verify no history before the call.
  assert env_cfg.observations["actor"].history_length is None
  assert env_cfg.observations["actor"].terms["obs1"].history_length == 0

  _apply_obs_history_from_checkpoint(env_cfg, path)

  assert env_cfg.observations["actor"].history_length == 5
  assert env_cfg.observations["actor"].flatten_history_dim is False


def test_apply_group_level_history_flatten_true(tmp_path):
  """flatten_history_dim=True is correctly restored."""
  obs_history_cfg = {
    "actor": {
      "history_length": 3,
      "flatten_history_dim": True,
      "terms": {},
    }
  }
  path = _make_checkpoint(obs_history_cfg, tmp_path)

  env_cfg = _make_env_cfg(
    {
      "actor": ObservationGroupCfg(
        terms={"obs1": ObservationTermCfg(func=_simple_obs_func, params={})},
      )
    }
  )

  # Verify no history before the call.
  assert env_cfg.observations["actor"].history_length is None

  _apply_obs_history_from_checkpoint(env_cfg, path)

  assert env_cfg.observations["actor"].history_length == 3
  assert env_cfg.observations["actor"].flatten_history_dim is True


def test_apply_per_term_history_when_no_group_override(tmp_path):
  """Per-term history is applied when the group has no group-level override."""
  obs_history_cfg = {
    "actor": {
      "history_length": None,  # No group-level override.
      "flatten_history_dim": True,
      "terms": {
        "obs1": {"history_length": 4, "flatten_history_dim": True},
        "obs2": {"history_length": 0, "flatten_history_dim": True},
      },
    }
  }
  path = _make_checkpoint(obs_history_cfg, tmp_path)

  env_cfg = _make_env_cfg(
    {
      "actor": ObservationGroupCfg(
        terms={
          "obs1": ObservationTermCfg(func=_simple_obs_func, params={}),
          "obs2": ObservationTermCfg(func=_simple_obs_func, params={}),
        },
      )
    }
  )

  # Verify no history before the call.
  assert env_cfg.observations["actor"].history_length is None
  assert env_cfg.observations["actor"].terms["obs1"].history_length == 0
  assert env_cfg.observations["actor"].terms["obs2"].history_length == 0

  _apply_obs_history_from_checkpoint(env_cfg, path)

  # Group-level should remain None (not overridden).
  assert env_cfg.observations["actor"].history_length is None
  # obs1 should have its term-level history restored.
  assert env_cfg.observations["actor"].terms["obs1"].history_length == 4
  assert env_cfg.observations["actor"].terms["obs1"].flatten_history_dim is True
  # obs2 has history_length=0 so it should not have been touched.
  assert env_cfg.observations["actor"].terms["obs2"].history_length == 0


def test_apply_multiple_groups(tmp_path):
  """History is applied independently to each observation group."""
  obs_history_cfg = {
    "actor": {
      "history_length": 5,
      "flatten_history_dim": True,
      "terms": {},
    },
    "critic": {
      "history_length": 3,
      "flatten_history_dim": False,
      "terms": {},
    },
  }
  path = _make_checkpoint(obs_history_cfg, tmp_path)

  env_cfg = _make_env_cfg(
    {
      "actor": ObservationGroupCfg(
        terms={"obs1": ObservationTermCfg(func=_simple_obs_func, params={})},
      ),
      "critic": ObservationGroupCfg(
        terms={"obs1": ObservationTermCfg(func=_simple_obs_func, params={})},
      ),
    }
  )

  # Verify no history before the call.
  assert env_cfg.observations["actor"].history_length is None
  assert env_cfg.observations["critic"].history_length is None

  _apply_obs_history_from_checkpoint(env_cfg, path)

  assert env_cfg.observations["actor"].history_length == 5
  assert env_cfg.observations["critic"].history_length == 3
  assert env_cfg.observations["critic"].flatten_history_dim is False


# ---------------------------------------------------------------------------
# _apply_obs_history_from_checkpoint — no-op / graceful degradation tests
# ---------------------------------------------------------------------------


def test_noop_on_old_checkpoint_without_key(tmp_path):
  """Checkpoints without obs_history_cfg silently leave env_cfg unchanged."""
  path = _make_checkpoint(obs_history_cfg=None, tmp_path=tmp_path)

  env_cfg = _make_env_cfg(
    {
      "actor": ObservationGroupCfg(
        history_length=None,
        terms={"obs1": ObservationTermCfg(func=_simple_obs_func, params={})},
      )
    }
  )

  _apply_obs_history_from_checkpoint(env_cfg, path)

  # Nothing should have changed.
  assert env_cfg.observations["actor"].history_length is None


def test_noop_on_missing_file(tmp_path, capsys):
  """A missing checkpoint path does not raise; a warning is printed."""
  env_cfg = _make_env_cfg(
    {
      "actor": ObservationGroupCfg(
        terms={"obs1": ObservationTermCfg(func=_simple_obs_func, params={})},
      )
    }
  )

  _apply_obs_history_from_checkpoint(env_cfg, str(tmp_path / "nonexistent.pt"))

  captured = capsys.readouterr()
  assert "[WARN]" in captured.out
  assert env_cfg.observations["actor"].history_length is None


def test_noop_for_group_not_in_env_cfg(tmp_path):
  """Groups in the checkpoint that are absent from env_cfg are ignored."""
  obs_history_cfg = {
    "actor": {"history_length": 5, "flatten_history_dim": True, "terms": {}},
    "some_extra_group": {"history_length": 2, "flatten_history_dim": True, "terms": {}},
  }
  path = _make_checkpoint(obs_history_cfg, tmp_path)

  env_cfg = _make_env_cfg(
    {
      "actor": ObservationGroupCfg(
        terms={"obs1": ObservationTermCfg(func=_simple_obs_func, params={})},
      )
    }
  )

  # Verify no history before the call.
  assert env_cfg.observations["actor"].history_length is None

  # Should not raise even though "some_extra_group" is not in env_cfg.
  _apply_obs_history_from_checkpoint(env_cfg, path)

  assert env_cfg.observations["actor"].history_length == 5


def test_noop_for_term_not_in_env_cfg(tmp_path):
  """Per-term entries in the checkpoint that are absent from env_cfg are ignored."""
  obs_history_cfg = {
    "actor": {
      "history_length": None,
      "flatten_history_dim": True,
      "terms": {
        "obs1": {"history_length": 4, "flatten_history_dim": True},
        "ghost_term": {"history_length": 4, "flatten_history_dim": True},
      },
    }
  }
  path = _make_checkpoint(obs_history_cfg, tmp_path)

  env_cfg = _make_env_cfg(
    {
      "actor": ObservationGroupCfg(
        terms={"obs1": ObservationTermCfg(func=_simple_obs_func, params={})},
      )
    }
  )

  # Verify no history before the call.
  assert env_cfg.observations["actor"].terms["obs1"].history_length == 0

  # Should not raise even though "ghost_term" is missing.
  _apply_obs_history_from_checkpoint(env_cfg, path)

  assert env_cfg.observations["actor"].terms["obs1"].history_length == 4


# ---------------------------------------------------------------------------
# MjlabOnPolicyRunner.save() — serialises obs history into the checkpoint
# ---------------------------------------------------------------------------


def _build_mock_runner(obs_group_cfgs: dict[str, ObservationGroupCfg | None]):
  """Build a minimal mock with the attributes MjlabOnPolicyRunner.save() accesses."""
  runner = MagicMock()
  runner.current_learning_iteration = 42
  runner.cfg = {"upload_model": False}
  runner.alg.save.return_value = {}

  # Wire up the observation manager cfg.
  runner.env.unwrapped.common_step_counter = 0
  runner.env.unwrapped.observation_manager.cfg = obs_group_cfgs

  return runner


def test_save_persists_group_level_history(tmp_path):
  """save() writes group-level history_length into the checkpoint."""
  from mjlab.rl.runner import MjlabOnPolicyRunner

  group_cfg = ObservationGroupCfg(
    history_length=5,
    flatten_history_dim=False,
    terms={"obs1": ObservationTermCfg(func=_simple_obs_func, params={})},
  )
  runner = _build_mock_runner({"actor": group_cfg})

  path = str(tmp_path / "model_1000.pt")
  MjlabOnPolicyRunner.save(runner, path)

  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  obs_hist = ckpt["infos"]["env_state"]["obs_history_cfg"]
  assert obs_hist["actor"]["history_length"] == 5
  assert obs_hist["actor"]["flatten_history_dim"] is False


def test_save_persists_per_term_history(tmp_path):
  """save() writes per-term history_length when no group-level override is set."""
  from mjlab.rl.runner import MjlabOnPolicyRunner

  term_cfg = ObservationTermCfg(
    func=_simple_obs_func,
    params={},
    history_length=4,
    flatten_history_dim=True,
  )
  group_cfg = ObservationGroupCfg(
    history_length=None,  # No group-level override.
    terms={"obs1": term_cfg},
  )
  runner = _build_mock_runner({"actor": group_cfg})

  path = str(tmp_path / "model_1000.pt")
  MjlabOnPolicyRunner.save(runner, path)

  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  obs_hist = ckpt["infos"]["env_state"]["obs_history_cfg"]
  assert obs_hist["actor"]["history_length"] is None
  assert obs_hist["actor"]["terms"]["obs1"]["history_length"] == 4
  assert obs_hist["actor"]["terms"]["obs1"]["flatten_history_dim"] is True


def test_save_skips_none_groups(tmp_path):
  """save() skips groups that are set to None."""
  from mjlab.rl.runner import MjlabOnPolicyRunner

  group_cfg = ObservationGroupCfg(
    terms={"obs1": ObservationTermCfg(func=_simple_obs_func, params={})},
  )
  runner = _build_mock_runner({"actor": group_cfg, "disabled_group": None})

  path = str(tmp_path / "model_1000.pt")
  MjlabOnPolicyRunner.save(runner, path)

  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  obs_hist = ckpt["infos"]["env_state"]["obs_history_cfg"]
  assert "actor" in obs_hist
  assert "disabled_group" not in obs_hist


# ---------------------------------------------------------------------------
# Full round-trip: save() → _apply_obs_history_from_checkpoint()
# ---------------------------------------------------------------------------


def test_round_trip_group_level(tmp_path):
  """History written by save() is faithfully restored by _apply_obs_history_from_checkpoint."""
  from mjlab.rl.runner import MjlabOnPolicyRunner

  group_cfg = ObservationGroupCfg(
    history_length=7,
    flatten_history_dim=False,
    terms={"obs1": ObservationTermCfg(func=_simple_obs_func, params={})},
  )
  runner = _build_mock_runner({"actor": group_cfg})

  path = str(tmp_path / "model_1000.pt")
  MjlabOnPolicyRunner.save(runner, path)

  env_cfg = _make_env_cfg(
    {
      "actor": ObservationGroupCfg(
        history_length=None,  # Default before restore.
        terms={"obs1": ObservationTermCfg(func=_simple_obs_func, params={})},
      )
    }
  )

  # Verify no history before the call.
  assert env_cfg.observations["actor"].history_length is None
  assert env_cfg.observations["actor"].terms["obs1"].history_length == 0

  _apply_obs_history_from_checkpoint(env_cfg, path)

  assert env_cfg.observations["actor"].history_length == 7
  assert env_cfg.observations["actor"].flatten_history_dim is False


def test_round_trip_per_term(tmp_path):
  """Per-term history written by save() is faithfully restored."""
  from mjlab.rl.runner import MjlabOnPolicyRunner

  saved_group_cfg = ObservationGroupCfg(
    history_length=None,
    terms={
      "obs1": ObservationTermCfg(
        func=_simple_obs_func, params={}, history_length=3, flatten_history_dim=True
      ),
      "obs2": ObservationTermCfg(func=_simple_obs_func, params={}, history_length=0),
    },
  )
  runner = _build_mock_runner({"actor": saved_group_cfg})

  path = str(tmp_path / "model_1000.pt")
  MjlabOnPolicyRunner.save(runner, path)

  # env_cfg starts with default (no history).
  env_cfg = _make_env_cfg(
    {
      "actor": ObservationGroupCfg(
        terms={
          "obs1": ObservationTermCfg(func=_simple_obs_func, params={}),
          "obs2": ObservationTermCfg(func=_simple_obs_func, params={}),
        },
      )
    }
  )

  # Verify no history before the call.
  assert env_cfg.observations["actor"].history_length is None
  assert env_cfg.observations["actor"].terms["obs1"].history_length == 0
  assert env_cfg.observations["actor"].terms["obs2"].history_length == 0

  _apply_obs_history_from_checkpoint(env_cfg, path)

  assert env_cfg.observations["actor"].terms["obs1"].history_length == 3
  assert env_cfg.observations["actor"].terms["obs1"].flatten_history_dim is True
  assert env_cfg.observations["actor"].terms["obs2"].history_length == 0
