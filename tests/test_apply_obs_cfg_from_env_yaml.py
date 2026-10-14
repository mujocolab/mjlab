"""Tests for _apply_obs_cfg_from_env_yaml in scripts/play.py."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from mjlab.scripts.play import _apply_obs_cfg_from_env_yaml
from mjlab.utils.noise.noise_cfg import UniformNoiseCfg


@dataclass
class FakeTermCfg:
  func: Any
  params: dict = field(default_factory=dict)
  history_length: int = 0
  flatten_history_dim: bool = True


@dataclass
class FakeGroupCfg:
  terms: dict
  history_length: int | None = None
  flatten_history_dim: bool = True
  concatenate_terms: bool = True


@dataclass
class FakeTermCfgWithNoise:
  func: Any
  params: dict = field(default_factory=dict)
  noise: UniformNoiseCfg = field(default_factory=UniformNoiseCfg)
  history_length: int = 0
  flatten_history_dim: bool = True


def _make_env_cfg(**groups: Any):
  """Return a minimal env_cfg whose .observations is a plain dict."""
  ns = SimpleNamespace()
  ns.observations = dict(groups)
  return ns


# Two simple functions whose qualified names differ — used for func validation.
def _func_a(env):
  pass


def _func_b(env):
  pass


def _patch_load_yaml(yaml_data: dict):
  """Patch mjlab.scripts.play.load_yaml to return *yaml_data*."""
  return patch("mjlab.scripts.play.load_yaml", return_value=yaml_data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_raises_file_not_found_for_missing_path(tmp_path):
  env_cfg = _make_env_cfg()
  missing = tmp_path / "no_such_file.yaml"
  with pytest.raises(FileNotFoundError, match="env.yaml not found"):
    _apply_obs_cfg_from_env_yaml(env_cfg, missing)


def test_no_observations_section_is_a_noop(tmp_path):
  """YAML with no 'observations' key should leave env_cfg untouched."""
  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.write_text("")  # file exists but load_yaml returns {}

  group = FakeGroupCfg(terms={"joint_pos": FakeTermCfg(func=_func_a)})
  env_cfg = _make_env_cfg(actor=group)

  with _patch_load_yaml({}):
    _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)

  # Nothing should have changed.
  assert env_cfg.observations["actor"].history_length is None


def test_raises_for_unknown_group(tmp_path):
  """RuntimeError when YAML references a group absent from env_cfg."""
  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.touch()

  env_cfg = _make_env_cfg(actor=FakeGroupCfg(terms={}))

  yaml_data = {"observations": {"critic": {}}}
  with _patch_load_yaml(yaml_data):
    with pytest.raises(RuntimeError, match="Observation group 'critic'"):
      _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)


def test_raises_for_unknown_term(tmp_path):
  """RuntimeError when YAML references a term absent from env_cfg."""
  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.touch()

  group = FakeGroupCfg(terms={"joint_pos": FakeTermCfg(func=_func_a)})
  env_cfg = _make_env_cfg(actor=group)

  yaml_data = {
    "observations": {
      "actor": {
        "terms": {
          "joint_pos": {},
          "ghost_term": {},  # not in env_cfg
        }
      }
    }
  }
  with _patch_load_yaml(yaml_data):
    with pytest.raises(RuntimeError, match="Observation term 'actor.ghost_term'"):
      _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)


def test_raises_on_func_mismatch(tmp_path):
  """RuntimeError when the YAML func doesn't match the env_cfg func."""
  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.touch()

  group = FakeGroupCfg(terms={"joint_pos": FakeTermCfg(func=_func_a)})
  env_cfg = _make_env_cfg(actor=group)

  yaml_data = {
    "observations": {
      "actor": {
        "terms": {
          "joint_pos": {"func": _func_b},  # mismatched callable
        }
      }
    }
  }
  with _patch_load_yaml(yaml_data):
    with pytest.raises(RuntimeError, match="func mismatch"):
      _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)


def test_applies_group_level_fields(tmp_path):
  """Group-level fields in the YAML are written onto the group config."""
  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.touch()

  group = FakeGroupCfg(terms={"joint_pos": FakeTermCfg(func=_func_a)})
  env_cfg = _make_env_cfg(actor=group)

  yaml_data = {
    "observations": {
      "actor": {
        "history_length": 10,
        "flatten_history_dim": False,
        "terms": {"joint_pos": {}},
      }
    }
  }
  with _patch_load_yaml(yaml_data):
    _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)

  actor = env_cfg.observations["actor"]
  assert actor.history_length == 10
  assert actor.flatten_history_dim is False


def test_applies_term_level_fields(tmp_path):
  """Term-level fields (excluding func/params) are written onto the term config."""
  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.touch()

  group = FakeGroupCfg(terms={"joint_pos": FakeTermCfg(func=_func_a)})
  env_cfg = _make_env_cfg(actor=group)

  yaml_data = {
    "observations": {
      "actor": {
        "terms": {
          "joint_pos": {
            "history_length": 5,
            "flatten_history_dim": False,
          }
        }
      }
    }
  }
  with _patch_load_yaml(yaml_data):
    _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)

  term = env_cfg.observations["actor"].terms["joint_pos"]
  assert term.history_length == 5
  assert term.flatten_history_dim is False


def test_func_and_params_are_not_overwritten(tmp_path):
  """func and params in the YAML must NOT overwrite env_cfg values."""
  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.touch()

  original_params = {"scale": 1.0}
  group = FakeGroupCfg(
    terms={"joint_pos": FakeTermCfg(func=_func_a, params=original_params)}
  )
  env_cfg = _make_env_cfg(actor=group)

  yaml_data = {
    "observations": {
      "actor": {
        "terms": {
          "joint_pos": {
            "func": _func_a,  # same func — should not be written
            "params": {"scale": 99.0},  # should be ignored
            "history_length": 3,
          }
        }
      }
    }
  }
  with _patch_load_yaml(yaml_data):
    _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)

  term = env_cfg.observations["actor"].terms["joint_pos"]
  assert term.func is _func_a
  assert term.params == original_params  # unchanged
  assert term.history_length == 3  # other fields still applied


def test_drops_extra_terms_not_in_yaml(tmp_path, capsys):
  """Terms present in env_cfg but absent from the YAML are removed."""
  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.touch()

  group = FakeGroupCfg(
    terms={
      "joint_pos": FakeTermCfg(func=_func_a),
      "extra_term": FakeTermCfg(func=_func_b),  # not in YAML
    }
  )
  env_cfg = _make_env_cfg(actor=group)

  yaml_data = {
    "observations": {
      "actor": {
        "terms": {"joint_pos": {}},  # only joint_pos
      }
    }
  }
  with _patch_load_yaml(yaml_data):
    _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)

  actor = env_cfg.observations["actor"]
  assert "extra_term" not in actor.terms
  assert "joint_pos" in actor.terms

  captured = capsys.readouterr()
  assert "extra_term" in captured.out
  assert "WARN" in captured.out


def test_no_func_in_term_cfg_skips_func_validation(tmp_path):
  """When term_cfg.func is None, the func field in the YAML is not validated."""
  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.touch()

  group = FakeGroupCfg(terms={"joint_pos": FakeTermCfg(func=None)})
  env_cfg = _make_env_cfg(actor=group)

  yaml_data = {
    "observations": {
      "actor": {
        "terms": {
          "joint_pos": {"func": _func_b, "history_length": 2},
        }
      }
    }
  }
  with _patch_load_yaml(yaml_data):
    # Should not raise despite func mismatch (term_cfg.func is None).
    _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)

  assert env_cfg.observations["actor"].terms["joint_pos"].history_length == 2


def test_nested_dataclass_non_init_fields_are_ignored(tmp_path):
  """init=False fields (e.g. _tensor_cache on UniformNoiseCfg) present in the YAML
  dict are not written back onto the dataclass instance."""

  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.touch()

  noise_obj = UniformNoiseCfg(operation="add", n_min=-0.5, n_max=0.5)
  term = FakeTermCfgWithNoise(func=_func_a, noise=noise_obj)
  group = FakeGroupCfg(terms={"base_lin_vel": term})
  env_cfg = _make_env_cfg(actor=group)

  # YAML dict includes _tensor_cache which is an init=False field on UniformNoiseCfg.
  yaml_data = {
    "observations": {
      "actor": {
        "terms": {
          "base_lin_vel": {
            "noise": {
              "operation": "add",
              "_tensor_cache": {"cuda:0": {}},  # should not overwrite the live cache
              "n_min": -0.5,
              "n_max": 0.5,
            }
          }
        }
      }
    }
  }
  with _patch_load_yaml(yaml_data):
    _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)

  updated = env_cfg.observations["actor"].terms["base_lin_vel"].noise
  assert updated is noise_obj
  assert updated.n_min == -0.5
  assert updated.n_max == 0.5
  # _tensor_cache must not have been overwritten with the YAML value.
  assert updated._tensor_cache != {"cuda:0": {}}


def test_nested_dataclass_same_values_produces_no_output(tmp_path, capsys):
  """When the YAML values for a nested dataclass match the existing values,
  no change output is printed."""

  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.touch()

  term = FakeTermCfgWithNoise(
    func=_func_a, noise=UniformNoiseCfg(operation="add", n_min=-0.01, n_max=0.01)
  )
  group = FakeGroupCfg(terms={"joint_pos": term})
  env_cfg = _make_env_cfg(actor=group)

  yaml_data = {
    "observations": {
      "actor": {
        "terms": {
          "joint_pos": {
            "noise": {
              "operation": "add",
              "_tensor_cache": {},
              "n_min": -0.01,
              "n_max": 0.01,
            }
          }
        }
      }
    }
  }
  with _patch_load_yaml(yaml_data):
    _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)

  captured = capsys.readouterr()
  # Only the INFO line should be printed, no field-change lines.
  lines = [ln for ln in captured.out.splitlines() if not ln.startswith("[INFO]")]
  assert lines == []


def test_nested_dataclass_changed_values_updates_and_prints(tmp_path, capsys):
  """When the YAML values for a nested dataclass differ from the existing values,
  the field is updated in-place and the change is printed."""

  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.touch()

  noise_obj = UniformNoiseCfg(operation="add", n_min=-0.01, n_max=0.01)
  term = FakeTermCfgWithNoise(func=_func_a, noise=noise_obj)
  group = FakeGroupCfg(terms={"joint_vel": term})
  env_cfg = _make_env_cfg(actor=group)

  yaml_data = {
    "observations": {
      "actor": {
        "terms": {
          "joint_vel": {
            "noise": {
              "operation": "add",
              "_tensor_cache": {},
              "n_min": -1.5,
              "n_max": 1.5,
            }
          }
        }
      }
    }
  }
  with _patch_load_yaml(yaml_data):
    _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)

  updated = env_cfg.observations["actor"].terms["joint_vel"].noise
  assert updated is noise_obj
  assert updated.n_min == -1.5
  assert updated.n_max == 1.5

  captured = capsys.readouterr()
  assert "n_min" in captured.out
  assert "n_max" in captured.out


def test_nested_dataclass_field_updated_in_place(tmp_path):
  """Nested dataclass values are updated in-place, not replaced with a raw dict."""

  @dataclass
  class NestedCfg:
    alpha: float = 0.0
    beta: int = 0

  @dataclass
  class TermWithNested:
    func: Any
    params: dict = field(default_factory=dict)
    nested: NestedCfg = field(default_factory=NestedCfg)
    history_length: int = 0
    flatten_history_dim: bool = True

  env_yaml_path = tmp_path / "env.yaml"
  env_yaml_path.touch()

  nested_obj = NestedCfg(alpha=1.0, beta=2)
  term = TermWithNested(func=_func_a, nested=nested_obj)

  @dataclass
  class GroupWithNested:
    terms: dict
    history_length: int | None = None
    flatten_history_dim: bool = True
    concatenate_terms: bool = True

  group = GroupWithNested(terms={"joint_pos": term})
  env_cfg = _make_env_cfg(actor=group)

  yaml_data = {
    "observations": {
      "actor": {
        "terms": {
          "joint_pos": {
            "nested": {"alpha": 9.9},  # only alpha, beta untouched
          }
        }
      }
    }
  }
  with _patch_load_yaml(yaml_data):
    _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)

  updated = env_cfg.observations["actor"].terms["joint_pos"].nested
  # The object should still be the same instance (updated in-place).
  assert updated is nested_obj
  assert updated.alpha == 9.9
  assert updated.beta == 2  # unchanged
