"""Tests for resolve_curriculum_iterations()."""

import copy

import pytest

from mjlab.managers.curriculum_manager import (
  CurriculumTermCfg,
  resolve_curriculum_iterations,
)


def _dummy_func(*args, **kwargs):
  pass


def _make_curriculum(stages_key, stages):
  """Build a minimal curriculum dict with one term."""
  return {
    "term": CurriculumTermCfg(
      func=_dummy_func,
      params={"some_name": "twist", stages_key: stages},
    )
  }


def test_converts_iteration_to_step():
  stages = [
    {"iteration": 0, "lin_vel_x": (-1.0, 1.0)},
    {"iteration": 5000, "lin_vel_x": (-2.0, 2.0)},
  ]
  curriculum = _make_curriculum("velocity_stages", stages)
  resolve_curriculum_iterations(curriculum, num_steps_per_env=24)

  resolved = curriculum["term"].params["velocity_stages"]
  assert resolved[0] == {"step": 0, "lin_vel_x": (-1.0, 1.0)}
  assert resolved[1] == {"step": 120000, "lin_vel_x": (-2.0, 2.0)}
  assert all("iteration" not in stage for stage in resolved)


def test_leaves_step_based_stages_untouched():
  stages = [
    {"step": 0, "weight": -0.01},
    {"step": 12000, "weight": -1.0},
  ]
  curriculum = _make_curriculum("weight_stages", stages)
  original = copy.deepcopy(curriculum)
  resolve_curriculum_iterations(curriculum, num_steps_per_env=24)

  assert (
    curriculum["term"].params["weight_stages"]
    == original["term"].params["weight_stages"]
  )


def test_leaves_non_stage_params_untouched():
  curriculum = _make_curriculum(
    "velocity_stages",
    [{"iteration": 100, "lin_vel_x": (-1.0, 1.0)}],
  )
  resolve_curriculum_iterations(curriculum, num_steps_per_env=24)
  assert curriculum["term"].params["some_name"] == "twist"


def test_raises_on_ambiguous_stage():
  stages = [{"iteration": 100, "step": 2400, "weight": -1.0}]
  curriculum = _make_curriculum("weight_stages", stages)
  with pytest.raises(ValueError, match="both 'iteration' and 'step'"):
    resolve_curriculum_iterations(curriculum, num_steps_per_env=24)


def test_velocity_env_cfg_stages_resolve():
  """End-to-end: velocity config's command_vel stages resolve correctly."""
  from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

  env_cfg = make_velocity_env_cfg()
  resolve_curriculum_iterations(env_cfg.curriculum, num_steps_per_env=24)

  stages = env_cfg.curriculum["command_vel"].params["velocity_stages"]
  assert stages[0]["step"] == 0
  assert stages[1]["step"] == 5000 * 24
  assert stages[2]["step"] == 10000 * 24
  assert all("iteration" not in stage for stage in stages)


def test_lift_cube_env_cfg_stages_resolve():
  """End-to-end: lift cube config's weight stages resolve correctly."""
  from mjlab.tasks.manipulation.lift_cube_env_cfg import (
    make_lift_cube_env_cfg,
  )

  env_cfg = make_lift_cube_env_cfg()
  resolve_curriculum_iterations(env_cfg.curriculum, num_steps_per_env=24)

  stages = env_cfg.curriculum["joint_vel_hinge_weight"].params["weight_stages"]
  assert stages[0]["step"] == 0
  assert stages[1]["step"] == 500 * 24
  assert stages[2]["step"] == 1000 * 24
  assert all("iteration" not in stage for stage in stages)
