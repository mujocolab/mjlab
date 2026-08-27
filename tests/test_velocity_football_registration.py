"""Registration contracts for the active coordinate-football tasks."""

from typing import Any, cast

import pytest

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity_football.config.g1 import (
  BASE_TASK_ID,
  TEACHER_BASELINE_TASK_ID,
)
from mjlab.tasks.velocity_football.rl import VelocityOnPolicyRunner


def test_only_two_coordinate_football_tasks_are_registered() -> None:
  task_ids = {
    task_id
    for task_id in list_tasks()
    if task_id.startswith("Mjlab-Velocity-Football") and "-Depth-" not in task_id
  }
  assert task_ids == {BASE_TASK_ID, TEACHER_BASELINE_TASK_ID}


@pytest.mark.parametrize("task_id", (BASE_TASK_ID, TEACHER_BASELINE_TASK_ID))
def test_active_coordinate_tasks_load(task_id: str) -> None:
  training_cfg = load_env_cfg(task_id)
  play_cfg = load_env_cfg(task_id, play=True)

  assert training_cfg.scene.num_envs >= 1
  assert play_cfg.scene.num_envs == 1
  assert load_runner_cls(task_id) is VelocityOnPolicyRunner


def test_teacher_baseline_observation_and_policy_contract() -> None:
  cfg = load_env_cfg(TEACHER_BASELINE_TASK_ID)
  runner_cfg = cast(Any, load_rl_cfg(TEACHER_BASELINE_TASK_ID))

  actor = cfg.observations["actor"]
  actor_history = cfg.observations["actor_history"]
  assert actor.history_length == 5
  assert actor_history.history_length == 10
  assert tuple(actor_history.terms) == (
    "ball_pos_b",
    "ball_to_feet_vectors_b",
    "ball_visible_mask",
  )
  assert (
    sum(
      term.params.get("transition_dropout_probability", 0.0) > 0.0
      for term in actor_history.terms.values()
    )
    == 3
  )
  assert runner_cfg.actor.hidden_dims == (512, 256, 128)
  assert runner_cfg.critic.hidden_dims == (512, 256, 128)
  assert runner_cfg.obs_groups["actor"] == ("actor", "actor_history")
  assert runner_cfg.algorithm.entropy_coef == pytest.approx(0.01)


def test_base_task_remains_a_simple_smoke_baseline() -> None:
  cfg = load_env_cfg(BASE_TASK_ID)
  runner_cfg = cast(Any, load_rl_cfg(BASE_TASK_ID))

  assert "actor" in cfg.observations
  assert "critic" in cfg.observations
  assert runner_cfg.actor.hidden_dims == (512, 256, 128)
