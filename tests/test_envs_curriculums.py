"""Tests for curriculum terms."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from mjlab.envs.mdp.curriculums import reward_curriculum, termination_curriculum
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp.curriculums import terrain_levels_vel


def _reward_func(env):
  return torch.ones(env.num_envs)


def _termination_func(env):
  return torch.zeros(env.num_envs, dtype=torch.bool)


def _make_reward_cfg(
  weight: float = 1.0,
  params: dict | None = None,
) -> RewardTermCfg:
  return RewardTermCfg(
    func=_reward_func,
    weight=weight,
    params=params if params is not None else {"std": 0.5, "scale": 1.0},
  )


def _make_termination_cfg(
  params: dict | None = None,
) -> TerminationTermCfg:
  return TerminationTermCfg(
    func=_termination_func,
    params=params if params is not None else {"threshold": float("inf")},
  )


def _build_reward(env, reward_name, stages):
  params = {"reward_name": reward_name, "stages": stages}
  cfg = CurriculumTermCfg(func=reward_curriculum, params=params)
  instance = reward_curriculum(cfg, env)
  return instance(env, env_ids=torch.tensor([0, 1]), **params)


def _build_termination(env, termination_name, stages):
  params = {"termination_name": termination_name, "stages": stages}
  cfg = CurriculumTermCfg(func=termination_curriculum, params=params)
  instance = termination_curriculum(cfg, env)
  return instance(env, env_ids=torch.tensor([0, 1]), **params)


def _make_reward_env(step_counter, reward_cfg):
  env = Mock()
  env.common_step_counter = step_counter
  env.reward_manager.get_term_cfg.return_value = reward_cfg
  return env


def _make_termination_env(step_counter, term_cfg):
  env = Mock()
  env.common_step_counter = step_counter
  env.termination_manager.get_term_cfg.return_value = term_cfg
  return env


def _make_terrain_levels_env(
  command: torch.Tensor,
  distance: torch.Tensor,
  *,
  max_episode_length_s: float = 10.0,
  terrain_size: float = 8.0,
):
  num_envs = command.shape[0]
  env_origins = torch.zeros((num_envs, 3))
  root_link_pos_w = env_origins.clone()
  root_link_pos_w[:, 0] = distance
  asset = SimpleNamespace(
    data=SimpleNamespace(root_link_pos_w=root_link_pos_w),
  )

  terrain_generator = SimpleNamespace(
    size=(terrain_size, terrain_size),
    sub_terrains={"flat": object(), "rough": object()},
  )
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=terrain_generator),
    update_env_origins=Mock(),
    terrain_levels=torch.tensor([1, 3], dtype=torch.int64)[:num_envs],
    terrain_types=torch.tensor([0, 1], dtype=torch.int64)[:num_envs],
    terrain_origins=torch.zeros((4, 2, 3)),
  )

  scene = MagicMock()
  scene.__getitem__.return_value = asset
  scene.env_origins = env_origins
  scene.terrain = terrain

  env = SimpleNamespace(
    scene=scene,
    command_manager=Mock(),
    max_episode_length_s=max_episode_length_s,
  )
  env.command_manager.get_command.return_value = command

  return env, torch.arange(num_envs), terrain


def _terrain_level_masks(terrain) -> tuple[torch.Tensor, torch.Tensor]:
  _, move_up, move_down = terrain.update_env_origins.call_args.args
  return move_up, move_down


# Terrain levels: velocity task


def test_terrain_levels_vel_promotes_full_expected_distance():
  command = torch.tensor([[0.5, 0.0, 0.0]])
  env, env_ids, terrain = _make_terrain_levels_env(
    command,
    torch.tensor([5.0]),
    max_episode_length_s=10.0,
  )

  terrain_levels_vel(env, env_ids, command_name="twist")

  move_up, move_down = _terrain_level_masks(terrain)
  assert move_up.tolist() == [True]
  assert move_down.tolist() == [False]


def test_terrain_levels_vel_promotes_low_speed_tracking_on_short_tiles():
  command = torch.tensor([[0.2, 0.0, 0.0]])
  env, env_ids, terrain = _make_terrain_levels_env(
    command,
    torch.tensor([1.0]),
    max_episode_length_s=5.0,
    terrain_size=3.0,
  )

  terrain_levels_vel(env, env_ids, command_name="twist")

  move_up, move_down = _terrain_level_masks(terrain)
  assert move_up.tolist() == [True]
  assert move_down.tolist() == [False]


def test_terrain_levels_vel_demotes_under_travel_and_masks_are_exclusive():
  command = torch.tensor([[0.5, 0.0, 0.0]])
  env, env_ids, terrain = _make_terrain_levels_env(
    command,
    torch.tensor([2.0]),
    max_episode_length_s=10.0,
  )

  terrain_levels_vel(env, env_ids, command_name="twist")

  move_up, move_down = _terrain_level_masks(terrain)
  assert move_up.tolist() == [False]
  assert move_down.tolist() == [True]
  assert not torch.any(move_up & move_down)


def test_terrain_levels_vel_ignores_near_zero_commands():
  command = torch.tensor([[0.01, 0.0, 0.0]])
  env, env_ids, terrain = _make_terrain_levels_env(
    command,
    torch.tensor([0.02]),
    max_episode_length_s=1.0,
  )

  terrain_levels_vel(env, env_ids, command_name="twist")

  move_up, move_down = _terrain_level_masks(terrain)
  assert move_up.tolist() == [False]
  assert move_down.tolist() == [False]


def test_terrain_levels_vel_original_arguments_and_result_keys():
  command = torch.tensor(
    [
      [0.5, 0.0, 0.0],
      [0.5, 0.0, 0.0],
    ]
  )
  env, env_ids, _terrain = _make_terrain_levels_env(
    command,
    torch.tensor([5.0, 0.0]),
  )

  result = terrain_levels_vel(env, env_ids, command_name="twist")

  assert set(result) == {"mean", "max", "flat", "rough"}


# Reward: weight


def test_reward_weight_unchanged_before_threshold():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  _build_reward(env, "r", [{"step": 100, "weight": 2.0}])
  assert rc.weight == pytest.approx(1.0)


def test_reward_weight_applied_at_threshold():
  rc = _make_reward_cfg()
  env = _make_reward_env(100, rc)
  _build_reward(env, "r", [{"step": 100, "weight": 2.0}])
  assert rc.weight == pytest.approx(2.0)


def test_reward_weight_later_stage_wins():
  rc = _make_reward_cfg()
  env = _make_reward_env(500, rc)
  _build_reward(
    env,
    "r",
    [
      {"step": 0, "weight": 0.5},
      {"step": 100, "weight": 1.5},
      {"step": 400, "weight": 3.0},
    ],
  )
  assert rc.weight == pytest.approx(3.0)


def test_reward_weight_partial_application():
  rc = _make_reward_cfg()
  env = _make_reward_env(150, rc)
  _build_reward(
    env,
    "r",
    [
      {"step": 100, "weight": 2.0},
      {"step": 200, "weight": 4.0},
    ],
  )
  assert rc.weight == pytest.approx(2.0)


def test_step_zero_applies_immediately():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  _build_reward(env, "r", [{"step": 0, "weight": 9.0}])
  assert rc.weight == pytest.approx(9.0)


# Reward: params


def test_reward_params_updated():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  _build_reward(env, "r", [{"step": 100, "params": {"std": 0.2}}])
  assert rc.params["std"] == 0.2


def test_reward_params_unchanged_before_threshold():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  _build_reward(env, "r", [{"step": 100, "params": {"std": 0.2}}])
  assert rc.params["std"] == 0.5


def test_reward_multiple_params_updated():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  _build_reward(env, "r", [{"step": 100, "params": {"std": 0.2, "scale": 2.0}}])
  assert rc.params["std"] == 0.2
  assert rc.params["scale"] == 2.0


# Reward: combined weight + params


def test_reward_weight_and_params_in_same_stage():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  _build_reward(env, "r", [{"step": 100, "weight": 5.0, "params": {"std": 0.1}}])
  assert rc.weight == pytest.approx(5.0)
  assert rc.params["std"] == 0.1


# Termination: params


def test_termination_params_updated():
  tc = _make_termination_cfg()
  env = _make_termination_env(200, tc)
  _build_termination(env, "energy", [{"step": 100, "params": {"threshold": 500.0}}])
  assert tc.params["threshold"] == 500.0


def test_termination_params_unchanged_before_threshold():
  tc = _make_termination_cfg()
  env = _make_termination_env(0, tc)
  _build_termination(env, "energy", [{"step": 100, "params": {"threshold": 500.0}}])
  assert tc.params["threshold"] == float("inf")


def test_termination_later_stage_wins():
  tc = _make_termination_cfg()
  env = _make_termination_env(500, tc)
  _build_termination(
    env,
    "energy",
    [
      {"step": 0, "params": {"threshold": 1000.0}},
      {"step": 100, "params": {"threshold": 700.0}},
      {"step": 400, "params": {"threshold": 400.0}},
    ],
  )
  assert tc.params["threshold"] == 400.0


# Validation: shared engine


def test_unknown_reward_param_raises():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  params = {"reward_name": "r", "stages": [{"step": 0, "params": {"stdd": 0.2}}]}
  cfg = CurriculumTermCfg(func=reward_curriculum, params=params)
  with pytest.raises(KeyError, match="unknown param"):
    reward_curriculum(cfg, env)


def test_unknown_termination_param_raises():
  tc = _make_termination_cfg()
  env = _make_termination_env(0, tc)
  params = {
    "termination_name": "energy",
    "stages": [{"step": 0, "params": {"thresholddd": 1.0}}],
  }
  cfg = CurriculumTermCfg(func=termination_curriculum, params=params)
  with pytest.raises(KeyError, match="unknown param"):
    termination_curriculum(cfg, env)


def test_unsorted_stages_raise():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  params = {
    "reward_name": "r",
    "stages": [
      {"step": 200, "weight": 1.0},
      {"step": 100, "weight": 2.0},
    ],
  }
  cfg = CurriculumTermCfg(func=reward_curriculum, params=params)
  with pytest.raises(ValueError, match="nondecreasing"):
    reward_curriculum(cfg, env)


def test_duplicate_steps_allowed():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  _build_reward(
    env,
    "r",
    [
      {"step": 100, "weight": 2.0},
      {"step": 100, "params": {"std": 0.1}},
    ],
  )
  assert rc.weight == pytest.approx(2.0)
  assert rc.params["std"] == 0.1


# Logging keys


def test_reward_logs_only_staged_keys():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  result = _build_reward(
    env, "r", [{"step": 100, "weight": 5.0, "params": {"std": 0.2}}]
  )
  assert result["weight"].item() == pytest.approx(5.0)
  assert result["std"].item() == pytest.approx(0.2)
  assert "scale" not in result  # Not in any stage.


def test_reward_omits_weight_when_not_staged():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  result = _build_reward(env, "r", [{"step": 100, "params": {"std": 0.2}}])
  assert "weight" not in result
  assert "std" in result


def test_termination_log_keys():
  tc = _make_termination_cfg()
  env = _make_termination_env(200, tc)
  result = _build_termination(
    env, "energy", [{"step": 100, "params": {"threshold": 500.0}}]
  )
  assert "threshold" in result
  assert result["threshold"].item() == pytest.approx(500.0)
  assert "weight" not in result  # No weight for termination.
