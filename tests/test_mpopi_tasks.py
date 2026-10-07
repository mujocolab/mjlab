"""Tests for the Mpopi-G1-2k tasks and the method presets."""

import tyro

import mjlab
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mpopi_train.config import MpopiRunnerCfg
from mpopi_train.presets import METHODS, MIN_ACTION_STD
from mpopi_train.runner import MpopiVelocityRunner
from mpopi_train.tasks import g1_2k_task_id


def _mpopi(method: str):
  cfg = load_rl_cfg(g1_2k_task_id(method))
  assert isinstance(cfg, MpopiRunnerCfg)
  return cfg.algorithm.mpopi


def test_every_method_has_a_g1_2k_task():
  tasks = list_tasks()
  for method in METHODS:
    task = g1_2k_task_id(method)
    assert task in tasks
    assert load_runner_cls(task) is MpopiVelocityRunner
    assert load_rl_cfg(task).run_name == method


def test_g1_2k_task_reaches_top_speed_within_its_iterations():
  task = g1_2k_task_id("PPO")
  rl_cfg = load_rl_cfg(task)
  stages = load_env_cfg(task).curriculum["command_vel"].params["velocity_stages"]
  last = stages[-1]
  assert rl_cfg.max_iterations == 2_000
  assert last["lin_vel_x"][1] == 1.5
  assert last["step"] < rl_cfg.max_iterations * rl_cfg.num_steps_per_env
  play_twist = load_env_cfg(task, play=True).commands["twist"]
  assert isinstance(play_twist, UniformVelocityCommandCfg)
  assert play_twist.ranges.lin_vel_x[1] == 1.5


def test_presets_configure_the_compared_methods():
  ppo, replay = _mpopi("PPO"), _mpopi("Replay-IS")
  dagger, combo = _mpopi("DAgger"), _mpopi("Replay-IS-DAgger")
  assert ppo.mode == "ppo"
  assert replay.mode == "mpopi_ppo" and replay.replay_buffer_size == 4
  for cfg in (dagger, combo):
    assert cfg.mode == "mpc_ppo"
    assert cfg.mpc.driver == "policy" and not cfg.mpc.use_in_ppo
    assert (cfg.mpc.num_envs, cfg.mpc.collect_every) == (64, 5)
    assert cfg.mpc.planner.num_samples == 16 and cfg.mpc.planner.iterations == 2
  assert not dagger.mpc.replay_own_rollouts and combo.mpc.replay_own_rollouts
  for cfg in (ppo, replay, dagger, combo):
    assert cfg.min_action_std == MIN_ACTION_STD
    cfg.validate()


def test_train_cli_overrides_preset_settings():
  task = g1_2k_task_id("Replay-IS-DAgger")
  args = ["--agent.seed", "3", "--agent.algorithm.mpopi.mpc.num-envs", "32"]
  cfg = tyro.cli(
    TrainConfig,
    args=args,
    default=TrainConfig.from_task(task),
    config=mjlab.TYRO_FLAGS,
  )
  assert isinstance(cfg.agent, MpopiRunnerCfg)
  assert cfg.agent.seed == 3
  assert cfg.agent.algorithm.mpopi.mpc.num_envs == 32
  assert cfg.agent.algorithm.mpopi.mpc.replay_own_rollouts
