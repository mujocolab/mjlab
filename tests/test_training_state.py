"""Real native PPO continuation, including simulation and manager state."""

from dataclasses import asdict
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv, mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.rl.training_state import ComponentState
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.utils.noise.noise_cfg import GaussianNoiseCfg


def make_runner(*, recurrent=False, rank=0, log_dir=None, device="cpu", **overrides):
  cfg = cartpole_balance_env_cfg()
  cfg.seed = 19 + rank
  cfg.scene.num_envs = 2
  cfg.episode_length_s = 0.35
  for mode in ("reset", "interval"):
    cfg.events[f"damping_{mode}"] = EventTermCfg(
      func=dr.joint_damping,
      mode=mode,
      interval_range_s=(0.12, 0.2) if mode == "interval" else None,
      params={"ranges": (0.01, 0.1), "asset_cfg": SceneEntityCfg("cartpole")},
    )
  cfg.curriculum["reward_stages"] = CurriculumTermCfg(
    func=mdp.reward_curriculum,
    params={
      "reward_name": "smooth_reward",
      "stages": [
        {"step": 0, "weight": 1.0},
        {"step": 6, "weight": 0.8},
        {"step": 10, "weight": 1.2},
      ],
    },
  )
  # This exercises live delay phase/lag/history and stochastic observations, plus
  # repeated auto-resets on either side of an update/checkpoint boundary.
  term = cfg.observations["actor"].terms["cart_pos"]
  term.history_length = 2
  term.delay_max_lag = 2
  term.delay_update_period = 3
  term.noise = GaussianNoiseCfg(std=0.03)
  agent = RslRlOnPolicyRunnerCfg(
    stateful_resume=True,
    num_steps_per_env=4,
    upload_model=False,
    logger="tensorboard",
    **overrides,
  )
  agent.actor.hidden_dims = (8,)
  agent.critic.hidden_dims = (8,)
  agent.actor.obs_normalization = True
  agent.critic.obs_normalization = True
  agent.algorithm.num_learning_epochs = 1
  agent.algorithm.num_mini_batches = 1
  if recurrent:
    for model in (agent.actor, agent.critic):
      model.class_name = "RNNModel"
      model.rnn_type = "lstm"
      model.rnn_hidden_dim = 4
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=device))
  try:
    return MjlabOnPolicyRunner(env, asdict(agent), log_dir=log_dir, device=device)
  except BaseException:
    env.close()
    raise


def assert_equal(actual, expected, path="state"):
  if isinstance(actual, torch.Tensor):
    assert torch.equal(actual, expected), path
  elif isinstance(actual, np.ndarray):
    np.testing.assert_array_equal(actual, expected, err_msg=path)
  elif isinstance(actual, dict):
    assert actual.keys() == expected.keys(), path
    for key in actual:
      assert_equal(actual[key], expected[key], f"{path}.{key}")
  elif isinstance(actual, (list, tuple)):
    assert len(actual) == len(expected), path
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
      assert_equal(a, e, f"{path}[{i}]")
  else:
    assert actual == expected, path


# MuJoCo's integration carry, including delayed/history and sleeping-tree state.
# Solver/contact scratch outside active counts is uninitialized by the engine and
# cannot be compared across independent allocations. It is still saved/restored;
# immediate restoration below checks every allocation, including those tails.
_INTEGRATION_ARRAYS = {
  "time",
  "qpos",
  "qvel",
  "act",
  "qacc_warmstart",
  "ctrl",
  "qfrc_applied",
  "xfrc_applied",
  "eq_active",
  "mocap_pos",
  "mocap_quat",
  "userdata",
  "plugin_state",
  "history",
  "tree_asleep",
  "tree_awake",
  "body_awake",
}


def continuation_state(checkpoint):
  checkpoint["training_state"].pop("snapshot_id")
  for rank in checkpoint["training_state"]["ranks"].values():
    rank["logger"].pop("tot_time")
    rank["simulation"]["arrays"] = {
      key: value
      for key, value in rank["simulation"]["arrays"].items()
      if key.startswith("model.") or key.removeprefix("data.") in _INTEGRATION_ARRAYS
    }
  return checkpoint


@pytest.mark.parametrize("recurrent", [False, True])
def test_native_updates_resume_without_reset_or_observation_advance(
  tmp_path, recurrent
):
  torch.set_num_threads(1)
  baseline = make_runner(recurrent=recurrent)
  try:
    baseline.learn(4, init_at_random_ep_len=True)
    baseline.save(str(tmp_path / "baseline.pt"))
  finally:
    baseline.env.close()
  split = make_runner(recurrent=recurrent)
  try:
    split.learn(2, init_at_random_ep_len=True)
    split.save(str(tmp_path / "split.pt"))
  finally:
    split.env.close()
  resumed = make_runner(recurrent=recurrent)
  try:
    resumed.load(str(tmp_path / "split.pt"))
    source_state = torch.load(tmp_path / "split.pt", weights_only=False)[
      "training_state"
    ]["ranks"][0]
    assert_equal(resumed._capture_rank_state(), source_state)
    resumed.learn(2, init_at_random_ep_len=True)
    resumed.save(str(tmp_path / "resumed.pt"))
  finally:
    resumed.env.close()
  expected = torch.load(tmp_path / "baseline.pt", weights_only=False)
  actual = torch.load(tmp_path / "resumed.pt", weights_only=False)
  assert_equal(continuation_state(actual), continuation_state(expected))
  assert actual["training_state"]["next_iteration"] == 4
  assert actual["iter"] == 3


def test_legacy_checkpoint_rejected_without_mutation(tmp_path):
  runner = make_runner()
  try:
    legacy = runner.alg.save()
    legacy.update(iter=100, infos=None)
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)
    before = runner.env.unwrapped.common_step_counter
    with pytest.raises(ValueError, match="lacks a supported full training state"):
      runner.load(str(path))
    assert runner.env.unwrapped.common_step_counter == before
  finally:
    runner.env.close()


def test_stateful_resume_rejects_custom_untracked_state():
  runner = make_runner()
  try:

    class Unsupported:
      pass

    runner.env.unwrapped.__dict__["custom_memory"] = Unsupported()
    with pytest.raises(ValueError, match="training_state_dict"):
      ComponentState(runner.env.unwrapped).capture()
  finally:
    runner.env.close()


def test_stateful_resume_rejects_invalid_rank_before_native_collectives(monkeypatch):
  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("RANK", "2")
  cfg = asdict(RslRlOnPolicyRunnerCfg(stateful_resume=True))
  with pytest.raises(ValueError, match="Invalid distributed"):
    MjlabOnPolicyRunner(MagicMock(), cfg)


def test_custom_runner_rejected_before_learner_allocation(monkeypatch):
  class CustomRunner(MjlabOnPolicyRunner):
    pass

  allocate = MagicMock()
  monkeypatch.setattr("rsl_rl.runners.OnPolicyRunner.__init__", allocate)
  cfg = asdict(RslRlOnPolicyRunnerCfg(stateful_resume=True))
  with pytest.raises(ValueError, match="custom runner subclasses are unsupported"):
    CustomRunner(MagicMock(), cfg)
  allocate.assert_not_called()


def test_stateful_map_location_must_match_before_read_or_mutation(tmp_path):
  runner = make_runner()
  try:
    before = {
      key: value.clone() for key, value in runner.alg.actor.state_dict().items()
    }
    with pytest.raises(ValueError, match="map_location must match the learner device"):
      runner.load(str(tmp_path / "missing.pt"), map_location="cuda:0")
    for key, value in runner.alg.actor.state_dict().items():
      assert torch.equal(value, before[key])
  finally:
    runner.env.close()


def test_native_go1_motor_commands_and_raycast_state_resume(tmp_path):
  """Keep the stock flat task's commands, DR, rewards, motor and sensing paths."""
  torch.set_num_threads(1)

  def make_go1():
    cfg = unitree_go1_flat_env_cfg()
    cfg.seed = 19
    cfg.scene.num_envs = 2
    agent = unitree_go1_ppo_runner_cfg()
    agent.stateful_resume = True
    agent.num_steps_per_env = 2
    agent.actor.hidden_dims = agent.critic.hidden_dims = (8,)
    agent.algorithm.num_learning_epochs = agent.algorithm.num_mini_batches = 1
    agent.upload_model = False
    agent.logger = "tensorboard"
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cpu"))
    try:
      return MjlabOnPolicyRunner(env, asdict(agent), device="cpu")
    except BaseException:
      env.close()
      raise

  for phase, updates in (("baseline", 2), ("split", 1), ("resumed", 1)):
    runner = make_go1()
    try:
      assert runner.env.unwrapped.scene.sensor_context is not None
      assert not runner.env.unwrapped.scene.sensor_context.has_cameras
      if phase == "resumed":
        runner.load(str(tmp_path / "split.pt"))
        source = torch.load(tmp_path / "split.pt", weights_only=False)[
          "training_state"
        ]["ranks"][0]
        assert_equal(runner._capture_rank_state(), source)
      runner.learn(updates)
      assert runner.alg.optimizer.state
      runner.save(str(tmp_path / f"{phase}.pt"))
    finally:
      runner.env.close()
  expected = torch.load(tmp_path / "baseline.pt", weights_only=False)
  actual = torch.load(tmp_path / "resumed.pt", weights_only=False)
  assert_equal(continuation_state(actual), continuation_state(expected))
