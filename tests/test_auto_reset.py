"""Tests for the auto_reset config flag."""

import pytest
import torch
from conftest import get_test_device
from test_command_manager import CounterCommand, CounterCommandCfg

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _make_cfg(auto_reset: bool):
  cfg = cartpole_balance_env_cfg()
  cfg.episode_length_s = 0.5  # 10 steps at dt=0.05
  cfg.scene.num_envs = 4
  cfg.auto_reset = auto_reset
  return cfg


def _step_until_done_env(env):
  """Step with zero actions until at least one env is done. Return step outputs."""
  for _ in range(env.max_episode_length + 5):
    action = torch.zeros((env.num_envs, 1), device=env.device)
    result = env.step(action)
    terminated, truncated = result[2], result[3]
    if (terminated | truncated).any():
      return result
  pytest.fail("No env terminated within max_episode_length steps")


def test_auto_reset_true_resets_done_envs(device):
  """With auto_reset=True (default), done envs are reset during step."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=True), device=device)
  env.reset()
  _, _, terminated, truncated, _ = _step_until_done_env(env)
  done = terminated | truncated
  done_ids = done.nonzero(as_tuple=False).squeeze(-1)

  # Episode counter was reset to 0 for done envs.
  assert (env.episode_length_buf[done_ids] == 0).all()
  env.close()


def test_auto_reset_false_preserves_terminal_state(device):
  """With auto_reset=False, done envs are NOT reset and obs is the terminal state."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=False), device=device)
  env.reset()
  obs, _, terminated, truncated, _ = _step_until_done_env(env)
  done = terminated | truncated
  done_ids = done.nonzero(as_tuple=False).squeeze(-1)

  # Episode counter was NOT reset (still at max_episode_length).
  assert (env.episode_length_buf[done_ids] == env.max_episode_length).all()

  # The returned obs must reflect the current (post-decimation terminal) sim
  # state. Since no reset ran and the sim wasn't touched after step(), a fresh
  # observation_manager.compute() on the current sim state must match exactly.
  # This catches regressions where step() might return stale or post-reset obs.
  env.observation_manager._obs_buffer = None  # bypass cache
  fresh_obs = env.observation_manager.compute()
  for group in obs:
    returned = obs[group]
    current = fresh_obs[group]
    assert isinstance(returned, torch.Tensor) and isinstance(current, torch.Tensor)
    assert torch.equal(returned, current)
  env.close()


def test_auto_reset_false_explicit_reset_works(device):
  """After auto_reset=False, calling reset(env_ids=...) resets those envs."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=False), device=device)
  env.reset()
  _, _, terminated, truncated, _ = _step_until_done_env(env)
  done = terminated | truncated
  done_ids = done.nonzero(as_tuple=False).squeeze(-1)

  # Manually reset done envs.
  env.reset(env_ids=done_ids)
  assert (env.episode_length_buf[done_ids] == 0).all()

  # Can continue stepping after manual reset.
  action = torch.zeros((env.num_envs, 1), device=env.device)
  obs, reward, _, _, _ = env.step(action)
  assert obs is not None
  assert reward is not None
  env.close()


def test_auto_reset_false_requires_manual_reset_before_next_step(device):
  """Raw env should reject another step until done envs are explicitly reset."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=False), device=device)
  env.reset()
  _step_until_done_env(env)

  action = torch.zeros((env.num_envs, 1), device=env.device)
  with pytest.raises(RuntimeError, match="must be reset via reset"):
    env.step(action)

  env.close()


def _slice_obs(obs: dict, ids: torch.Tensor) -> dict[str, torch.Tensor]:
  """Return a new obs dict containing only the rows at ``ids`` (per group)."""
  return {k: v[ids] for k, v in obs.items() if isinstance(v, torch.Tensor)}


def test_auto_reset_false_user_loop_pattern(device):
  """Example: run your own training loop against an auto_reset=False env.

  The pattern is:
    1. After step(), derive done_ids from terminated | truncated.
    2. Slice obs[done_ids] to get the true terminal observation and use it for
      bootstrap / target computation.
    3. Call env.reset(env_ids=done_ids) to reset only the done envs.
    4. Continue stepping with the full batch.
  """
  env = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=False), device=device)
  obs, _ = env.reset(seed=0)
  episode_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  last_terminal_obs: dict[str, torch.Tensor] | None = None

  action = torch.zeros((env.num_envs, 1), device=env.device)
  for _ in range((env.max_episode_length + 2) * 3):
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated | truncated
    if not done.any():
      continue

    done_ids = done.nonzero(as_tuple=False).squeeze(-1)
    last_terminal_obs = _slice_obs(obs, done_ids)  # feed this to your critic/replay

    episode_count[done_ids] += 1
    obs, _ = env.reset(env_ids=done_ids)
    if (episode_count >= 2).all():
      break

  assert (episode_count >= 2).all()
  assert last_terminal_obs is not None
  env.close()


def test_auto_reset_false_obs_differs_from_auto_reset_true(device):
  """Terminal obs (auto_reset=False) differs from post-reset obs (auto_reset=True)."""
  # Run with auto_reset=True, capture post-reset obs for done envs.
  env_on = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=True), device=device)
  env_on.reset(seed=42)
  obs_on, _, _, _, _ = _step_until_done_env(env_on)
  env_on.close()

  # Run with auto_reset=False with the same seed, capture terminal obs.
  env_off = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=False), device=device)
  env_off.reset(seed=42)
  obs_off, _, _, _, _ = _step_until_done_env(env_off)
  env_off.close()

  # The observations should differ: one is post-reset, the other is terminal.
  for group in obs_on:
    on_val = obs_on[group]
    off_val = obs_off[group]
    assert isinstance(on_val, torch.Tensor) and isinstance(off_val, torch.Tensor)
    assert not torch.equal(on_val, off_val)


def test_partial_reset_leaves_other_envs_obs_buffers_untouched(device):
  """reset(env_ids=...) must not advance other envs' history/delay buffers."""
  cfg = _make_cfg(auto_reset=False)
  cfg.observations["actor"].terms["cart_pos"].history_length = 4
  cfg.observations["actor"].terms["cart_vel"].delay_min_lag = 2
  cfg.observations["actor"].terms["cart_vel"].delay_max_lag = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  action = torch.zeros((env.num_envs, 1), device=env.device)
  for _ in range(3):
    env.step(action)

  om = env.observation_manager
  hist = om._group_obs_term_history_buffer["actor"]["cart_pos"]
  delay = om._group_obs_term_delay_buffer["actor"]["cart_vel"]
  h_before = hist.buffer[0].clone()
  d_before = delay.peek()[0].clone()

  env.reset(env_ids=torch.tensor([1], dtype=torch.int64, device=env.device))

  # Env 0 was not reset: history window and delayed obs are untouched.
  assert torch.equal(hist.buffer[0], h_before)
  assert torch.equal(delay.peek()[0], d_before)

  # Env 1 was reset: history is backfilled with its single post-reset frame.
  h1 = hist.buffer[1]
  assert torch.all(h1 == h1[0])
  assert hist.current_length[1].item() == 1
  env.close()


# Section: parity between auto-reset and the explicit reset() flow.

_COMMAND_T = 7.0
_INTERVAL_T = 5.0
_MARKER_VEL = 37.0


def _noop_event(env, env_ids) -> None:
  del env, env_ids


def _write_marker_velocity(env, env_ids) -> None:
  """Overwrite joint velocities with a recognizable marker value."""
  asset = env.scene["cartpole"]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  joint_pos = asset.data.joint_pos[env_ids]
  joint_vel = torch.full_like(asset.data.joint_vel[env_ids], _MARKER_VEL)
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def _make_parity_cfg(auto_reset: bool):
  """Cartpole cfg with fixed-interval command and event timers.

  Fixed ranges make timer values deterministic, so assertions are exact and
  independent of RNG state and physics nondeterminism.
  """
  cfg = _make_cfg(auto_reset)
  cfg.commands = {
    "counter": CounterCommandCfg(resampling_time_range=(_COMMAND_T, _COMMAND_T))
  }
  cfg.events["probe"] = EventTermCfg(
    func=_noop_event, mode="interval", interval_range_s=(_INTERVAL_T, _INTERVAL_T)
  )
  return cfg


def _stagger_env0(env, steps_until_reset: int) -> torch.Tensor:
  """Advance env 0's episode clock so it times out after the given steps."""
  env.reset()
  env.episode_length_buf[0] = env.max_episode_length - steps_until_reset
  return torch.zeros((env.num_envs, 1), device=env.device)


def test_auto_reset_preserves_fresh_command_timer(device):
  """A command timer resampled by an auto-reset is not decremented that step."""
  env = ManagerBasedRlEnv(cfg=_make_parity_cfg(auto_reset=True), device=device)
  action = _stagger_env0(env, steps_until_reset=2)
  env.step(action)
  env.step(action)  # Env 0 times out and auto-resets here.
  assert env.episode_length_buf[0].item() == 0

  term = env.command_manager.get_term("counter")
  assert isinstance(term, CounterCommand)
  time_left = term.time_left
  expected_running = _COMMAND_T - 2 * env.step_dt
  assert torch.allclose(time_left[0], torch.tensor(_COMMAND_T, device=env.device))
  assert torch.allclose(
    time_left[1:], torch.tensor(expected_running, device=env.device)
  )
  env.close()


def test_auto_reset_preserves_fresh_interval_event_timer(device):
  """An interval event timer resampled by an auto-reset is not decremented."""
  env = ManagerBasedRlEnv(cfg=_make_parity_cfg(auto_reset=True), device=device)
  action = _stagger_env0(env, steps_until_reset=2)
  env.step(action)
  env.step(action)  # Env 0 times out and auto-resets here.
  assert env.episode_length_buf[0].item() == 0

  timer = env.event_manager._interval_term_time_left[0]
  expected_running = _INTERVAL_T - 2 * env.step_dt
  assert torch.allclose(timer[0], torch.tensor(_INTERVAL_T, device=env.device))
  assert torch.allclose(timer[1:], torch.tensor(expected_running, device=env.device))
  env.close()


def test_interval_event_acts_on_pre_reset_state(device):
  """An interval event firing on a reset step hits the terminal state, so the
  freshly reset env comes out clean, as in the explicit reset flow."""
  cfg = _make_cfg(auto_reset=True)
  cfg.events["kick"] = EventTermCfg(
    func=_write_marker_velocity, mode="interval", interval_range_s=(0.0, 0.0)
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  action = _stagger_env0(env, steps_until_reset=1)
  env.step(action)  # Kick fires for all envs; env 0 then auto-resets.
  assert env.episode_length_buf[0].item() == 0

  joint_vel = env.scene["cartpole"].data.joint_vel
  # Env 0 was reset after the kick: its velocity is the reset distribution's,
  # not the marker. Env 1 was not reset and still carries the marker.
  assert joint_vel[0].abs().max().item() < 1.0
  assert torch.allclose(joint_vel[1], torch.full_like(joint_vel[1], _MARKER_VEL))
  env.close()


def test_auto_reset_matches_manual_reset_timers(device):
  """After a reset, both flows leave identical command/event timer state."""
  auto_env = ManagerBasedRlEnv(cfg=_make_parity_cfg(auto_reset=True), device=device)
  manual_env = ManagerBasedRlEnv(cfg=_make_parity_cfg(auto_reset=False), device=device)
  auto_env.reset(seed=0)
  manual_env.reset(seed=0)

  action = torch.zeros((auto_env.num_envs, 1), device=auto_env.device)
  done = torch.zeros(auto_env.num_envs, dtype=torch.bool, device=auto_env.device)
  for _ in range(auto_env.max_episode_length):
    auto_env.step(action)
    _, _, terminated, truncated, _ = manual_env.step(action)
    done = terminated | truncated
  done_ids = done.nonzero(as_tuple=False).squeeze(-1)
  assert len(done_ids) == manual_env.num_envs  # Time-out is synchronized.
  manual_env.reset(env_ids=done_ids)

  for env in (auto_env, manual_env):
    term = env.command_manager.get_term("counter")
    assert isinstance(term, CounterCommand)
    interval_timer = env.event_manager._interval_term_time_left[0]
    assert torch.all(env.episode_length_buf == 0)
    assert torch.allclose(term.time_left, torch.full_like(term.time_left, _COMMAND_T))
    assert torch.allclose(interval_timer, torch.full_like(interval_timer, _INTERVAL_T))
    # Stateful command advance: resample zeroed the counter, then exactly one
    # post-reset update ran in both flows.
    assert torch.all(term.ticks == 1)

  auto_env.close()
  manual_env.close()
