"""Tests for capture_terminal_observations (true terminal obs under auto-reset).

See ManagerBasedRlEnvCfg.capture_terminal_observations. The feature stores the
real terminal observation (s_{t+1} before reset) for done environments in
extras["terminal"] while keeping auto-reset enabled.
"""

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _t(value) -> torch.Tensor:
  """Narrow an obs-group value to a Tensor (cartpole groups are concatenated)."""
  assert isinstance(value, torch.Tensor)
  return value


def _staggered_timeout(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Time out env i one step earlier than env i+1, producing partial resets.

  Depends only on episode_length_buf, the env index, and max_episode_length, so
  it is valid at the init-time shape-validation call (where the counter is 0 and
  this returns all-False).
  """
  idx = torch.arange(env.num_envs, device=env.device)
  return env.episode_length_buf + idx >= env.max_episode_length


def _make_cfg(
  *,
  auto_reset: bool = True,
  capture: bool = False,
  num_envs: int = 4,
  corruption: bool = False,
  history: int = 0,
  staggered: bool = False,
):
  cfg = cartpole_balance_env_cfg()
  cfg.episode_length_s = 0.5  # 10 steps at dt=0.05.
  cfg.scene.num_envs = num_envs
  cfg.auto_reset = auto_reset
  cfg.capture_terminal_observations = capture
  cfg.observations["actor"].enable_corruption = corruption
  if history > 0:
    for group in cfg.observations.values():
      group.history_length = history
  if staggered:
    cfg.terminations = {
      "staggered": TerminationTermCfg(func=_staggered_timeout, time_out=True)
    }
  return cfg


def _step_until_done(env):
  action = torch.zeros((env.num_envs, 1), device=env.device)
  for _ in range(env.max_episode_length + 5):
    result = env.step(action)
    terminated, truncated = result[2], result[3]
    if (terminated | truncated).any():
      return result
  pytest.fail("No env terminated within max_episode_length steps")


# Presence / contract.


def test_disabled_by_default_no_terminal_key(device):
  """Without the flag, extras never gains a 'terminal' key, even on reset steps."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(capture=False), device=device)
  env.reset()
  _, _, terminated, truncated, extras = _step_until_done(env)
  assert (terminated | truncated).any()  # we did hit a reset step
  assert "terminal" not in extras
  env.close()


def test_terminal_present_only_on_reset_steps(device):
  """extras['terminal'] appears exactly on steps where >=1 env resets."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(capture=True, staggered=True), device=device)
  env.reset()
  action = torch.zeros((env.num_envs, 1), device=env.device)
  saw_reset_step = saw_non_reset_step = False
  for _ in range(env.max_episode_length + 5):
    _, _, terminated, truncated, extras = env.step(action)
    done = terminated | truncated
    if done.any():
      saw_reset_step = True
      assert "terminal" in extras
    else:
      saw_non_reset_step = True
      assert "terminal" not in extras
  assert saw_reset_step and saw_non_reset_step
  env.close()


def test_terminal_env_ids_and_shapes(device):
  """env_ids match the done mask; observation rows mirror groups with [k] rows."""
  env = ManagerBasedRlEnv(
    cfg=_make_cfg(capture=True, history=3, staggered=True), device=device
  )
  env.reset()
  obs, _, terminated, truncated, extras = _step_until_done(env)
  done = terminated | truncated
  done_ids = done.nonzero(as_tuple=False).squeeze(-1)

  terminal = extras["terminal"]
  assert torch.equal(terminal["env_ids"], done_ids)

  num_done = int(done.sum().item())
  assert set(terminal["observations"].keys()) == set(obs.keys())
  for group, term_obs in terminal["observations"].items():
    returned = obs[group]
    assert isinstance(term_obs, torch.Tensor) and isinstance(returned, torch.Tensor)
    # Same trailing shape as the returned obs (e.g. history is flattened in).
    assert term_obs.shape == (num_done, *returned.shape[1:])
  env.close()


def test_terminal_obs_mirrors_non_concatenated_group(device):
  """A non-concatenated group yields a term dict (not a tensor) in extras.

  Exercises the dict branch of both the peek observation pass and the env-id
  selector, which the concatenated cartpole groups otherwise leave untested.
  """
  cfg = _make_cfg(capture=True, history=3, staggered=True)
  cfg.observations["critic"].concatenate_terms = False
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  obs, _, terminated, truncated, extras = _step_until_done(env)
  num_done = int((terminated | truncated).sum().item())

  term_obs = extras["terminal"]["observations"]
  # actor stays concatenated (a tensor); critic is now a dict of per-term rows.
  assert isinstance(term_obs["actor"], torch.Tensor)
  returned_critic = obs["critic"]
  assert isinstance(term_obs["critic"], dict) and isinstance(returned_critic, dict)
  assert set(term_obs["critic"].keys()) == set(returned_critic.keys())
  for name, term in term_obs["critic"].items():
    assert term.shape == (num_done, *returned_critic[name].shape[1:])
  env.close()


def test_no_effect_when_auto_reset_false(device):
  """The flag is a no-op under auto_reset=False (returned obs is already terminal)."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(auto_reset=False, capture=True), device=device)
  env.reset()
  _, _, terminated, truncated, extras = _step_until_done(env)
  assert (terminated | truncated).any()
  assert "terminal" not in extras
  env.close()


def test_terminal_differs_from_post_reset_obs(device):
  """Terminal obs (pre-reset) differs from the returned post-reset obs."""
  env = ManagerBasedRlEnv(cfg=_make_cfg(capture=True, staggered=True), device=device)
  env.reset(seed=0)
  obs, _, terminated, truncated, extras = _step_until_done(env)
  done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
  for group, terminal in extras["terminal"]["observations"].items():
    returned = _t(obs[group])[done_ids]
    # Post-reset obs come from a freshly randomized initial state, so they must
    # differ from the terminal state that triggered the reset.
    assert not torch.equal(_t(terminal), returned)
  env.close()


# Correctness: equivalence to auto_reset=False.


def _terminal_at_first_reset(cfg, seed, device):
  """Run a capture env alone to its first reset; return (ids, terminal obs dict).

  Run in isolation so the (global) RNG stream is identical to any other run up
  to the first reset: with corruption off, no RNG is consumed between the
  initial reset and the first episode boundary.
  """
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset(seed=seed)
  action = torch.zeros((env.num_envs, 1), device=env.device)
  for _ in range(env.max_episode_length + 5):
    _, _, terminated, truncated, extras = env.step(action)
    done = terminated | truncated
    if done.any():
      ids = done.nonzero(as_tuple=False).squeeze(-1)
      out = (
        ids.clone(),
        {g: _t(v).clone() for g, v in extras["terminal"]["observations"].items()},
      )
      env.close()
      return out
  env.close()
  pytest.fail("No env terminated")


def _returned_at_first_reset(cfg, seed, device):
  """Run an auto_reset=False env alone to its first reset; return (ids, obs[ids])."""
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset(seed=seed)
  action = torch.zeros((env.num_envs, 1), device=env.device)
  for _ in range(env.max_episode_length + 5):
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated | truncated
    if done.any():
      ids = done.nonzero(as_tuple=False).squeeze(-1)
      out = (ids.clone(), {g: _t(v)[ids].clone() for g, v in obs.items()})
      env.close()
      return out
  env.close()
  pytest.fail("No env terminated")


@pytest.mark.parametrize("history", [0, 2])
def test_terminal_matches_auto_reset_false(device, history):
  """The captured terminal obs equals what auto_reset=False would return.

  This is the gold-standard correctness check, including the history path. With
  observation noise disabled the whole pipeline is deterministic, so equivalence
  is exact. Compared at the first reset to keep the shared global RNG in sync.
  """
  seed = 123
  capture_ids, terminal = _terminal_at_first_reset(
    _make_cfg(auto_reset=True, capture=True, corruption=False, history=history),
    seed,
    device,
  )
  ref_ids, returned = _returned_at_first_reset(
    _make_cfg(auto_reset=False, corruption=False, history=history),
    seed,
    device,
  )

  assert torch.equal(capture_ids, ref_ids)
  assert set(terminal.keys()) == set(returned.keys())
  for group in returned:
    assert torch.allclose(terminal[group], returned[group], atol=1e-6), (
      f"terminal obs mismatch for group '{group}'"
    )


# Correctness: enabling the feature must not change the trajectory.


def test_capture_does_not_perturb_trajectory(device):
  """Returned obs and rewards are identical with the flag on vs off.

  Stresses every state channel the WIP corrupted: observation noise (global
  RNG), history buffers, and partial (staggered) resets. If peek drew RNG or
  mutated a buffer, the two runs would diverge.
  """
  steps = 30
  seed = 7

  def run(capture: bool):
    env = ManagerBasedRlEnv(
      cfg=_make_cfg(
        capture=capture,
        num_envs=4,
        corruption=True,
        history=3,
        staggered=True,
      ),
      device=device,
    )
    env.reset(seed=seed)
    action = torch.zeros((env.num_envs, 1), device=env.device)
    obs_log, rew_log = [], []
    for _ in range(steps):
      obs, rew, _, _, _ = env.step(action)
      obs_log.append({g: _t(v).clone() for g, v in obs.items()})
      rew_log.append(rew.clone())
    env.close()
    return obs_log, rew_log

  obs_off, rew_off = run(capture=False)
  obs_on, rew_on = run(capture=True)

  for t in range(steps):
    assert torch.equal(rew_off[t], rew_on[t]), f"reward diverged at step {t}"
    for group in obs_off[t]:
      assert torch.equal(obs_off[t][group], obs_on[t][group]), (
        f"obs group '{group}' diverged at step {t}"
      )


# Correctness: the safety invariant the extra forward() relies on.


@pytest.mark.slow
def test_forward_is_a_fixed_point_under_contact():
  """Guard the invariant that makes capture's extra forward() non-perturbing.

  When capturing, step() runs an extra forward()+sense() at the terminal state
  before the unconditional post-reset forward(). That is safe only because
  forward() is a fixed point: it copies qacc into qacc_warmstart, and re-solving
  from that warm start reproduces the same qacc, so the extra call cannot change
  the result of the following forward(). If a solver change ever under-converged
  this, enabling the flag would silently perturb training. Assert it directly on
  a contact-rich env, in dynamic high-force states.

  Pinned to CPU: warp's parallel contact kernels use nondeterministic reductions
  on GPU, so bit-identity of two forwards only holds on CPU (and on GPU the
  feature's effect is already within the sim's inherent nondeterminism).
  """
  device = "cpu"
  cfg = unitree_go1_flat_env_cfg()
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset(seed=1)
  gen = torch.Generator(device=device)
  gen.manual_seed(0)
  dim = env.action_manager.total_action_dim
  for _ in range(15):
    # Strong random actions drive dynamic, high-force ground contacts.
    action = (torch.rand((env.num_envs, dim), generator=gen, device=device) * 2 - 1) * 3
    env.step(action)

    env.sim.forward()
    env.sim.sense()
    sensordata = env.sim.data.sensordata.clone()
    qacc = env.sim.data.qacc.clone()

    env.sim.forward()
    env.sim.sense()
    assert torch.equal(sensordata, env.sim.data.sensordata), (
      "forward() moved sensordata"
    )
    assert torch.equal(qacc, env.sim.data.qacc), "forward() moved qacc"
  env.close()
