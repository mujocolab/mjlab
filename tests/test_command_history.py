"""Tests for the opt-in per-episode command history on CommandTerm."""

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import (
  CommandHistory,
  CommandTerm,
  CommandTermCfg,
)
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand


@pytest.fixture(scope="module")
def device():
  return get_test_device()


# Unit tests for the container.


def make_history(num_envs=2, capacity=4, command_dim=2) -> CommandHistory:
  return CommandHistory(num_envs, capacity, command_dim, device="cpu")


def test_flush_appends_only_envs_marked_pending():
  history = make_history(num_envs=3)
  commands = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  times = torch.tensor([0.5, 1.5, 2.5])

  history.mark_pending(torch.tensor([0, 2]))
  history.flush(commands, times)

  assert history.lengths.tolist() == [1, 0, 1]
  assert history.commands[0, 0].tolist() == [1.0, 2.0]
  assert history.commands[2, 0].tolist() == [5.0, 6.0]
  assert torch.count_nonzero(history.commands[1]) == 0
  assert history.start_times[:, 0].tolist() == [0.5, 0.0, 2.5]

  # Flushing again without marking anything leaves the record untouched.
  history.flush(commands * 2.0, times + 1.0)
  assert history.lengths.tolist() == [1, 0, 1]
  assert history.commands[0, 0].tolist() == [1.0, 2.0]


def test_repeated_marks_collapse_into_one_segment():
  """Only the settled value is recorded, however often it was resampled."""
  history = make_history(num_envs=1)
  env_ids = torch.tensor([0])

  history.mark_pending(env_ids)
  history.mark_pending(env_ids)
  history.flush(torch.tensor([[7.0, 8.0]]), torch.tensor([1.0]))

  assert history.lengths.tolist() == [1]
  assert history.commands[0, 0].tolist() == [7.0, 8.0]


def test_clear_drops_only_given_envs_and_cancels_pending():
  history = make_history(num_envs=2)
  history.mark_pending(torch.arange(2))
  history.flush(torch.ones(2, 2), torch.ones(2))
  history.mark_pending(torch.arange(2))

  history.clear(torch.tensor([0]))
  history.flush(torch.full((2, 2), 9.0), torch.full((2,), 2.0))

  # Env 0 was cleared and its pending append cancelled; env 1 kept both.
  assert history.lengths.tolist() == [0, 2]
  assert torch.count_nonzero(history.commands[0]) == 0
  assert history.commands[1, 1].tolist() == [9.0, 9.0]


def test_durations_close_the_last_segment_at_end_time():
  history = make_history(num_envs=2, capacity=4, command_dim=1)
  for time in (0.0, 1.0, 3.0):
    history.mark_pending(torch.arange(2))
    history.flush(torch.zeros(2, 1), torch.full((2,), time))

  # A scalar end time applies to every env; slots past lengths stay zero.
  assert history.durations(4.0).tolist() == [[1.0, 2.0, 1.0, 0.0]] * 2

  # A per-env end time closes each env's own last segment.
  assert history.durations(torch.tensor([4.0, 5.0])).tolist() == [
    [1.0, 2.0, 1.0, 0.0],
    [1.0, 2.0, 2.0, 0.0],
  ]


def test_flush_saturates_at_capacity():
  """Overflow overwrites the newest slot rather than raising."""
  history = make_history(num_envs=1, capacity=2, command_dim=1)
  for value in (1.0, 2.0, 3.0):
    history.mark_pending(torch.tensor([0]))
    history.flush(torch.tensor([[value]]), torch.tensor([value]))

  assert history.lengths.tolist() == [2]
  assert history.commands[0, :, 0].tolist() == [1.0, 3.0]


# Integration tests against a real environment.


class RampCommand(CommandTerm):
  """Every resample bumps a counter, so segments are distinguishable."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.value = torch.zeros(self.num_envs, 1, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.value

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.value[env_ids] += 1.0

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids


@dataclass(kw_only=True)
class RampCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (0.1, 0.1)
  track_command_history: bool = True

  def build(self, env) -> RampCommand:
    return RampCommand(self, env)


def make_env(device, *, episode_length_s=0.4, curriculum=None, **term_kwargs):
  """A cartpole env driven by a RampCommand. step_dt is 0.05 s."""
  cfg = cartpole_balance_env_cfg()
  cfg.scene.num_envs = 3
  cfg.episode_length_s = episode_length_s
  cfg.commands = {"ramp": RampCommandCfg(**term_kwargs)}
  if curriculum is not None:
    cfg.curriculum = {"probe": curriculum}
  return ManagerBasedRlEnv(cfg=cfg, device=device)


@pytest.fixture
def ramp_env(device):
  env = make_env(device)
  yield env
  env.close()


def ramp_term(env) -> RampCommand:
  term = env.command_manager.get_term("ramp")
  assert isinstance(term, RampCommand)
  return term


def step(env, count=1):
  action = torch.zeros((env.num_envs, 1), device=env.device)
  for _ in range(count):
    env.step(action)


def test_history_is_none_when_tracking_is_disabled(device):
  env = make_env(device, track_command_history=False)
  assert ramp_term(env).command_history is None
  env.close()


def test_episode_is_recorded_from_reset_through_resamples(ramp_env):
  env = ramp_env
  history = ramp_term(env).command_history
  assert history is not None

  # ceil(0.4 / 0.1) + 1 slots, for a 0.4 s episode resampled every 0.1 s.
  assert history.capacity == 5

  env.reset()
  # The reset draw lands at episode time zero.
  assert history.lengths.tolist() == [1, 1, 1]
  assert history.start_times[0, 0].item() == 0.0
  assert history.commands[0, 0, 0].item() == 1.0

  # A resample every 0.1 s, i.e. every other 0.05 s step.
  step(env, 6)
  assert history.lengths.tolist() == [4, 4, 4]
  assert history.start_times[0, :4].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
  assert history.commands[0, :4, 0].tolist() == [1.0, 2.0, 3.0, 4.0]

  # Held to the end of the episode, the segments cover it exactly.
  assert history.durations(0.4)[0].sum().item() == pytest.approx(0.4)


def test_partial_reset_clears_only_that_env(ramp_env):
  env = ramp_env
  history = ramp_term(env).command_history
  assert history is not None

  env.reset()
  step(env, 4)
  assert history.lengths.tolist() == [3, 3, 3]

  env.reset(env_ids=torch.tensor([1], dtype=torch.int64, device=env.device))
  assert history.lengths.tolist() == [3, 1, 3]
  assert history.start_times[1, 0].item() == 0.0
  assert torch.count_nonzero(history.start_times[1, 1:]) == 0


def test_curriculum_sees_the_finished_episode(device):
  """The curriculum manager resets before the command manager clears."""
  seen = {}

  def probe(env, env_ids):
    history = ramp_term(env).command_history
    assert history is not None
    seen["lengths"] = history.lengths.clone()
    seen["durations"] = history.durations(env.max_episode_length_s).clone()
    return torch.tensor(0.0)

  env = make_env(device, curriculum=CurriculumTermCfg(func=probe))
  env.reset()
  seen.clear()

  # 0.4 s at 0.05 s per step: the episode times out on step 8 and auto-resets.
  step(env, 8)

  # Every segment of the episode that just ended is still visible.
  assert seen["lengths"].tolist() == [4, 4, 4]
  assert seen["durations"][0].sum().item() == pytest.approx(0.4)
  env.close()


# MotionCommand, which resamples outside CommandTerm._resample.


def test_motion_command_records_wraparound_resample():
  cmd = Mock()
  cmd.time_steps = torch.tensor([2, 9], dtype=torch.long)
  cmd.motion = Mock()
  cmd.motion.time_step_total = 10
  cmd.cfg = Mock()
  cmd.cfg.sampling_mode = "uniform"
  cmd._pending_forward = False
  cmd._resample_command = Mock()
  history = CommandHistory(num_envs=2, capacity=4, command_dim=1, device="cpu")
  cmd._command_history = history
  # Bind the real hook, so the stub drives production code.
  cmd._record_command_resample = lambda ids: CommandTerm._record_command_resample(
    cmd, ids
  )

  MotionCommand._update_command(cmd, env_ids=None)

  # Env 1 wrapped past the end of its clip and must be recorded; env 0 did not.
  history.flush(torch.tensor([[0.0], [7.0]]), torch.tensor([0.5, 0.5]))
  assert history.lengths.tolist() == [0, 1]
  assert history.commands[1, 0, 0].item() == 7.0


def test_motion_command_capacity_budgets_wraparound():
  # An uninitialized real instance, so the override's zero-arg super() works.
  cmd = object.__new__(MotionCommand)
  cmd._env = Mock()
  cmd._env.max_episode_length_s = 20.0
  cmd._env.step_dt = 0.02
  cmd.cfg = Mock()
  cmd.cfg.resampling_time_range = (5.0, 8.0)
  cmd.motion = Mock()
  cmd.motion.time_step_total = 100  # A 2.0 s clip.

  # ceil(20 / 5) + 1 timer segments, plus ceil(20 / 2) wraparounds, plus one
  # for an env that starts near the end of its clip.
  assert CommandTerm._history_capacity(cmd) == 5
  assert cmd._history_capacity() == 16


def test_capacity_is_capped_without_a_real_horizon():
  """episode_length_s defaults to 0.0 and play configs set it to 1e10."""
  cmd = Mock()
  cmd._env = Mock()
  cmd._env.step_dt = 0.02
  cmd.cfg = Mock()
  cmd.cfg.resampling_time_range = (5.0, 8.0)

  for horizon in (0.0, 1e10):
    cmd._env.max_episode_length_s = horizon
    assert CommandTerm._segments_per_episode(cmd, 5.0) == 256
