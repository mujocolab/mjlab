"""Tests for the single-accumulator BaseViewer in base_alt.py."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from mjlab.viewer.base import BaseViewer


class FakeViewer(BaseViewer):
  """Minimal concrete viewer for testing."""

  def __init__(self, step_dt: float = 0.01, frame_rate: float = 60.0):
    env = MagicMock()
    env.unwrapped.step_dt = step_dt
    env.cfg.viewer = MagicMock()
    super().__init__(env, MagicMock(return_value=MagicMock()), frame_rate=frame_rate)
    self.sim_step_count = 0
    self.render_count = 0
    self._last_tick_time = time.perf_counter()

  def setup(self) -> None: ...

  def sync_env_to_viewer(self) -> None:
    self.render_count += 1

  def sync_viewer_to_env(self) -> None: ...
  def close(self) -> None: ...
  def is_running(self) -> bool:
    return True

  def _execute_step(self) -> bool:
    self.sim_step_count += 1
    self._step_count += 1
    self._stats_steps += 1
    return True

  def inject_tick(self, dt: float) -> bool:
    """Call tick() with a controlled dt."""
    now = time.perf_counter()
    self._last_tick_time = now - dt
    return self.tick()


# Physics stepping.


def test_stepping():
  """Physics steps match sim-time budget."""
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(dt=0.01)
  assert v.sim_step_count == 1

  v.inject_tick(dt=0.03)
  assert v.sim_step_count == 4

  v.inject_tick(dt=0.0)
  assert v.sim_step_count == 4


def test_accumulator_carries():
  """Fractional budget carries across ticks."""
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(dt=0.01)
  assert v.sim_step_count == 1

  v.inject_tick(dt=0.015)
  assert v.sim_step_count == 2  # 0.015 budget, 1 step, 0.005 carry

  v.inject_tick(dt=0.015)
  assert v.sim_step_count == 4  # 0.005+0.015=0.02, 2 steps


def test_high_speed():
  v = FakeViewer(step_dt=0.01, frame_rate=60.0)
  v._speed_index = v.SPEED_MULTIPLIERS.index(8.0)
  v._time_multiplier = 8.0
  v.inject_tick(dt=1.0 / 60.0)
  assert v.sim_step_count == 13


def test_slow_speed():
  v = FakeViewer(step_dt=0.01, frame_rate=60.0)
  v._speed_index = 0
  v._time_multiplier = 1 / 32
  for _ in range(19):
    v.inject_tick(dt=1.0 / 60.0)
  assert v.sim_step_count == 0
  for _ in range(3):
    v.inject_tick(dt=1.0 / 60.0)
  assert v.sim_step_count >= 1


# Render timing.


def test_render_at_frame_rate():
  v = FakeViewer(step_dt=0.01, frame_rate=60.0)
  assert v.inject_tick(dt=0.001) is True  # First tick renders.
  assert v.inject_tick(dt=0.001) is False  # Too soon.
  assert v.inject_tick(dt=1.0 / 60.0) is True  # Frame time elapsed.


# Pause and resume.


def test_pause_stops_physics():
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(dt=0.02)
  before = v.sim_step_count

  v.pause()
  v.inject_tick(dt=0.5)
  assert v.sim_step_count == before


def test_resume_no_burst():
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(dt=0.03)
  count_before = v.sim_step_count

  v.pause()
  v._last_error = "some error"

  v.resume()
  assert v._last_error is None
  assert v._sim_budget == 0.0

  v.inject_tick(dt=0.01)
  assert v.sim_step_count - count_before == 1


# Single step.


def test_single_step_while_paused():
  v = FakeViewer(step_dt=0.01)
  v.pause()

  v.request_single_step()
  v.inject_tick(dt=0.0)

  assert v.sim_step_count == 1
  assert v._is_paused


def test_single_step_ignored_when_running():
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(dt=0.02)
  before = v.sim_step_count

  v.request_single_step()
  v.inject_tick(dt=0.02)
  assert v.sim_step_count > before


# Error recovery.


def test_error_pauses():
  v = FakeViewer(step_dt=0.01)
  v._execute_step = BaseViewer._execute_step.__get__(v, FakeViewer)  # type: ignore[attr-defined]
  v.policy = MagicMock(side_effect=RuntimeError("test error"))

  v.inject_tick(dt=0.02)

  assert v._is_paused
  assert v._last_error is not None
  assert "test error" in v._last_error


def test_reset_clears():
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(dt=0.02)
  v._last_error = "err"

  v.reset_environment()
  assert v._step_count == 0
  assert v._sim_budget == 0.0
  assert v._last_error is None


def test_reset_calls_policy_reset():
  """reset_environment() calls policy.reset() if available."""
  v = FakeViewer(step_dt=0.01)
  v.policy = MagicMock()
  v.policy.reset = MagicMock()

  v.reset_environment()
  v.policy.reset.assert_called_once()


def test_reset_without_policy_reset():
  """reset_environment() works fine when policy has no reset method."""
  v = FakeViewer(step_dt=0.01)
  v.policy = lambda obs: obs  # plain callable, no reset attribute

  v.reset_environment()  # should not raise


# Formatting and status.


def test_format_speed():
  assert BaseViewer._format_speed(1.0) == "1x"
  assert BaseViewer._format_speed(0.5) == "1/2x"
  assert BaseViewer._format_speed(0.25) == "1/4x"
  assert BaseViewer._format_speed(1 / 32) == "1/32x"


def test_status_snapshot():
  v = FakeViewer(step_dt=0.01)
  v._fps = 60.0
  v._sps = 50.0
  v._last_error = "err"

  status = v.get_status()
  assert abs(status.actual_realtime - 0.5) < 1e-10
  assert status.speed_label == "1x"
  assert status.last_error == "err"


def test_capped_clears_each_tick():
  """Capped resets to False each tick; only True when deadline hit."""
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(dt=0.02)
  assert not v._was_capped

  # Force capped state, then verify next tick clears it.
  v._was_capped = True
  v.inject_tick(dt=0.02)
  assert not v._was_capped


def test_capped_false_when_no_remaining_budget():
  """Deadline hit with no remaining budget is NOT capped (transient stall)."""
  v = FakeViewer(step_dt=0.01, frame_rate=60.0)
  # dt=0.01 at 1x: budget=0.01, exactly 1 step, budget goes to 0.
  # Even if deadline fires (via GC), no remaining work to drop.
  v.inject_tick(dt=0.01)
  assert not v._was_capped


# Spiral protection.


def test_no_spiral():
  v = FakeViewer(step_dt=0.01, frame_rate=60.0)
  v._speed_index = v.SPEED_MULTIPLIERS.index(8.0)
  v._time_multiplier = 8.0

  original = v._execute_step

  def slow_step() -> bool:
    time.sleep(0.005)
    return original()

  v._execute_step = slow_step  # type: ignore[assignment]

  start = time.perf_counter()
  v.inject_tick(dt=1.0 / 60.0)
  elapsed = time.perf_counter() - start

  assert elapsed < 0.05
  assert v.sim_step_count < 13
  assert v.sim_step_count >= 1
  # Capped because slow steps + large budget means remaining work was dropped.
  assert v._was_capped
