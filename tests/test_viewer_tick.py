"""Tests for the two-clock physics/render decoupling in BaseViewer.tick()."""

from __future__ import annotations

from unittest.mock import MagicMock

from mjlab.viewer.base import BaseViewer


class FakeViewer(BaseViewer):
  """Minimal concrete viewer with controllable timing."""

  def __init__(self, step_dt: float = 0.01, frame_rate: float = 60.0):
    env = MagicMock()
    env.unwrapped.step_dt = step_dt
    env.cfg.viewer = MagicMock()
    super().__init__(env, MagicMock(return_value=MagicMock()), frame_rate=frame_rate)
    self.sim_step_count = 0

  def setup(self) -> None: ...
  def sync_env_to_viewer(self) -> None: ...
  def sync_viewer_to_env(self) -> None: ...
  def close(self) -> None: ...
  def is_running(self) -> bool:
    return True

  def _execute_step(self) -> bool:
    self.sim_step_count += 1
    self._step_count += 1
    self._sps_accum_steps += 1
    return True

  def inject_tick(self, wall_dt: float) -> bool:
    """Call tick() with controlled wall_dt."""
    self._timer.tick = lambda: wall_dt  # type: ignore[assignment]
    return self.tick()


def test_tick_stepping():
  """Physics steps match sim-time budget: 1 step, 3 steps, then 0."""
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(wall_dt=0.01)
  assert v.sim_step_count == 1

  v.inject_tick(wall_dt=0.03)
  assert v.sim_step_count == 4

  v.inject_tick(wall_dt=0.0)
  assert v.sim_step_count == 4  # No budget → no steps


def test_budget_cap():
  """Large wall_dt is capped to max_steps_per_tick (spiral-of-death prevention)."""
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(wall_dt=1.0)
  assert v.sim_step_count == 10


def test_pause_and_resume():
  """Pausing stops physics; resuming resyncs clocks (no catch-up burst)."""
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(wall_dt=0.03)
  assert v.sim_step_count == 3

  v.pause()
  v.inject_tick(wall_dt=0.5)
  assert v.sim_step_count == 3  # No steps while paused

  v.resume()
  v.inject_tick(wall_dt=0.01)
  assert v.sim_step_count == 4  # Exactly 1, no burst


def test_full_wall_time_used_for_budget():
  """Full wall time feeds the sim budget (max_steps_per_tick caps spirals).

  With step_dt=0.01:
    tick 1: tracked=0.010, deficit=0.010 → 1 step,  actual=0.01
    tick 2: tracked=0.025, deficit=0.015 → 1 step,  actual=0.02  (5ms carry)
    tick 3: tracked=0.040, deficit=0.020 → 2 steps, actual=0.04  (carry consumed)
    tick 4: tracked=0.055, deficit=0.015 → 1 step,  actual=0.05  (5ms carry)

  Over 4 ticks (55ms wall), 50ms sim time → RTF=0.91x, approaching 1.0 as
  the half-step carry oscillation averages out.
  """
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(wall_dt=0.01)
  assert v.sim_step_count == 1

  v.inject_tick(wall_dt=0.015)
  assert v.sim_step_count == 2

  v.inject_tick(wall_dt=0.015)
  assert v.sim_step_count == 4  # 2 steps (carry from tick 2 consumed)

  v.inject_tick(wall_dt=0.015)
  assert v.sim_step_count == 5


def test_render_independent_of_physics():
  """Render timing follows frame_rate, not physics stepping."""
  v = FakeViewer(step_dt=0.01, frame_rate=60.0)

  assert v.inject_tick(wall_dt=0.001) is True  # First tick always renders
  assert v.inject_tick(wall_dt=0.001) is False  # Too soon
  assert v.inject_tick(wall_dt=1.0 / 60.0) is True  # Frame time elapsed


# --- New tests ---


def test_single_step_while_paused():
  """Single-step advances exactly one step and updates sim time."""
  v = FakeViewer(step_dt=0.01)
  v.pause()

  v.request_single_step()
  v.inject_tick(wall_dt=0.0)

  assert v.sim_step_count == 1
  assert v._is_paused  # Still paused after single step
  assert abs(v._actual_sim_time - 0.01) < 1e-10
  assert abs(v._tracked_sim_time - v._actual_sim_time) < 1e-10


def test_single_step_ignored_when_running():
  """Single-step does nothing when not paused."""
  v = FakeViewer(step_dt=0.01)
  # Advance a bit first
  v.inject_tick(wall_dt=0.01)
  assert v.sim_step_count == 1

  v.request_single_step()
  v.inject_tick(wall_dt=0.0)
  # Only the normal tick stepping should apply (0 wall_dt = 0 steps)
  assert v.sim_step_count == 1


def test_error_recovery_pauses():
  """Exception in _execute_step pauses viewer and stores error."""
  v = FakeViewer(step_dt=0.01)
  # Replace _execute_step with real base implementation to test error recovery.
  # Then make the policy raise.
  v._execute_step = BaseViewer._execute_step.__get__(v, FakeViewer)  # type: ignore[attr-defined]
  v.policy = MagicMock(side_effect=RuntimeError("test error"))

  v.inject_tick(wall_dt=0.01)

  assert v._is_paused
  assert v._last_error is not None
  assert "test error" in v._last_error
  assert abs(v._actual_sim_time) < 1e-10


def test_single_step_failure_does_not_advance_sim_time():
  """Single-step failure should not advance sim clocks."""
  v = FakeViewer(step_dt=0.01)
  v._execute_step = BaseViewer._execute_step.__get__(v, FakeViewer)  # type: ignore[attr-defined]
  v.policy = MagicMock(side_effect=RuntimeError("single-step error"))
  v.pause()

  v.request_single_step()
  v.inject_tick(wall_dt=0.0)

  assert abs(v._actual_sim_time) < 1e-10
  assert abs(v._tracked_sim_time) < 1e-10


def test_error_cleared_on_reset():
  """Resetting the environment clears the stored error."""
  v = FakeViewer(step_dt=0.01)
  v._last_error = "some error"

  v.reset_environment()

  assert v._last_error is None


def test_error_cleared_on_resume():
  """Resuming clears the stored error."""
  v = FakeViewer(step_dt=0.01)
  v.pause()
  v._last_error = "some error"

  v.resume()

  assert v._last_error is None


def test_speed_multipliers_power_of_two():
  """Speed multipliers are power-of-2 fractions from 1/32 to 8."""
  expected = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0]
  assert BaseViewer.SPEED_MULTIPLIERS == expected


def test_format_speed():
  """_format_speed produces human-readable fraction strings."""
  assert BaseViewer._format_speed(1.0) == "1x"
  assert BaseViewer._format_speed(0.5) == "1/2x"
  assert BaseViewer._format_speed(0.25) == "1/4x"
  assert BaseViewer._format_speed(1 / 32) == "1/32x"


def test_configurable_max_steps_per_tick():
  """_max_steps_per_tick controls the budget cap."""
  v = FakeViewer(step_dt=0.01)
  v._max_steps_per_tick = 5
  v.inject_tick(wall_dt=1.0)
  assert v.sim_step_count == 5


def test_status_snapshot():
  """Status snapshot exposes a consistent UI-facing state contract."""
  v = FakeViewer(step_dt=0.01)
  v._smoothed_fps = 60.0
  v._smoothed_sps = 50.0
  v._is_capped = True
  v._last_error = "err"

  status = v.get_status()

  assert status.paused is False
  assert status.step_count == v._step_count
  assert status.speed_multiplier == 1.0
  assert status.speed_label == "1x"
  assert abs(status.target_realtime - 1.0) < 1e-10
  assert abs(status.actual_realtime - 0.5) < 1e-10
  assert abs(status.smoothed_fps - 60.0) < 1e-10
  assert status.capped is True
  assert status.last_error == "err"
