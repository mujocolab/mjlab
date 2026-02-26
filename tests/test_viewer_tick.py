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

  def step_simulation(self) -> None:
    self.sim_step_count += 1
    self._step_count += 1
    self._sps_accum_steps += 1

  def inject_tick(self, wall_dt: float, step_wall_time: float | None = None) -> bool:
    """Call tick() with controlled wall_dt and optional step_wall_time."""
    self._timer.tick = lambda: wall_dt  # type: ignore[assignment]
    if step_wall_time is not None:
      self._step_wall_time = step_wall_time
    return self.tick()


def test_tick_stepping():
  """Physics steps match sim-time budget: 1 step, 3 steps, then 0."""
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(wall_dt=0.01, step_wall_time=0.0)
  assert v.sim_step_count == 1

  v.inject_tick(wall_dt=0.03, step_wall_time=0.0)
  assert v.sim_step_count == 4

  v.inject_tick(wall_dt=0.0, step_wall_time=0.0)
  assert v.sim_step_count == 4  # No budget → no steps


def test_budget_cap():
  """Large wall_dt is capped to 10 steps (spiral-of-death prevention)."""
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(wall_dt=1.0)
  assert v.sim_step_count == 10


def test_pause_and_resume():
  """Pausing stops physics; resuming resyncs clocks (no catch-up burst)."""
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(wall_dt=0.03, step_wall_time=0.0)
  assert v.sim_step_count == 3

  v.pause()
  v.inject_tick(wall_dt=0.5)
  assert v.sim_step_count == 3  # No steps while paused

  v.resume()
  v.inject_tick(wall_dt=0.01, step_wall_time=0.0)
  assert v.sim_step_count == 4  # Exactly 1, no burst


def test_step_wall_time_excluded_from_budget():
  """Slow physics wall time is subtracted to prevent feedback spiral."""
  v = FakeViewer(step_dt=0.01)
  v.inject_tick(wall_dt=0.01, step_wall_time=0.0)
  assert v.sim_step_count == 1

  # wall_dt=15ms but 5ms was step time → idle_dt=10ms → 1 step, not 2.
  v.inject_tick(wall_dt=0.015, step_wall_time=0.005)
  assert v.sim_step_count == 2

  # Stable: stays at 1 step/tick instead of spiraling.
  v.inject_tick(wall_dt=0.015, step_wall_time=0.005)
  assert v.sim_step_count == 3


def test_render_independent_of_physics():
  """Render timing follows frame_rate, not physics stepping."""
  v = FakeViewer(step_dt=0.01, frame_rate=60.0)

  assert v.inject_tick(wall_dt=0.001) is True  # First tick always renders
  assert v.inject_tick(wall_dt=0.001) is False  # Too soon
  assert v.inject_tick(wall_dt=1.0 / 60.0) is True  # Frame time elapsed
