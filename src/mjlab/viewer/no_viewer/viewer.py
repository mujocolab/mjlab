"""mjlab headless play viewer.

Enables user to run the policy without having a viewer and no fixed/max framerate
"""

from __future__ import annotations

import time
from threading import Lock

from typing_extensions import override

from mjlab.viewer.base import (
  BaseViewer,
  EnvProtocol,
  PolicyProtocol,
)


class HeadlessPlayViewer(BaseViewer):
  """Interactive Viser-based viewer with playback controls."""

  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
  ) -> None:
    super().__init__(env, policy)
    self._sim_lock = Lock()

  @override
  def setup(self):
    pass

  @override
  def sync_env_to_viewer(self):
    pass

  @override
  def sync_viewer_to_env(self):
    pass

  @override
  def close(self):
    pass

  @override
  def is_running(self) -> bool:
    """Check if viewer is running."""
    return True  # Viser runs until process is killed.

  @override
  def _step_physics(self, dt: float) -> None:
    """Run physics steps for this frame's sim-time budget."""
    step_dt = self.env.unwrapped.step_dt
    self._sim_budget += dt * self._time_multiplier
    self._was_capped = False

    if self._sim_budget < step_dt:
      return

    while self._sim_budget >= step_dt:
      if not self._execute_step():
        self._sim_budget = 0.0
        return
      self._sim_budget -= step_dt

  @override
  def tick(self) -> bool:
    """Advance one tick: drain actions, step physics, maybe render.

    Returns True when a render frame was produced, False otherwise.
    """
    now = time.perf_counter()
    dt = now - self._last_tick_time
    self._last_tick_time = now

    self._process_actions()

    if self._is_paused:
      self._forward_paused()
    else:
      self._step_physics(dt)

    self._stats_frames += 1

    return True
