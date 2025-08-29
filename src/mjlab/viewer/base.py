"""Base class for environment viewers."""

from __future__ import annotations

import contextlib
import time
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Optional, Protocol

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedEnvCfg


class VerbosityLevel(IntEnum):
  """Verbosity levels for viewer output."""

  SILENT = 0  # No output.
  INFO = 1  # Basic info (help menu, pause/resume, etc.).
  DEBUG = 2  # Performance metrics and detailed logs.


class Timer:
  """Measures time elapsed between two ticks."""

  def __init__(self):
    """Instance initializer."""
    self._previous_time = time.time()
    self._measured_time = 0.0

  def tick(self):
    """Updates the timer.

    Returns:
      Time elapsed since the last call to this method.
    """
    curr_time = time.time()
    self._measured_time = curr_time - self._previous_time
    self._previous_time = curr_time
    return self._measured_time

  @contextlib.contextmanager
  def measure_time(self):
    start_time = time.time()
    yield
    self._measured_time = time.time() - start_time

  @property
  def measured_time(self):
    return self._measured_time


class EnvProtocol(Protocol):
  """Protocol for environment interface."""

  device: torch.device

  # @property is convenient for treating cfg as read-only. This makes typing easier.
  @property
  def cfg(self) -> ManagerBasedEnvCfg: ...

  def get_observations(self) -> Any: ...

  def step(self, actions: torch.Tensor) -> tuple[Any, ...]: ...

  def reset(self) -> Any: ...

  @property
  def unwrapped(self) -> Any: ...

  @property
  def num_envs(self) -> int: ...


class PolicyProtocol(Protocol):
  """Protocol for policy interface."""

  def __call__(self, obs: torch.Tensor) -> torch.Tensor: ...


class BaseViewer(ABC):
  """Abstract base class for environment viewers."""

  # fmt: off
  SPEED_MULTIPLIERS = [
    0.01, 0.016, 0.025, 0.04, 0.063, 0.1, 0.16, 0.25, 0.4, 0.63,
    1.0,
    1.6, 2.5, 4.0, 6.3, 10.0,
  ]
  # fmt: on

  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 60.0,
    render_all_envs: bool = True,
    verbosity: int = VerbosityLevel.SILENT,
  ):
    """Initialize the base viewer.

    Args:
      env: The environment to visualize.
      policy: The policy to use for action generation.
      frame_rate: Target frame rate for visualization.
      render_all_envs: Whether to render all environments or just one.
      verbosity: Verbosity level (0=silent, 1=info, 2=debug with performance metrics).
    """
    self.env = env
    self.policy = policy
    self.frame_rate = frame_rate
    self.frame_time = 1.0 / frame_rate
    self.render_all_envs = render_all_envs
    self.verbosity = VerbosityLevel(verbosity)
    self.cfg = env.cfg.viewer

    # State management.
    self._is_running = False
    self._is_paused = False
    self._step_count = 0

    # Timing components.
    self._timer = Timer()
    self._sim_timer = Timer()
    self._render_timer = Timer()
    self._time_until_next_frame = 0.0

    self._speed_index = self.SPEED_MULTIPLIERS.index(1.0)  # Start at real-time
    self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

    # Performance tracking.
    self._frame_count = 0
    self._last_fps_log_time = 0.0
    self._accumulated_sim_time = 0.0
    self._accumulated_render_time = 0.0

  def log(self, message: str, level: VerbosityLevel = VerbosityLevel.INFO) -> None:
    """Log a message if verbosity level allows it.

    Args:
      message: The message to log.
      level: The verbosity level required to show this message.
    """
    if self.verbosity >= level:
      print(message)

  @abstractmethod
  def setup(self) -> None:
    """Setup the viewer resources."""
    pass

  @abstractmethod
  def render_frame(self) -> None:
    """Render a single frame."""
    pass

  @abstractmethod
  def sync_env_to_viewer(self) -> None:
    """Synchronize environment state to viewer."""
    pass

  @abstractmethod
  def sync_viewer_to_env(self) -> None:
    """Synchronize viewer state to environment (e.g., perturbations)."""
    pass

  @abstractmethod
  def update_visualizations(self) -> None:
    """Update any additional visualizations."""
    pass

  @abstractmethod
  def close(self) -> None:
    """Close the viewer and cleanup resources."""
    pass

  def is_running(self) -> bool:
    """Check if the viewer is still running."""
    return self._is_running

  def step_simulation(self) -> None:
    """Execute one simulation step."""
    if self._is_paused:
      return

    with self._sim_timer.measure_time():
      obs = self.env.get_observations()
      actions = self.policy(obs)
      self.env.step(actions)
      self._step_count += 1

    self._accumulated_sim_time += self._sim_timer.measured_time

  def reset_environment(self) -> None:
    """Reset the environment."""
    self.env.reset()
    self._step_count = 0

  def pause(self) -> None:
    """Pause the simulation."""
    self._is_paused = True
    print("[INFO] Simulation paused")

  def resume(self) -> None:
    """Resume the simulation."""
    self._is_paused = False
    # Reset timer to avoid large time jump.
    self._timer.tick()
    print("[INFO] Simulation resumed")

  def toggle_pause(self) -> None:
    """Toggle pause state."""
    if self._is_paused:
      self.resume()
    else:
      self.pause()

  def tick(self) -> bool:
    """Execute one tick of the viewer loop.

    Returns:
      True if a frame was rendered, False otherwise.
    """
    # Measure time since last tick.
    elapsed_time = self._timer.tick() * self._time_multiplier
    self._time_until_next_frame -= elapsed_time

    # No time to render.
    if self._time_until_next_frame > 0:
      return False

    # Reset frame timer for next frame.
    self._time_until_next_frame += self.frame_time

    # Clamp to prevent accumulating delays.
    if self._time_until_next_frame < -self.frame_time:
      self._time_until_next_frame = 0

    # Perform frame operations.
    with self._render_timer.measure_time():
      self.sync_viewer_to_env()
      self.step_simulation()
      self.update_visualizations()
      self.sync_env_to_viewer()
      self.render_frame()

    self._accumulated_render_time += self._render_timer.measured_time
    self._frame_count += 1

    if self.verbosity >= VerbosityLevel.DEBUG:
      current_time = time.time()
      if current_time - self._last_fps_log_time >= 1.0:  # Log every second
        self.log_performance()
        self._last_fps_log_time = current_time
        self._frame_count = 0
        self._accumulated_sim_time = 0
        self._accumulated_render_time = 0

    return True

  def run(self, num_steps: Optional[int] = None) -> None:
    """Run the viewer loop.

    Args:
      num_steps: Number of steps to run. If None, run indefinitely.
    """
    self.setup()
    self._last_fps_log_time = time.time()
    self._timer.tick()
    try:
      while self.is_running() and (num_steps is None or self._step_count < num_steps):
        frame_rendered = self.tick()
        if not frame_rendered:
          time.sleep(0.001)
    finally:
      self.close()

  def log_performance(self) -> None:
    """Log performance metrics."""
    if self._frame_count > 0:
      avg_sim_time = self._accumulated_sim_time / self._frame_count * 1000
      avg_render_time = self._accumulated_render_time / self._frame_count * 1000
      total_time = avg_sim_time + avg_render_time

      status = "PAUSED" if self._is_paused else "RUNNING"
      speed = f"{self._time_multiplier:.1f}x" if self._time_multiplier != 1.0 else "1x"

      print(
        f"[{status}] Step {self._step_count} | FPS: {self._frame_count:.1f} | "
        f"Speed: {speed} | Sim: {avg_sim_time:.1f}ms | "
        f"Render: {avg_render_time:.1f}ms | Total: {total_time:.1f}ms"
      )

  def increase_speed(self) -> None:
    """Increase playback speed to next level."""
    if self._speed_index < len(self.SPEED_MULTIPLIERS) - 1:
      self._speed_index += 1
      self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]
      self._log_speed_change()

  def decrease_speed(self) -> None:
    """Decrease playback speed to previous level."""
    if self._speed_index > 0:
      self._speed_index -= 1
      self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]
      self._log_speed_change()

  def _log_speed_change(self) -> None:
    """Log the current speed setting."""
    speed = self._time_multiplier
    if speed < 1.0:
      percent = speed * 100
      slowdown = 1.0 / speed
      self.log(
        f"[INFO] Speed: {percent:.1f}% ({slowdown:.1f}x slower) [{self._speed_index}/{len(self.SPEED_MULTIPLIERS) - 1}]",
        VerbosityLevel.INFO,
      )
    elif speed > 1.0:
      self.log(
        f"[INFO] Speed: {speed:.1f}x faster [{self._speed_index}/{len(self.SPEED_MULTIPLIERS) - 1}]",
        VerbosityLevel.INFO,
      )
    else:
      self.log(
        f"[INFO] Speed: Real-time (1.0x) [{self._speed_index}/{len(self.SPEED_MULTIPLIERS) - 1}]",
        VerbosityLevel.INFO,
      )
