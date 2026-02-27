"""Base class for environment viewers."""

from __future__ import annotations

import contextlib
import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any, Optional, Protocol

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg


class VerbosityLevel(IntEnum):
  SILENT = 0
  INFO = 1
  DEBUG = 2


class Timer:
  def __init__(self):
    self._previous_time = time.time()
    self._measured_time = 0.0

  def tick(self):
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
  """Interface we expect from RL environments, which can be either vanilla
  `ManagerBasedRlEnv` objects or wrapped with `VideoRecorder`,
  `RslRlVecEnvWrapper`, etc."""

  num_envs: int

  @property
  def device(self) -> torch.device | str: ...

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg: ...

  @property
  def unwrapped(self) -> Any: ...

  def get_observations(self) -> Any: ...
  def step(self, actions: torch.Tensor) -> tuple[Any, ...]: ...
  def reset(self) -> Any: ...
  def close(self) -> None: ...


class PolicyProtocol(Protocol):
  def __call__(self, obs: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ViewerStatus:
  paused: bool
  step_count: int
  speed_multiplier: float
  speed_label: str
  target_realtime: float
  actual_realtime: float
  smoothed_fps: float
  capped: bool
  last_error: str | None


class ViewerAction(Enum):
  RESET = "reset"
  TOGGLE_PAUSE = "toggle_pause"
  SINGLE_STEP = "single_step"
  RESET_SPEED = "reset_speed"
  SPEED_UP = "speed_up"
  SPEED_DOWN = "speed_down"
  PREV_ENV = "prev_env"
  NEXT_ENV = "next_env"
  TOGGLE_PLOTS = "toggle_plots"
  TOGGLE_DEBUG_VIS = "toggle_debug_vis"
  TOGGLE_SHOW_ALL_ENVS = "toggle_show_all_envs"
  CUSTOM = "custom"


class BaseViewer(ABC):
  """Abstract base class for environment viewers."""

  SPEED_MULTIPLIERS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0]

  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 30.0,
    verbosity: int = VerbosityLevel.SILENT,
  ):
    self.env = env
    self.policy = policy
    self.frame_rate = frame_rate
    self.frame_time = 1.0 / frame_rate
    self.verbosity = VerbosityLevel(verbosity)
    self.cfg = env.cfg.viewer

    # Loop state.
    self._is_paused = False
    self._step_count = 0
    self._last_error: str | None = None
    self._max_steps_per_tick: int = 10

    # Timing.
    self._timer = Timer()
    self._sim_timer = Timer()
    self._render_timer = Timer()
    self._tracked_sim_time = 0.0  # Expected sim time (wall_dt * multiplier)
    self._actual_sim_time = 0.0  # Real sim time (advances by step_dt)
    self._time_until_next_render = 0.0

    self._speed_index = self.SPEED_MULTIPLIERS.index(1.0)
    self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

    # Perf tracking.
    self._frame_count = 0
    self._last_fps_log_time = 0.0
    self._accumulated_sim_time = 0.0
    self._accumulated_render_time = 0.0

    # FPS and realtime tracking.
    self._smoothed_fps: float = 0.0
    self._smoothed_sps: float = 0.0
    self._fps_accum_frames: int = 0
    self._sps_accum_steps: int = 0
    self._fps_accum_time: float = 0.0
    self._fps_last_frame_time: Optional[float] = None
    self._fps_update_interval: float = 0.5
    self._fps_alpha: float = 0.35

    # Capped indicator: true when max_steps_per_tick was hit recently.
    self._capped_ticks: int = 0
    self._capped_window: int = 0
    self._is_capped: bool = False

    # Thread-safe action queue (drained in main loop).
    self._actions: deque[tuple[ViewerAction, Optional[Any]]] = deque()

  # Abstract hooks every concrete viewer must implement.

  @abstractmethod
  def setup(self) -> None: ...
  @abstractmethod
  def sync_env_to_viewer(self) -> None: ...
  @abstractmethod
  def sync_viewer_to_env(self) -> None: ...
  @abstractmethod
  def close(self) -> None: ...
  @abstractmethod
  def is_running(self) -> bool: ...

  # Logging.

  def log(self, message: str, level: VerbosityLevel = VerbosityLevel.INFO) -> None:
    if self.verbosity >= level:
      print(message)

  # Public controls.

  def request_reset(self) -> None:
    self._actions.append((ViewerAction.RESET, None))

  def request_toggle_pause(self) -> None:
    self._actions.append((ViewerAction.TOGGLE_PAUSE, None))

  def request_single_step(self) -> None:
    self._actions.append((ViewerAction.SINGLE_STEP, None))

  def request_speed_up(self) -> None:
    self._actions.append((ViewerAction.SPEED_UP, None))

  def request_speed_down(self) -> None:
    self._actions.append((ViewerAction.SPEED_DOWN, None))

  def request_reset_speed(self) -> None:
    self._actions.append((ViewerAction.RESET_SPEED, None))

  def request_action(self, name: str, payload: Optional[Any] = None) -> None:
    """Viewer-specific actions (e.g., PREV_ENV/NEXT_ENV for native)."""
    try:
      action = ViewerAction[name]
    except KeyError:
      action = ViewerAction.CUSTOM
    self._actions.append((action, payload))

  # Core loop.

  def _execute_step(self) -> bool:
    """Run one obs/policy/step cycle. No pause check.

    Returns True on success, False if step failed.
    """
    # Wrap in no_grad mode to prevent gradient accumulation and memory leaks.
    # NOTE: Using torch.inference_mode() causes a "RuntimeError: Inplace update to
    # inference tensor outside InferenceMode is not allowed" inside the command
    # manager when resetting the env with a key callback.
    try:
      with torch.no_grad():
        with self._sim_timer.measure_time():
          obs = self.env.get_observations()
          actions = self.policy(obs)
          self.env.step(actions)
          self._step_count += 1
          self._sps_accum_steps += 1
        self._accumulated_sim_time += self._sim_timer.measured_time
        return True
    except Exception:
      self._last_error = traceback.format_exc()
      self.log(
        f"[ERROR] Exception during step:\n{self._last_error}",
        VerbosityLevel.SILENT,
      )
      self.pause()
      return False

  def step_simulation(self) -> bool:
    if self._is_paused:
      return False
    return self._execute_step()

  def _single_step(self) -> None:
    """Advance exactly one step while paused."""
    if not self._is_paused:
      return
    self.sync_viewer_to_env()
    step_ok = self._execute_step()
    if not step_ok:
      return
    step_dt = self.env.unwrapped.step_dt
    self._actual_sim_time += step_dt
    self._tracked_sim_time = self._actual_sim_time

  def reset_environment(self) -> None:
    self.env.reset()
    self._step_count = 0
    self._tracked_sim_time = 0.0
    self._actual_sim_time = 0.0
    self._last_error = None
    self._timer.tick()

  def pause(self) -> None:
    self._is_paused = True
    self._fps_last_frame_time = None
    self.log("[INFO] Simulation paused", VerbosityLevel.INFO)

  def resume(self) -> None:
    self._is_paused = False
    self._last_error = None
    self._tracked_sim_time = self._actual_sim_time  # Prevent catch-up burst
    self._timer.tick()
    self._fps_last_frame_time = time.time()
    self.log("[INFO] Simulation resumed", VerbosityLevel.INFO)

  def toggle_pause(self) -> None:
    if self._is_paused:
      self.resume()
    else:
      self.pause()

  def _process_actions(self) -> None:
    """Drain action queue. Runs on the main loop thread."""
    while self._actions:
      action, payload = self._actions.popleft()
      if action == ViewerAction.RESET:
        self.reset_environment()
      elif action == ViewerAction.TOGGLE_PAUSE:
        self.toggle_pause()
      elif action == ViewerAction.SINGLE_STEP:
        self._single_step()
      elif action == ViewerAction.RESET_SPEED:
        self.reset_speed()
      elif action == ViewerAction.SPEED_UP:
        self.increase_speed()
      elif action == ViewerAction.SPEED_DOWN:
        self.decrease_speed()
      else:
        # Hook for subclasses to handle PREV_ENV/NEXT_ENV or CUSTOM actions
        _ = self._handle_custom_action(action, payload)

  def _handle_custom_action(self, action: ViewerAction, payload: Optional[Any]) -> bool:
    del action, payload  # Unused.
    return False

  def _forward_paused(self) -> None:  # noqa: B027
    """Hook for subclasses to run forward kinematics while paused."""

  def tick(self) -> bool:
    """Advance the viewer by one tick.

    Physics and rendering are decoupled: physics steps are governed by a
    sim-time budget (accumulated from wall time * speed multiplier), while
    rendering always happens at the configured ``frame_rate``. This keeps
    the display smooth even at slow playback speeds.

    Returns True when a render frame was produced, False otherwise.
    """
    self._process_actions()

    wall_dt = self._timer.tick()

    # Step physics using two-clock decoupling: tracked time (where sim
    # *should* be) vs actual time (where it *is*).  When physics overshoots
    # a frame budget the next tick naturally runs zero iterations, giving
    # the renderer a chance to breathe.
    if not self._is_paused:
      step_dt = self.env.unwrapped.step_dt

      self._tracked_sim_time += wall_dt * self._time_multiplier

      # Cap: don't let tracked time race more than N steps ahead.
      max_lead = step_dt * self._max_steps_per_tick
      hit_cap = self._tracked_sim_time - self._actual_sim_time > max_lead
      if hit_cap:
        self._tracked_sim_time = self._actual_sim_time + max_lead

      # Track whether the cap is hit repeatedly (over a sliding window).
      self._capped_window += 1
      if hit_cap:
        self._capped_ticks += 1
      if self._capped_window >= 30:
        self._is_capped = self._capped_ticks > self._capped_window // 2
        self._capped_ticks = 0
        self._capped_window = 0

      # Step physics until actual time catches up to tracked time.
      # Use a small tolerance to avoid missing steps due to float rounding
      # (e.g. 0.015 - 0.005 = 0.009999… instead of 0.01).
      deficit = self._tracked_sim_time - self._actual_sim_time
      if deficit >= step_dt - 1e-10:
        self.sync_viewer_to_env()
        while deficit >= step_dt - 1e-10:
          step_ok = self.step_simulation()
          if not step_ok:
            break
          self._actual_sim_time += step_dt
          deficit -= step_dt
    else:
      self._forward_paused()

    # Render at fixed frame rate, independent of physics speed.
    self._time_until_next_render -= wall_dt
    if self._time_until_next_render > 0:
      return False

    self._time_until_next_render += self.frame_time
    if self._time_until_next_render < -self.frame_time:
      self._time_until_next_render = 0.0

    with self._render_timer.measure_time():
      self.sync_env_to_viewer()
    self._accumulated_render_time += self._render_timer.measured_time
    self._frame_count += 1
    self._update_fps()

    if self.verbosity >= VerbosityLevel.DEBUG:
      now = time.time()
      if now - self._last_fps_log_time >= 1.0:
        self.log_performance()
        self._last_fps_log_time = now
        self._frame_count = 0
        self._accumulated_sim_time = 0.0
        self._accumulated_render_time = 0.0

    return True

  def run(self, num_steps: Optional[int] = None) -> None:
    self.setup()
    self._last_fps_log_time = time.time()
    self._timer.tick()
    self._fps_last_frame_time = time.time()
    try:
      while self.is_running() and (num_steps is None or self._step_count < num_steps):
        if not self.tick():
          time.sleep(0.001)
    finally:
      self.close()

  @property
  def target_realtime(self) -> float:
    """Target realtime factor based on the current speed multiplier."""
    return self._time_multiplier

  @property
  def actual_realtime(self) -> float:
    """Actual realtime factor based on smoothed steps per second."""
    return self._smoothed_sps * self.env.unwrapped.step_dt

  @staticmethod
  def _format_speed(multiplier: float) -> str:
    """Format speed multiplier as a human-readable string (e.g. '1/4x')."""
    if multiplier == 1.0:
      return "1x"
    # Check for clean power-of-2 fractions.
    inv = 1.0 / multiplier
    inv_rounded = round(inv)
    if abs(inv - inv_rounded) < 1e-9 and inv_rounded > 0:
      return f"1/{inv_rounded}x"
    return f"{multiplier:.3g}x"

  def get_status(self) -> ViewerStatus:
    """Build a read-only status snapshot for viewer UIs."""
    return ViewerStatus(
      paused=self._is_paused,
      step_count=self._step_count,
      speed_multiplier=self._time_multiplier,
      speed_label=self._format_speed(self._time_multiplier),
      target_realtime=self.target_realtime,
      actual_realtime=self.actual_realtime,
      smoothed_fps=self._smoothed_fps,
      capped=self._is_capped,
      last_error=self._last_error,
    )

  def log_performance(self) -> None:
    if self._frame_count > 0:
      status = self.get_status()
      avg_sim_ms = self._accumulated_sim_time / self._frame_count * 1000
      avg_render_ms = self._accumulated_render_time / self._frame_count * 1000
      total_ms = avg_sim_ms + avg_render_ms
      print(
        f"[{'PAUSED' if status.paused else 'RUNNING'}] "
        f"Step {status.step_count} | FPS: {self._frame_count:.1f} | "
        f"Speed: {status.speed_label} | Sim: {avg_sim_ms:.1f}ms | "
        f"Render: {avg_render_ms:.1f}ms | "
        f"Total: {total_ms:.1f}ms"
      )

  def increase_speed(self) -> None:
    if self._speed_index < len(self.SPEED_MULTIPLIERS) - 1:
      self._speed_index += 1
      self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

  def decrease_speed(self) -> None:
    if self._speed_index > 0:
      self._speed_index -= 1
      self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

  def reset_speed(self) -> None:
    self._speed_index = self.SPEED_MULTIPLIERS.index(1.0)
    self._time_multiplier = 1.0

  def _update_fps(self) -> None:
    if self._is_paused:
      return
    now = time.time()
    if self._fps_last_frame_time is None:
      self._fps_last_frame_time = now
      return
    dt = now - self._fps_last_frame_time
    self._fps_last_frame_time = now
    if dt <= 0:
      return
    self._fps_accum_frames += 1
    self._fps_accum_time += dt
    if self._fps_accum_time >= self._fps_update_interval:
      alpha = self._fps_alpha
      inst_fps = self._fps_accum_frames / self._fps_accum_time
      inst_sps = self._sps_accum_steps / self._fps_accum_time
      if self._smoothed_fps == 0.0:
        self._smoothed_fps = inst_fps
        self._smoothed_sps = inst_sps
      else:
        self._smoothed_fps = alpha * inst_fps + (1 - alpha) * self._smoothed_fps
        self._smoothed_sps = alpha * inst_sps + (1 - alpha) * self._smoothed_sps
      self._fps_accum_frames = 0
      self._sps_accum_steps = 0
      self._fps_accum_time = 0.0
