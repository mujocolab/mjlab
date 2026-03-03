"""Base class for environment viewers.

The viewer runs a policy against an RL environment and displays the
result in real time.

The Problem
===========

Each call to ``env.step()`` advances the simulation by a fixed amount
of sim time (``step_dt``, set by the env config). The simplest viewer
would call ``env.step()`` once per loop iteration, but iterations take
however long the hardware needs. A fast machine loops 67 times per
second and the simulation runs too fast; a slow machine loops 30 times
and it runs too slow. Playback speed would depend on hardware.

The viewer needs to answer a different question: given how much real
time just elapsed, how many calls to ``env.step()`` will keep the
simulation advancing at the right pace?

Budget Accumulator
==================

A single variable, ``_sim_budget``, tracks how much sim time has
accumulated but not yet been simulated. Each tick of the main loop:

  1. Measure real time elapsed since the last tick.
  2. Multiply by the speed setting and add to the budget.
  3. Call ``env.step()`` in a loop, subtracting ``step_dt`` from the
     budget each time, until the budget is less than one step.
  4. Carry the leftover to the next tick.

Example at 1x speed, step_dt = 0.02s (50 Hz control), 60 fps::

  tick 1:  +0.0167s  ->  budget = 0.0167  ->  no step (< 0.02)
  tick 2:  +0.0167s  ->  budget = 0.0334  ->  1 step, 0.0134 left
  tick 3:  +0.0167s  ->  budget = 0.0301  ->  1 step, 0.0101 left
  ...

This averages to 50 steps per second on any hardware. At 2x speed the
elapsed time is doubled before adding to the budget, so steps happen
twice as often. At 0.5x, half as often.

Rendering is independent: the display refreshes at ``frame_rate``
(e.g. 60 Hz) whether or not a new step happened. Some frames will
re-display the same state.

If physics is too slow to keep up, the budget grows without bound. A
real time deadline (one frame period) caps each burst so the renderer
always gets a turn. Leftover budget is dropped and ``_was_capped`` is
set.

Main Loop
=========

::

  run()
    setup()
    while running:
      tick()                    -> True if a frame was produced
        _process_actions()      drain UI action queue
        _step_physics(dt)       accumulate budget, step until spent
        sync_env_to_viewer()    push state to display (at frame_rate)
      sleep(1ms)                yield CPU when no frame is due

Subclasses implement setup(), sync_env_to_viewer(), sync_viewer_to_env(),
close(), and is_running().
"""

from __future__ import annotations

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

    # State.
    self._is_paused = False
    self._step_count = 0
    self._last_error: str | None = None

    # Speed.
    self._speed_index = self.SPEED_MULTIPLIERS.index(1.0)
    self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

    # Physics accumulator and render timer.
    self._sim_budget = 0.0
    self._time_until_next_render = 0.0
    self._last_tick_time = 0.0
    self._was_capped = False

    # Windowed stats, updated every 0.5s.
    self._stats_frames = 0
    self._stats_steps = 0
    self._stats_last_time = 0.0
    self._fps = 0.0
    self._sps = 0.0

    # Action queue, drained on main thread each tick.
    self._actions: deque[tuple[ViewerAction, Optional[Any]]] = deque()

  # Abstract hooks.

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

  def _forward_paused(self) -> None:  # noqa: B027
    """Hook for subclasses to run forward kinematics while paused."""

  def _handle_custom_action(self, action: ViewerAction, payload: Optional[Any]) -> bool:
    del action, payload
    return False

  # Logging.

  def log(self, message: str, level: VerbosityLevel = VerbosityLevel.INFO) -> None:
    if self.verbosity >= level:
      print(message)

  # Thread-safe action requests.

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
    try:
      action = ViewerAction[name]
    except KeyError:
      action = ViewerAction.CUSTOM
    self._actions.append((action, payload))

  # Speed controls.

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

  # Pause and resume.

  def pause(self) -> None:
    self._is_paused = True
    self.log("[INFO] Simulation paused", VerbosityLevel.INFO)

  def resume(self) -> None:
    self._is_paused = False
    self._last_error = None
    self._sim_budget = 0.0
    self._last_tick_time = time.perf_counter()
    self.log("[INFO] Simulation resumed", VerbosityLevel.INFO)

  def toggle_pause(self) -> None:
    if self._is_paused:
      self.resume()
    else:
      self.pause()

  # Core loop.

  def _execute_step(self) -> bool:
    """Run one obs/policy/step cycle.

    Returns True on success, False if step failed.
    """
    try:
      with torch.no_grad():
        obs = self.env.get_observations()
        actions = self.policy(obs)
        self.env.step(actions)
        self._step_count += 1
        self._stats_steps += 1
        return True
    except Exception:
      self._last_error = traceback.format_exc()
      self.log(
        f"[ERROR] Exception during step:\n{self._last_error}",
        VerbosityLevel.SILENT,
      )
      self.pause()
      return False

  def _step_physics(self, dt: float) -> None:
    """Run physics steps for this frame's sim-time budget."""
    step_dt = self.env.unwrapped.step_dt
    self._sim_budget += dt * self._time_multiplier
    self._was_capped = False

    if self._sim_budget < step_dt:
      return

    self.sync_viewer_to_env()
    hit_deadline = False
    deadline = time.perf_counter() + self.frame_time
    while self._sim_budget >= step_dt:
      if not self._execute_step():
        self._sim_budget = 0.0
        return
      self._sim_budget -= step_dt
      if time.perf_counter() > deadline:
        hit_deadline = True
        break

    if hit_deadline:
      # Only report capped if we actually had to drop remaining work. A transient stall
      # (GC pause) during a single step triggers the deadline but leaves no remaining
      # budget, so it's not a real cap.
      self._was_capped = self._sim_budget >= step_dt
      self._sim_budget = min(self._sim_budget, step_dt)

  def _single_step(self) -> None:
    """Advance exactly one step while paused."""
    if not self._is_paused:
      return
    self.sync_viewer_to_env()
    self._execute_step()

  def reset_environment(self) -> None:
    self.env.reset()
    reset_fn = getattr(self.policy, "reset", None)
    if reset_fn is not None:
      reset_fn()
    self._step_count = 0
    self._sim_budget = 0.0
    self._last_error = None
    self._last_tick_time = time.perf_counter()

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
        _ = self._handle_custom_action(action, payload)

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

    # Render at fixed frame rate.
    self._time_until_next_render -= dt
    if self._time_until_next_render > 0:
      return False

    self._time_until_next_render += self.frame_time
    if self._time_until_next_render < -self.frame_time:
      self._time_until_next_render = 0.0

    self.sync_env_to_viewer()
    self._stats_frames += 1
    return True

  def run(self, num_steps: Optional[int] = None) -> None:
    self.setup()
    now = time.perf_counter()
    self._stats_last_time = now
    self._last_tick_time = now
    try:
      while self.is_running() and (num_steps is None or self._step_count < num_steps):
        if not self.tick():
          time.sleep(0.001)
        self._update_stats()
    finally:
      self.close()

  # Stats.

  def _update_stats(self) -> None:
    if self._is_paused:
      return
    now = time.perf_counter()
    dt = now - self._stats_last_time
    if dt >= 0.5:
      self._fps = self._stats_frames / dt
      self._sps = self._stats_steps / dt
      self._stats_frames = 0
      self._stats_steps = 0
      self._stats_last_time = now

      if self.verbosity >= VerbosityLevel.DEBUG:
        status = self.get_status()
        print(
          f"[{'PAUSED' if status.paused else 'RUNNING'}] "
          f"Step {status.step_count} | "
          f"FPS: {status.smoothed_fps:.0f} | "
          f"Speed: {status.speed_label} | "
          f"RTF: {status.actual_realtime:.2f}x / "
          f"{status.target_realtime:.2f}x"
        )

  @property
  def target_realtime(self) -> float:
    return self._time_multiplier

  @property
  def actual_realtime(self) -> float:
    return self._sps * self.env.unwrapped.step_dt

  @staticmethod
  def _format_speed(multiplier: float) -> str:
    if multiplier == 1.0:
      return "1x"
    inv = 1.0 / multiplier
    inv_rounded = round(inv)
    if abs(inv - inv_rounded) < 1e-9 and inv_rounded > 0:
      return f"1/{inv_rounded}x"
    return f"{multiplier:.3g}x"

  def get_status(self) -> ViewerStatus:
    return ViewerStatus(
      paused=self._is_paused,
      step_count=self._step_count,
      speed_multiplier=self._time_multiplier,
      speed_label=self._format_speed(self._time_multiplier),
      target_realtime=self.target_realtime,
      actual_realtime=self.actual_realtime,
      smoothed_fps=self._fps,
      capped=self._was_capped,
      last_error=self._last_error,
    )
