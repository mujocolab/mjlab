"""Command manager for generating and updating commands."""

from __future__ import annotations

import abc
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBase

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_MAX_HISTORY_CAPACITY = 256
"""Slot cap, for configs with no real horizon."""


class CommandHistory:
  """Per-episode record of the commands a term sampled.

  An episode is a sequence of segments: the command drawn on reset, then one per
  resample. Slot ``k`` of env ``i`` holds the command that took effect at
  ``start_times[i, k]``, in seconds since that env's episode began; only the
  first ``lengths[i]`` slots are valid.

  Appending is two-phase: :meth:`mark_pending` on resample, :meth:`flush` once
  the term has post-processed the sampled value into the one it will track.
  """

  def __init__(
    self, num_envs: int, capacity: int, command_dim: int, device: str
  ) -> None:
    if capacity < 1:
      raise ValueError(f"capacity must be at least 1, got {capacity}.")
    self.num_envs = num_envs
    self.capacity = capacity
    self.commands = torch.zeros(num_envs, capacity, command_dim, device=device)
    self.start_times = torch.zeros(num_envs, capacity, device=device)
    self.lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    self._pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self._slots = torch.arange(capacity, device=device)

  @property
  def valid_mask(self) -> torch.Tensor:
    """Which slots hold a segment, shape (num_envs, capacity)."""
    return self._slots < self.lengths.unsqueeze(-1)

  def clear(self, env_ids: torch.Tensor) -> None:
    """Drop the record of the given envs and cancel their pending appends."""
    self.commands[env_ids] = 0.0
    self.start_times[env_ids] = 0.0
    self.lengths[env_ids] = 0
    self._pending[env_ids] = False

  def mark_pending(self, env_ids: torch.Tensor) -> None:
    self._pending[env_ids] = True

  def flush(self, commands: torch.Tensor, times: torch.Tensor) -> None:
    """Append ``commands``/``times`` for every env marked pending."""
    env_ids = self._pending.nonzero(as_tuple=False).flatten()
    if len(env_ids) == 0:
      return
    # Saturate rather than raise: a resampling range shortened at runtime can
    # outrun the capacity the buffer was sized for.
    slots = self.lengths[env_ids].clamp(max=self.capacity - 1)
    self.commands[env_ids, slots] = commands[env_ids]
    self.start_times[env_ids, slots] = times[env_ids]
    self.lengths[env_ids] = (self.lengths[env_ids] + 1).clamp(max=self.capacity)
    self._pending[env_ids] = False

  def durations(self, end_time: float | torch.Tensor) -> torch.Tensor:
    """Return how long each segment lasted, shape (num_envs, capacity).

    A segment ends when the next one starts; the last one is still open, so the
    caller says when it ends via ``end_time`` (scalar or per env) -- the current
    episode time to measure the episode as it happened, the full episode length
    to hold the last command to the end. Zero past :attr:`lengths`.
    """
    end = torch.as_tensor(
      end_time, dtype=self.start_times.dtype, device=self.start_times.device
    )
    end = end.expand(self.num_envs).reshape(self.num_envs, 1)
    next_start = torch.empty_like(self.start_times)
    next_start[:, :-1] = self.start_times[:, 1:]
    next_start[:, -1:] = end
    last_slot = (self.lengths - 1).clamp(min=0).unsqueeze(-1)
    next_start.scatter_(1, last_slot, end)
    return (next_start - self.start_times).clamp(min=0.0) * self.valid_mask


@dataclass(kw_only=True)
class CommandTermCfg(abc.ABC):
  """Configuration for a command generator term.

  Command terms generate goal commands for the agent (e.g., target velocity,
  target position). Commands are automatically resampled at configurable
  intervals and can track metrics for logging.
  """

  resampling_time_range: tuple[float, float]
  """Time range in seconds for command resampling. When the timer expires, a new
  command is sampled and the timer is reset to a value uniformly drawn from
  ``[min, max]``. Set both values equal for fixed-interval resampling."""

  debug_vis: bool = False
  """Whether to enable debug visualization for this command term. When True,
  the command term's ``_debug_vis_impl`` method is called each frame to render
  visual aids (e.g., velocity arrows, target markers)."""

  track_command_history: bool = False
  """Whether to keep a :class:`CommandHistory` of every command sampled during
  an episode, stamped with the episode time it took effect.
  """

  @abc.abstractmethod
  def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
    """Build the command term from this config."""
    raise NotImplementedError


class CommandTerm(ManagerTermBase):
  """Base class for command terms."""

  def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRlEnv):
    self.cfg = cfg
    super().__init__(env)
    self._check_update_command_signature()
    self.metrics = dict()
    self.time_left = torch.zeros(self.num_envs, device=self.device)
    self.command_counter = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self._debug_vis_enabled: bool = True
    self._command_history: CommandHistory | None = None

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    if self.cfg.debug_vis and self._debug_vis_enabled:
      self._debug_vis_impl(visualizer)

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    pass

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create interactive GUI controls for this command term.

    Override in subclasses to add task-specific controls (e.g., velocity sliders) to
    the Viser viewer. Called once during viewer setup.

    The *name* argument is the term's key in the command manager config (e.g.,
    ``"twist"``).
    """

  def on_viewer_pause(self, paused: bool) -> None:
    """Called when the viewer pause state changes."""

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    """Apply GUI-selected state as an env reset override.

    Returns True if this term wrote state to sim.
    """
    return False

  @property
  @abc.abstractmethod
  def command(self):
    raise NotImplementedError

  @property
  def command_history(self) -> CommandHistory | None:
    """Commands sampled this episode, or None unless ``cfg`` asks to track them.

    Cleared per episode in :meth:`reset`. A curriculum term still sees the
    finished episode, since the curriculum manager resets before this one.
    """
    return self._command_history

  def _ensure_command_history(self) -> None:
    if self._command_history is not None or not self.cfg.track_command_history:
      return
    self._command_history = CommandHistory(
      num_envs=self.num_envs,
      capacity=min(self._history_capacity(), _MAX_HISTORY_CAPACITY),
      command_dim=self.command.shape[-1],
      device=self.device,
    )

  def _segments_per_episode(self, interval: float) -> int:
    """How often something on a period of ``interval`` seconds can fire in one
    episode. For sizing overrides of :meth:`_history_capacity`."""
    horizon = self._env.max_episode_length_s
    if horizon <= 0.0 or not math.isfinite(horizon):
      return _MAX_HISTORY_CAPACITY
    # Nothing can fire faster than once per env step; this also covers a
    # zero-length interval.
    return min(
      math.ceil(horizon / max(interval, self._env.step_dt)), _MAX_HISTORY_CAPACITY
    )

  def _history_capacity(self) -> int:
    """Command history slots per env: one per resample, plus the reset draw.

    A term that also samples outside the resampling timer should override this
    and budget for that cadence on top of ``super()._history_capacity()``. The
    result is clamped to :data:`_MAX_HISTORY_CAPACITY`.
    """
    return self._segments_per_episode(self.cfg.resampling_time_range[0]) + 1

  def _record_command_resample(self, env_ids: torch.Tensor) -> None:
    """Note that ``env_ids`` were just given a new command.

    :meth:`_resample` calls this. A term that samples outside it -- as
    ``MotionCommand`` does on clip wraparound -- must call it itself for the
    history to see the new command. The value is committed at the end of the
    current :meth:`compute`, once :meth:`_update_command` has settled it.
    """
    if self._command_history is not None:
      self._command_history.mark_pending(env_ids)

  @property
  def _episode_time(self) -> torch.Tensor:
    """Seconds elapsed in the current episode, shape (num_envs,)."""
    return self._env.episode_length_buf * self._env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    self._ensure_command_history()
    extras = {}
    for metric_name, metric_value in self.metrics.items():
      extras[metric_name] = torch.mean(metric_value[env_ids]).item()
      metric_value[env_ids] = 0.0
    self.command_counter[env_ids] = 0
    # Clear before resampling, so the command drawn below is the new episode's
    # first entry. The compute() that follows commits it, by which point the env
    # has zeroed the episode clock.
    if self._command_history is not None:
      self._command_history.clear(env_ids)
    self._resample(env_ids)
    return extras

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    """Advance the command state by dt.

    With env_ids=None (the per-step path) all envs are updated; with env_ids
    (the reset path) timers and the command update are scoped to those envs.
    Metrics are always refreshed.

    dt may be a scalar (all envs) or a per-env tensor (auto-reset path,
    where freshly reset envs get zero to keep their timers full). A tensor
    dt requires env_ids=None.

    History entries are committed last, once _update_command has settled the
    value a resample drew.
    """
    self._ensure_command_history()
    self._update_metrics()
    if env_ids is None:
      self.time_left -= dt
      resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
    else:
      assert not isinstance(dt, torch.Tensor)
      self.time_left[env_ids] -= dt
      resample_env_ids = env_ids[self.time_left[env_ids] <= 0.0]
    if len(resample_env_ids) > 0:
      self._resample(resample_env_ids)
    self._update_command(env_ids)
    if self._command_history is not None:
      self._command_history.flush(self.command, self._episode_time)

  def _check_update_command_signature(self) -> None:
    """Fail fast with a migration hint for terms with the old signature."""
    try:
      sig = inspect.signature(self._update_command)
    except (TypeError, ValueError):
      return
    if len(sig.parameters) == 0:
      raise TypeError(
        f"{type(self).__name__}._update_command must accept env_ids: "
        "_update_command(self, env_ids: torch.Tensor | None). It receives "
        "None on the per-step update and the reset env ids on reset(); "
        "scope per-step state advances to env_ids."
      )

  def _resample(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) != 0:
      self.time_left[env_ids] = self.time_left[env_ids].uniform_(
        *self.cfg.resampling_time_range
      )
      self._resample_command(env_ids)
      self.command_counter[env_ids] += 1
      self._record_command_resample(env_ids)

  @abc.abstractmethod
  def _update_metrics(self) -> None:
    """Update the metrics based on the current state."""
    raise NotImplementedError

  @abc.abstractmethod
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Resample the command for the specified environments."""
    raise NotImplementedError

  @abc.abstractmethod
  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    """Update the command based on the current state.

    env_ids is None on the per-step update (all envs) and the reset env ids on reset().
    Scope per-step state advances (e.g. a motion frame index) to env_ids; pure
    functions of the current state may ignore it.
    """
    raise NotImplementedError


class CommandManager(ManagerBase):
  """Manages command generation for the environment.

  The command manager generates and updates goal commands for the agent (e.g.,
  target velocity, target position). Commands are resampled at configurable
  intervals and can track metrics for logging.
  """

  _env: ManagerBasedRlEnv

  def __init__(self, cfg: dict[str, CommandTermCfg], env: ManagerBasedRlEnv):
    self._terms: dict[str, CommandTerm] = dict()

    self.cfg = cfg
    super().__init__(env)
    self._commands = dict()

  def __str__(self) -> str:
    msg = f"<CommandManager> contains {len(self._terms.values())} active terms.\n"
    table = PrettyTable()
    table.title = "Active Command Terms"
    table.field_names = ["Index", "Name", "Type"]
    table.align["Name"] = "l"
    for index, (name, term) in enumerate(self._terms.items()):
      table.add_row([index, name, term.__class__.__name__])
    msg += table.get_string()
    msg += "\n"
    return msg

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    for term in self._terms.values():
      term.debug_vis(visualizer)

  def create_gui(
    self,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Let each command term create its GUI controls."""
    for name, term in self._terms.items():
      term.create_gui(
        name,
        server,
        get_env_idx,
        on_change=on_change,
        request_action=request_action,
      )

  def on_viewer_pause(self, paused: bool) -> None:
    """Notify all command terms of viewer pause state change."""
    for term in self._terms.values():
      term.on_viewer_pause(paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    """Apply GUI-selected state from all terms. Returns True if any applied."""
    applied = False
    for term in self._terms.values():
      applied |= term.apply_gui_reset(env_ids)
    return applied

  def create_debug_vis_gui(
    self,
    server: viser.ViserServer,
    on_change: Callable[[], None] | None = None,
  ) -> None:
    """Add per-term debug visualization checkboxes."""
    vis_terms = {name: term for name, term in self._terms.items() if term.cfg.debug_vis}
    if not vis_terms:
      return
    for name, term in vis_terms.items():
      cb = server.gui.add_checkbox(
        name.capitalize(),
        initial_value=term._debug_vis_enabled,
      )

      def _on_update(_ev, _term: CommandTerm = term, _cb=cb) -> None:
        _term._debug_vis_enabled = _cb.value
        if on_change is not None:
          on_change()

      cb.on_update(_on_update)

  # Properties.

  @property
  def active_terms(self) -> list[str]:
    return list(self._terms.keys())

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []
    idx = 0
    for name, term in self._terms.items():
      terms.append((name, term.command[env_idx].cpu().tolist()))
      idx += term.command.shape[1]
    return terms

  def reset(self, env_ids: torch.Tensor | None) -> dict[str, torch.Tensor]:
    extras = {}
    for name, term in self._terms.items():
      metrics = term.reset(env_ids=env_ids)
      for metric_name, metric_value in metrics.items():
        extras[f"Metrics/{name}/{metric_name}"] = metric_value
    return extras

  def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None):
    for term in self._terms.values():
      term.compute(dt, env_ids)

  def get_command(self, name: str) -> torch.Tensor:
    return self._terms[name].command

  def get_term(self, name: str) -> CommandTerm:
    return self._terms[name]

  def get_term_cfg(self, name: str) -> CommandTermCfg:
    return self.cfg[name]

  def _prepare_terms(self):
    for term_name, term_cfg in self.cfg.items():
      term_cfg: CommandTermCfg | None
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      term = term_cfg.build(self._env)
      if not isinstance(term, CommandTerm):
        raise TypeError(
          f"Returned object for the term {term_name} is not of type CommandType."
        )
      # Allocate now, so that a None command_history means tracking is off
      # rather than not-yet-allocated.
      term._ensure_command_history()
      self._terms[term_name] = term


class NullCommandManager:
  """Placeholder for absent command manager that safely no-ops all operations."""

  def __init__(self):
    self.active_terms: list[str] = []
    self._terms: dict[str, Any] = {}
    self.cfg = None

  def __str__(self) -> str:
    return "<NullCommandManager> (inactive)"

  def __repr__(self) -> str:
    return "NullCommandManager()"

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    pass

  def create_gui(
    self,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    pass

  def on_viewer_pause(self, paused: bool) -> None:
    pass

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    return False

  def create_debug_vis_gui(
    self,
    server: viser.ViserServer,
    on_change: Callable[[], None] | None = None,
  ) -> None:
    pass

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    return []

  def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    return {}

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    pass

  def get_command(self, name: str) -> None:
    return None

  def get_term(self, name: str) -> None:
    return None

  def get_term_cfg(self, name: str) -> None:
    return None
