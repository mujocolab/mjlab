"""Command manager for generating and updating commands."""

from __future__ import annotations

import abc
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

  @abc.abstractmethod
  def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
    """Build the command term from this config."""
    raise NotImplementedError


class CommandTerm(ManagerTermBase):
  """Base class for command terms."""

  def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRlEnv):
    self.cfg = cfg
    super().__init__(env)
    self.metrics = dict()
    self.time_left = torch.zeros(self.num_envs, device=self.device)
    self.command_counter = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self._debug_vis_enabled: bool = True

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    if self.cfg.debug_vis and self._debug_vis_enabled:
      self._debug_vis_impl(visualizer)

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    pass

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Create interactive GUI controls for this command term.

    Override in subclasses to add task-specific controls (e.g., velocity
    sliders) to the Viser viewer. Called once during viewer setup.

    The *name* argument is the term's key in the command manager config
    (e.g., ``"twist"``).
    """

  @property
  @abc.abstractmethod
  def command(self):
    raise NotImplementedError

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    extras = {}
    for metric_name, metric_value in self.metrics.items():
      extras[metric_name] = torch.mean(metric_value[env_ids]).item()
      metric_value[env_ids] = 0.0
    self.command_counter[env_ids] = 0
    self._resample(env_ids)
    return extras

  def compute(self, dt: float) -> None:
    self._update_metrics()
    self.time_left -= dt
    resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
    if len(resample_env_ids) > 0:
      self._resample(resample_env_ids)
    self._update_command()

  def _resample(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) != 0:
      self.time_left[env_ids] = self.time_left[env_ids].uniform_(
        *self.cfg.resampling_time_range
      )
      self._resample_command(env_ids)
      self.command_counter[env_ids] += 1

  @abc.abstractmethod
  def _update_metrics(self) -> None:
    """Update the metrics based on the current state."""
    raise NotImplementedError

  @abc.abstractmethod
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Resample the command for the specified environments."""
    raise NotImplementedError

  @abc.abstractmethod
  def _update_command(self) -> None:
    """Update the command based on the current state."""
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
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Let each command term create its GUI controls."""
    for name, term in self._terms.items():
      term.create_gui(name, server, get_env_idx)

  def create_debug_vis_gui(self, server: "viser.ViserServer") -> None:
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

  def compute(self, dt: float):
    for term in self._terms.values():
      term.compute(dt)

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
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    pass

  def create_debug_vis_gui(self, server: "viser.ViserServer") -> None:
    pass

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    return []

  def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    return {}

  def compute(self, dt: float) -> None:
    pass

  def get_command(self, name: str) -> None:
    return None

  def get_term(self, name: str) -> None:
    return None

  def get_term_cfg(self, name: str) -> None:
    return None
