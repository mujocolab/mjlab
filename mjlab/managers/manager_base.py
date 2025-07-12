from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv
  from mjlab.managers.manager_term_config import ManagerTermBaseCfg


class ManagerTermBase(abc.ABC):
  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
    self.cfg = cfg
    self._env = env

  # Properties.

  @property
  def num_envs(self) -> int:
    return self._env.num_envs

  @property
  def device(self) -> str:
    return self._env.device

  @property
  def name(self) -> str:
    return self.__class__.__name__

  # Methods.

  def reset(self, env_ids: Sequence[int] | None = None) -> None:
    """Resets the manager term."""
    del env_ids  # Unused.
    pass

  def __call__(self, *args) -> Any:
    """Returns the value of the term required by the manager."""
    raise NotImplementedError


class ManagerBase(abc.ABC):
  """Base class for all managers."""

  def __init__(self, cfg, env: ManagerBasedEnv):
    self.cfg = cfg
    self._env = env

    self._prepare_terms()

  # Properties.

  @property
  def num_envs(self) -> int:
    return self._env.num_envs

  @property
  def device(self) -> str:
    return self._env.device

  @property
  @abc.abstractmethod
  def active_terms(self) -> list[str] | dict[str, list[str]]:
    raise NotImplementedError

  def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
    """Resets the manager and returns logging info for the current step."""
    del env_ids  # Unused.
    return {}

  @abc.abstractmethod
  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    raise NotImplementedError

  @abc.abstractmethod
  def _prepare_terms(self):
    raise NotImplementedError
