"""Recorder manager for logging environment data during rollouts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBase, ManagerTermBaseCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class RecorderTermCfg(ManagerTermBaseCfg):
  """Configuration for a recorder term.

  ``func`` must be a :class:`RecorderTerm` subclass. Function-based terms are
  not supported because recorder terms are stateful (file handles, buffers,
  etc.).
  """


class RecorderTerm(ManagerTermBase):
  """Base class for recorder terms.

  Override only the lifecycle methods you need. Each method is a no-op by
  default so subclasses are not required to implement all of them.

  The environment is available as ``self._env``, giving access to
  ``self._env.obs_buf``, ``self._env.action_manager.action``, and all
  other environment state.

  Example::

    class CsvRecorder(RecorderTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._file = open(cfg.params["path"], "w", newline="")
            self._writer = csv.writer(self._file)

        def record_pre_reset(self, env_ids):
            # Terminal transition: action is still intact here.
            # It will be zeroed by _reset_idx immediately after this returns.
            obs = self._env.obs_buf["actor"][env_ids].cpu().numpy()
            act = self._env.action_manager.action[env_ids].cpu().numpy()
            for o, a in zip(obs, act):
                self._writer.writerow(o.tolist() + a.tolist())

        def record_post_step(self):
            # Skip envs that just reset: their terminal pair was written
            # in record_pre_reset and their action is now zeroed.
            mask = ~self._env.reset_buf
            obs = self._env.obs_buf["actor"][mask].cpu().numpy()
            act = self._env.action_manager.action[mask].cpu().numpy()
            for o, a in zip(obs, act):
                self._writer.writerow(o.tolist() + a.tolist())

        def close(self):
            self._file.close()
  """

  def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)  # ManagerTermBase only accepts env
    self.cfg = cfg

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    """Called in ``env.step()`` before terminated environments are reset.

    **What is available:**

    - ``obs_buf`` contains the observation from the *end of the previous
      step* (the input the agent used to choose the terminal action). It
      does **not** contain the post-action terminal observation (the state
      reached after applying the action), which is never computed for
      resetting environments.
    - ``action_manager.action`` contains the action applied during this
      step. This is the correct terminal action. It will be zeroed for
      these environments by ``_reset_idx`` immediately after this hook
      returns, so capture it here if you need it later.
    - ``reward_buf`` contains the reward for this terminal step.
    - ``reset_terminated`` and ``reset_time_outs`` reflect why each
      environment is resetting.

    This is the right hook to record the terminal transition
    ``(obs_t, action_t, reward_t, done=True)`` for each resetting
    environment.

    Args:
      env_ids: Indices of environments that are about to be reset.
    """
    del env_ids  # Unused in base implementation.

  def record_post_reset(self, env_ids: torch.Tensor) -> None:
    """Called after a reset completes with fresh observations computed.

    Fires at the end of ``env.reset()`` (covering all environments on
    the initial call) and within ``env.step()`` for each batch of
    environments that terminates, after state has been overwritten and
    new observations computed.

    At this point ``obs_buf[env_ids]`` holds the initial observation of
    the new episode and ``action_manager.action[env_ids]`` is zero (no
    action has been taken in the new episode yet).

    Use this hook to initialize per-episode state or record the first
    observation of a new episode.

    Args:
      env_ids: Indices of environments that were reset.
    """
    del env_ids  # Unused in base implementation.

  def record_post_step(self) -> None:
    """Called at the end of every ``env.step()`` with fresh observations.

    At this point ``obs_buf`` holds the new observation for every
    environment and ``action_manager.action`` holds the action that was
    applied during this step. **Exception:** for environments that reset
    during this step, ``action_manager.action`` has been zeroed by
    ``_reset_idx`` and ``obs_buf`` holds the initial observation of the
    new episode rather than the post-action terminal observation. Use
    ``record_pre_reset`` to capture the terminal ``(obs, action)`` pair
    for those environments. Resetting environments are identified by
    ``self._env.reset_buf``.
    """

  def close(self) -> None:
    """Called when the environment closes.

    Release file handles, flush write buffers, or finalize output here.
    """

  def __call__(self):
    raise NotImplementedError(
      "RecorderTerm is not invoked via __call__. "
      "Override the lifecycle methods instead."
    )


class RecorderManager(ManagerBase):
  """Orchestrates recorder terms during environment rollouts.

  Holds a collection of :class:`RecorderTerm` instances and calls their
  lifecycle methods at the appropriate points in the environment loop. The
  manager has no opinion on how data is stored; each term handles its own
  I/O entirely.

  Register terms by adding them to the ``recorders`` dict on
  :class:`~mjlab.envs.ManagerBasedRlEnvCfg`. If the dict is empty, the
  environment substitutes a :class:`NullRecorderManager` with zero overhead.
  """

  def __init__(self, cfg: dict[str, RecorderTermCfg], env: ManagerBasedRlEnv):
    self._term_names: list[str] = []
    self._terms: list[RecorderTerm] = []
    self.cfg = deepcopy(cfg)
    super().__init__(env)  # calls _prepare_terms()

  def __str__(self) -> str:
    msg = f"<RecorderManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()
    table.title = "Active Recorder Terms"
    table.field_names = ["Index", "Name"]
    for idx, name in enumerate(self._term_names):
      table.add_row([idx, name])
    return msg + table.get_string()

  @property
  def active_terms(self) -> list[str]:
    return self._term_names

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    """Forward to each term's :meth:`RecorderTerm.record_pre_reset`."""
    for term in self._terms:
      term.record_pre_reset(env_ids)

  def record_post_reset(self, env_ids: torch.Tensor) -> None:
    """Forward to each term's :meth:`RecorderTerm.record_post_reset`."""
    for term in self._terms:
      term.record_post_reset(env_ids)

  def record_post_step(self) -> None:
    """Forward to each term's :meth:`RecorderTerm.record_post_step`."""
    for term in self._terms:
      term.record_post_step()

  def close(self) -> None:
    """Forward to each term's :meth:`RecorderTerm.close`."""
    for term in self._terms:
      term.close()

  def _prepare_terms(self):
    for name, cfg in self.cfg.items():
      if cfg is None:
        continue
      self._resolve_common_term_cfg(name, cfg)
      # _resolve_common_term_cfg instantiates class-based terms in-place.
      if not isinstance(cfg.func, RecorderTerm):
        raise TypeError(
          f"Recorder term '{name}': func must be a RecorderTerm subclass,"
          f" got {type(cfg.func).__name__}. Function-based terms are not"
          " supported."
        )
      self._term_names.append(name)
      self._terms.append(cfg.func)


class NullRecorderManager:
  """No-op fallback used when no recorder terms are configured.

  All methods are no-ops. This class is not a :class:`ManagerBase` subclass
  so it carries zero overhead.
  """

  def __init__(self):
    self.active_terms: list[str] = []
    self.cfg = None

  def __str__(self) -> str:
    return "<NullRecorderManager> (inactive)"

  def __repr__(self) -> str:
    return "NullRecorderManager()"

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def record_post_reset(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def record_post_step(self) -> None:
    pass

  def close(self) -> None:
    pass
