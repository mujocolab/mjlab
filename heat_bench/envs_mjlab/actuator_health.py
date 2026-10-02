"""Per-(env, joint) actuator health state and the actuator writes it drives.

``apply_actuator_health`` is the active counterpart to the passive
``ThermalEnergyObservation``: a separate ``mode="step"`` event term that will
own thermal derating and failure states (see ``heat_bench/PLAN.md``). It
reads the observation term's cached state and writes ``actuator_forcerange``
-- the observation term itself is never modified.

Phase 0 (this file today) is plumbing only: the state buffers exist and
reset per episode, and every control step writes ``baseline * derate`` with
``derate`` fixed at 1.0, i.e. an identity write that leaves the simulation
unchanged.

Timing: step events run after the decimation loop and before observation
compute (``ManagerBasedRlEnv.step``), so each call sees joint temperatures
from the end of the previous control step and its writes govern every
physics substep of the next one. Control-step granularity is intentional --
failure state shouldn't flicker at the physics rate.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


class ActuatorState(IntEnum):
  """Discrete per-joint consequence state (see PLAN.md, "Causes vs.
  consequence states"). Only HEALTHY is used in Phase 0."""

  HEALTHY = 0
  DERATED = 1
  FREE = 2
  LOCKED = 3
  DEAD = 4


class apply_actuator_health:
  """Owns per-(env, joint) actuator health and writes ``actuator_forcerange``.

  Use with ``mode="step"``. Buffers are indexed in the same joint order as
  ``ThermalEnergyObservation.last_joint_temps`` (both resolve joints with
  ``find_joints`` on the same ``asset_cfg``).
  """

  # What @requires_model_fields would set; declared directly because that
  # decorator is typed for functions. EventManager reads these off the term
  # to expand actuator_forcerange per world.
  model_fields = ("actuator_forcerange",)
  recompute = RecomputeLevel.none

  def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedRlEnv"):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._env = env
    asset = env.scene[asset_cfg.name]
    joint_ids, self.joint_names = asset.find_joints(asset_cfg.joint_names)

    # Actuator ctrl order need not match joint order, so map each resolved
    # joint to the ctrl index of the actuator driving it.
    ctrl_for_joint: dict[int, int] = {}
    for actuator in asset.actuators:
      for joint_id, ctrl_id in zip(
        actuator.target_ids.tolist(), actuator.global_ctrl_ids.tolist(), strict=True
      ):
        ctrl_for_joint[joint_id] = ctrl_id
    missing = [
      n
      for j, n in zip(joint_ids, self.joint_names, strict=True)
      if j not in ctrl_for_joint
    ]
    if missing:
      raise ValueError(f"No actuator drives joints {missing}.")
    self.ctrl_ids = torch.tensor(
      [ctrl_for_joint[j] for j in joint_ids], device=env.device, dtype=torch.long
    )

    num_joints = len(joint_ids)
    self.derate = torch.ones(env.num_envs, num_joints, device=env.device)
    """Continuous torque-ceiling scale in [0, 1]; 1.0 = full effort limit."""
    self.state = torch.full(
      (env.num_envs, num_joints),
      ActuatorState.HEALTHY,
      device=env.device,
      dtype=torch.int8,
    )
    """Discrete ``ActuatorState`` per (env, joint)."""
    self.baseline_forcerange = env.sim.model.actuator_forcerange[
      :, self.ctrl_ids
    ].clone()
    """Per-env undegraded forcerange that ``derate`` scales. Refreshed at
    reset only where a reset-mode DR event (e.g. ``dr.effort_limits``) has
    rewritten the field, so derating scales the randomized limit instead of
    overwriting it -- see ``reset``."""
    self._last_written = self.baseline_forcerange.clone()

    # Resolved lazily: the EventManager is built before the ObservationManager.
    self._thermal_term = None

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.derate[env_ids] = 1.0
    self.state[env_ids] = ActuatorState.HEALTHY
    # The live field still holds this term's own (possibly derated) write
    # from the previous episode unless a DR event rewrote it since. Only take
    # values that differ from our last write as the new baseline; otherwise a
    # joint derated or killed last episode would stay degraded forever.
    live = self._env.sim.model.actuator_forcerange[env_ids][:, self.ctrl_ids]
    rewritten = live != self._last_written[env_ids]
    self.baseline_forcerange[env_ids] = torch.where(
      rewritten, live, self.baseline_forcerange[env_ids]
    )
    # Write the restored limits now: the next step event only runs after
    # that step's physics, so otherwise the new episode's first control step
    # would still use last episode's degraded forcerange.
    self._last_written[env_ids] = self.baseline_forcerange[env_ids]
    rows = torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
    self._env.sim.model.actuator_forcerange[rows[:, None], self.ctrl_ids] = (
      self.baseline_forcerange[env_ids]
    )

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    obs_group: str,
    obs_term: str,
  ) -> None:
    del env_ids, asset_cfg  # Step mode always covers all envs.
    if self._thermal_term is None:
      self._thermal_term = env.observation_manager.get_term_cfg(
        obs_group, obs_term
      ).func

    # Phase 1 hook: update self.derate / self.state from
    # self._thermal_term.last_joint_temps here.

    self._last_written = self.baseline_forcerange * self.derate.unsqueeze(-1)
    env.sim.model.actuator_forcerange[:, self.ctrl_ids] = self._last_written


class scripted_joint_fault:
  """Demo scenario: drive joints healthy -> derated -> dead on a timer.

  Not a physical failure model -- a scripted stand-in for exercising the
  ``apply_actuator_health`` write path before Phases 1-2 decide derate/state
  from tracked temperature. Writes that term's ``derate``/``state`` buffers
  for the given joints as a function of per-env episode time, so the schedule
  restarts whenever an env resets. Use with ``mode="step"``, registered
  *before* the health term so its write lands in the same step (see
  ``add_scripted_joint_fault``).

  Schedule: full limit until ``derate_start_s``; linear ramp down to
  ``derate_floor`` by ``derate_end_s``; held there until ``dead_at_s``, then
  zero torque (``DEAD``) for the rest of the episode.
  """

  def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedRlEnv"):
    del env
    self._joint_names: tuple[str, ...] = tuple(cfg.params["joint_names"])
    self._health_term_name: str = cfg.params["health_term"]
    self._health: apply_actuator_health | None = None
    self._joint_idx = torch.empty(0, dtype=torch.long)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    joint_names: tuple[str, ...],
    health_term: str,
    derate_start_s: float,
    derate_end_s: float,
    derate_floor: float,
    dead_at_s: float,
  ) -> None:
    del env_ids, joint_names, health_term
    if self._health is None:
      self._health = env.event_manager.get_term_cfg(self._health_term_name).func
      unknown = [n for n in self._joint_names if n not in self._health.joint_names]
      if unknown:
        raise ValueError(
          f"Unknown joints {unknown}; expected any of {self._health.joint_names}."
        )
      self._joint_idx = torch.tensor(
        [self._health.joint_names.index(n) for n in self._joint_names],
        device=env.device,
      )

    t = env.episode_length_buf.float() * env.step_dt
    ramp = ((t - derate_start_s) / (derate_end_s - derate_start_s)).clamp(0.0, 1.0)
    derate = 1.0 - ramp * (1.0 - derate_floor)
    dead = t >= dead_at_s
    state = torch.where(
      dead,
      ActuatorState.DEAD,
      torch.where(t >= derate_start_s, ActuatorState.DERATED, ActuatorState.HEALTHY),
    )
    # Same schedule for every selected joint: broadcast [N] -> [N, K].
    num_joints = len(self._joint_idx)
    self._health.derate[:, self._joint_idx] = (
      torch.where(dead, 0.0, derate).unsqueeze(-1).expand(-1, num_joints)
    )
    self._health.state[:, self._joint_idx] = (
      state.to(torch.int8).unsqueeze(-1).expand(-1, num_joints)
    )
