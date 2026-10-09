"""Per-(env, joint) actuator health state and the actuator writes it drives.

``apply_actuator_health`` is the active counterpart to the read-only
``ThermalEnergyObservation``: a separate ``mode="step"`` event term that owns
thermal derating and failure states (see ``heat_bench/PLAN.md``). It reads
the observation term's cached joint temperatures and writes
``actuator_forcerange`` every control step as::

  baseline * derate * thermal_derate

- ``thermal_derate`` (Phase 1) is ``Kt(T) / Kt_spec``, capped at 1: a real
  motor driver limits *current*, so as the magnets warm and Kt fades the
  same current limit yields proportionally less torque
  (``physics/motor_thermal.py``). Reversible -- recomputed every step.
- ``derate`` is the externally-set fault factor (``scripted_joint_fault``,
  future shutdown conditions); the two multiply, as a current-capacity loss
  and a torque-per-amp loss would physically.
- Thermal death (Phase 2): once a joint's temperature reaches
  ``actuator_health.dead_temp_c`` it latches ``DEAD`` (``derate`` forced to
  0) until the episode resets, even after it cools. The threshold is
  user-set and compared against the lumped joint node, which runs cooler
  than the winding hot spot. ``None`` disables it.

That product is the *current-limited* range. When
``battery.voltage_limited_torque`` is on, the range is further narrowed to
what the battery's bus voltage can drive at the joint's current speed
(``MotorThermalModel.voltage_torque_bounds``): back-EMF eats into torque in
the direction of motion, so a sagging or depleted pack clips fast motions
first while braking and low-speed torque stay current-limited. The voltage
bounds are clamped *into* the current-limited range, so the written range
is always valid (a dead joint stays [0, 0]).

Timing: step events run after the decimation loop and before observation
compute (``ManagerBasedRlEnv.step``), so each call sees joint temperatures
from the end of the previous control step and its writes govern every
physics substep of the next one. Control-step granularity is intentional --
failure state shouldn't flicker at the physics rate. The voltage bounds use
the joint speed at the end of the decimation loop, so they lag fast swings
by up to one control step.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from heat_bench.physics.motor_thermal import MotorThermalModel
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


class ActuatorState(IntEnum):
  """Discrete per-joint consequence state (see PLAN.md, "Causes vs.
  consequence states"). Thermal Kt fade alone never changes it; only fault
  conditions (e.g. ``scripted_joint_fault``, the thermal death latch) set
  non-HEALTHY states."""

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
    self._asset = asset
    joint_ids, self.joint_names = asset.find_joints(asset_cfg.joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

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
    """Fault torque-ceiling scale in [0, 1], set externally; 1.0 = no fault."""
    self.thermal_derate = torch.ones(env.num_envs, num_joints, device=env.device)
    """Kt(T)/Kt_spec torque-ceiling scale in (0, 1], recomputed every step."""
    hb_cfg: dict = cfg.params["config"]
    self._motor = MotorThermalModel.from_config(hb_cfg["thermal"])
    self._gear_ratio = float(hb_cfg["thermal"]["gear_ratio_N"])
    self._voltage_limited = bool(hb_cfg["battery"].get("voltage_limited_torque", False))
    self._dead_temp_c: float | None = hb_cfg["actuator_health"].get("dead_temp_c")
    self.voltage_lo = torch.full(
      (env.num_envs, num_joints), -torch.inf, device=env.device
    )
    """Bus-voltage lower torque bound (N·m); -inf when the limit is off."""
    self.voltage_hi = torch.full(
      (env.num_envs, num_joints), torch.inf, device=env.device
    )
    """Bus-voltage upper torque bound (N·m); +inf when the limit is off."""
    self.state = torch.full(
      (env.num_envs, num_joints),
      ActuatorState.HEALTHY,
      device=env.device,
      dtype=torch.int8,
    )
    """Discrete ``ActuatorState`` per (env, joint)."""
    self.dead = torch.zeros(
      env.num_envs, num_joints, device=env.device, dtype=torch.bool
    )
    """Thermal death latch per (env, joint); cleared only on reset."""
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
    self.thermal_derate[env_ids] = 1.0
    self.voltage_lo[env_ids] = -torch.inf
    self.voltage_hi[env_ids] = torch.inf
    self.state[env_ids] = ActuatorState.HEALTHY
    self.dead[env_ids] = False
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
    config: dict,
  ) -> None:
    del env_ids, asset_cfg, config  # Step mode covers all envs; config read in init.
    if self._thermal_term is None:
      self._thermal_term = env.observation_manager.get_term_cfg(
        obs_group, obs_term
      ).func

    # Temps are from the end of the previous control step (see module
    # docstring). Colder-than-spec magnets don't raise the spec effort limit.
    temps = self._thermal_term.last_joint_temps
    self.thermal_derate = (
      self._motor.torque_constant(temps) / self._motor.kt_spec
    ).clamp(max=1.0)

    if self._dead_temp_c is not None:
      self.dead |= temps >= self._dead_temp_c
      # Applied after any fault event's write this step, so nothing revives
      # a dead joint before reset.
      self.derate.masked_fill_(self.dead, 0.0)
      self.state.masked_fill_(self.dead, ActuatorState.DEAD)

    scale = self.derate * self.thermal_derate
    current_range = self.baseline_forcerange * scale.unsqueeze(-1)
    if self._voltage_limited:
      lo_c, hi_c = current_range[..., 0], current_range[..., 1]
      self.voltage_lo, self.voltage_hi = self._motor.voltage_torque_bounds(
        temps,
        self._asset.data.joint_vel[:, self._joint_ids],
        self._thermal_term.battery.bus_voltage,
        self._gear_ratio,
      )
      # voltage_lo < voltage_hi always, so clamping both into [lo_c, hi_c]
      # keeps lo <= hi even at overspeed, where they collapse to an edge.
      current_range = torch.stack(
        [
          self.voltage_lo.clamp(min=lo_c, max=hi_c),
          self.voltage_hi.clamp(min=lo_c, max=hi_c),
        ],
        dim=-1,
      )
    self._last_written = current_range
    env.sim.model.actuator_forcerange[:, self.ctrl_ids] = self._last_written


class scripted_joint_fault:
  """Demo scenario: drive joints healthy -> derated -> dead on a timer.

  Not a physical failure model -- a scripted stand-in for exercising the
  ``apply_actuator_health`` write path alongside Phase 1's temperature-driven
  ``thermal_derate``. Writes that term's ``derate``/``state`` buffers
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
