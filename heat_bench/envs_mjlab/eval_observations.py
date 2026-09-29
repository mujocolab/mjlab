"""Observation term wiring the thermal + battery physics engines into mjlab.

``ThermalEnergyObservation`` is a passive, read-only observation term: it
reads actuator state each control step, feeds it through the two physics
engines, and returns their state as an observation vector. It does not
modify torque, rewards, or termination -- see the ``# FAULT-INJECTION SEAM``
comment below for where a future active-fault-mitigation phase would hook
in without touching the engines themselves.

Joule heat and mechanical power are accumulated at *physics-substep*
resolution rather than sampled once after the control step's decimation
loop. Empirically (measured against a trained Go1 policy on rough terrain,
32 envs x 200 control steps), sampling only the post-decimation actuator
state understates mean I^2R heat by ~7% and misses foot-impact torque
spikes by up to ~3x at the tail (p99 relative error ~170%), because torque
can vary sharply between the ``decimation`` physics substeps inside one
control step and qfrc_actuator only reflects the last one. mjlab has no
official per-substep hook for observation terms (only ``MetricsManager``
supports ``per_substep``, and only for scalar-per-env terms), so this term
wraps ``Simulation.step`` -- called exactly once per physics substep, only
inside the decimation loop -- to accumulate heat/power there instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from heat_bench.physics.battery_ecm import BatchedBatteryECM
from heat_bench.physics.lptn_engine import BatchedLPTNEngine
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg


class ThermalEnergyObservation:
  """Steps the LPTN thermal engine and battery ECM, exposing their state."""

  def __init__(self, cfg: "ObservationTermCfg", env: "ManagerBasedRlEnv"):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    hb_cfg: dict = cfg.params["config"]
    # Fault-injection seam: a future phase can pass a `fault_cfg` dict here
    # (e.g. torque clamp ranges, OVP thresholds) without changing this
    # constructor's call signature; today it's always None.
    self._fault_cfg: dict | None = cfg.params.get("fault_cfg")

    self._asset = env.scene[asset_cfg.name]
    joint_ids, joint_names = self._asset.find_joints(asset_cfg.joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    self.joint_names: list[str] = joint_names
    """Resolved joint names, in the same order as last_current/last_torque
    and the first 12 entries of thermal.T (e.g. viewer code reads this)."""

    thermal_cfg = hb_cfg["thermal"]
    self._gear_ratio_n = float(thermal_cfg["gear_ratio_N"])
    self._kt = float(thermal_cfg["motor_torque_constant_Kt"])
    self._rd = float(thermal_cfg["phase_resistance_Rd"])

    self.thermal = BatchedLPTNEngine(thermal_cfg, env.num_envs, env.device)
    self.battery = BatchedBatteryECM(hb_cfg["battery"], env.num_envs, env.device)

    self.last_joint_temps = torch.zeros(env.num_envs, 12, device=env.device)
    self.last_energy_wh_step = torch.zeros(env.num_envs, device=env.device)
    self.last_soc = torch.ones(env.num_envs, device=env.device)
    self.last_current = torch.zeros(env.num_envs, 12, device=env.device)
    self.last_torque = torch.zeros(env.num_envs, 12, device=env.device)

    # Physics-substep accumulators, drained and reset every __call__. See
    # the module docstring for why heat/power are accumulated here instead
    # of sampled once after the control step's decimation loop.
    self._heat_accum = torch.zeros(env.num_envs, 12, device=env.device)
    self._current_accum = torch.zeros(env.num_envs, 12, device=env.device)
    self._torque_accum = torch.zeros(env.num_envs, 12, device=env.device)
    self._mech_power_accum = torch.zeros(env.num_envs, device=env.device)
    self._substep_count = 0

    # Simulation.step() is called exactly once per physics substep, only
    # inside ManagerBasedRlEnv.step()'s decimation loop -- never during
    # reset. Wrapping it here is the only way to observe actuator state at
    # substep resolution; there's no official per-substep hook for
    # observation terms (only MetricsManager.compute_substep(), and that's
    # scalar-per-env only).
    self._orig_sim_step = env.sim.step
    env.sim.step = self._accumulate_substep  # ty: ignore[invalid-assignment]

  def _accumulate_substep(self) -> None:
    self._orig_sim_step()
    tau = self._asset.data.qfrc_actuator[:, self._joint_ids]
    qd = self._asset.data.joint_vel[:, self._joint_ids]

    # FAULT-INJECTION SEAM: a future phase would clamp/perturb `tau` here
    # (e.g. dropping commanded torque to 0 on an OVP trip) before it feeds
    # the electro-thermal model below. This runs every physics substep, so
    # a future fault (e.g. a millisecond-scale back-EMF trip) would be
    # visible here even though the control step is coarser.
    current = tau / (self._gear_ratio_n * self._kt)

    self._heat_accum += current.square() * self._rd
    self._current_accum += current.abs()
    self._torque_accum += tau
    self._mech_power_accum += (tau * qd).clamp(min=0.0).sum(dim=-1)
    self._substep_count += 1

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    self.thermal.reset(env_ids)
    self.battery.reset(env_ids)

  def __call__(
    self, env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg, **kwargs
  ) -> torch.Tensor:
    # The manager calls func(env, **cfg.params) every step, so this must
    # accept (and ignore) the same params dict used by __init__ (e.g.
    # "config"), not just the ones __call__ actually needs.
    del kwargs

    if self._substep_count > 0:
      joule_heat = self._heat_accum / self._substep_count
      mean_current = self._current_accum / self._substep_count
      mean_torque = self._torque_accum / self._substep_count
      mech_power = self._mech_power_accum / self._substep_count
    else:
      # No physics substep has run yet -- e.g. the observation compute
      # that happens during env.reset(), before any Simulation.step()
      # call. Fall back to a direct sample so the term still returns a
      # well-formed value.
      tau = self._asset.data.qfrc_actuator[:, self._joint_ids]
      qd = self._asset.data.joint_vel[:, self._joint_ids]
      current = tau / (self._gear_ratio_n * self._kt)
      joule_heat = current.square() * self._rd
      mean_current = current.abs()
      mean_torque = tau
      mech_power = (tau * qd).clamp(min=0.0).sum(dim=-1)

    self._heat_accum.zero_()
    self._current_accum.zero_()
    self._torque_accum.zero_()
    self._mech_power_accum.zero_()
    self._substep_count = 0

    base_lin_vel_xy = env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, :2]
    temps = self.thermal.step(joule_heat, base_lin_vel_xy, env.step_dt)
    energy_step, bus_voltage, soc = self.battery.step(
      mean_current, mech_power, joule_heat, env.step_dt
    )

    self.last_joint_temps = temps[:, :12]
    self.last_energy_wh_step = energy_step
    self.last_soc = soc
    self.last_current = mean_current
    self.last_torque = mean_torque

    return torch.cat(
      [temps[:, :12], soc.unsqueeze(-1), bus_voltage.unsqueeze(-1)], dim=-1
    )


def _thermal_energy_term(env: "ManagerBasedRlEnv", obs_group: str, obs_term: str):
  return env.observation_manager.get_term_cfg(obs_group, obs_term).func


def thermal_max_joint_temp(
  env: "ManagerBasedRlEnv", obs_group: str, obs_term: str
) -> torch.Tensor:
  term = _thermal_energy_term(env, obs_group, obs_term)
  return term.last_joint_temps.max(dim=-1).values


def thermal_mean_joint_temp(
  env: "ManagerBasedRlEnv", obs_group: str, obs_term: str
) -> torch.Tensor:
  term = _thermal_energy_term(env, obs_group, obs_term)
  return term.last_joint_temps.mean(dim=-1)


def battery_energy_wh_step(
  env: "ManagerBasedRlEnv", obs_group: str, obs_term: str
) -> torch.Tensor:
  term = _thermal_energy_term(env, obs_group, obs_term)
  return term.last_energy_wh_step


def battery_soc(
  env: "ManagerBasedRlEnv", obs_group: str, obs_term: str
) -> torch.Tensor:
  term = _thermal_energy_term(env, obs_group, obs_term)
  return term.last_soc


def battery_cumulative_energy_wh(
  env: "ManagerBasedRlEnv", obs_group: str, obs_term: str
) -> torch.Tensor:
  """Running total since the last reset, for live monitoring (the ``sum``
  reduce metric only exposes this at episode end, not per step)."""
  term = _thermal_energy_term(env, obs_group, obs_term)
  return term.battery.cum_wh
