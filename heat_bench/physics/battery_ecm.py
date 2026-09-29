"""Batched battery equivalent-circuit model (Coulomb counting + voltage sag).

Tracks state of charge via Coulomb counting on total pack current, and
reports total energy consumed (mechanical work plus Joule-heating waste) and
instantaneous bus voltage, which sags under load via a fixed internal
resistance. Open-circuit voltage is modeled as linear in state of charge
between two placeholder endpoint voltages: no real Go2 pack datasheet is
available, and a real Li-ion OCV curve is plateau-shaped, but a linear
approximation is the right fidelity for a placeholder used to sanity-check
energy bookkeeping, and is trivially replaced with a lookup curve later
without changing this class's public API.
"""

from __future__ import annotations

import torch


class BatchedBatteryECM:
  """Vectorized battery state, stepped once per env control step."""

  def __init__(self, cfg: dict, num_envs: int, device: str):
    self.num_envs = num_envs
    self.device = device

    self._capacity_ah = float(cfg["capacity_ah"])
    self._internal_resistance = float(cfg["internal_resistance_ohm"])
    self._ocv_at_soc0 = float(cfg["ocv_at_soc0_v"])
    self._ocv_at_soc1 = float(cfg["ocv_at_soc1_v"])
    self._initial_soc = float(cfg["initial_soc"])

    self.soc = torch.zeros(num_envs, device=device)
    self.cum_ah = torch.zeros(num_envs, device=device)
    self.cum_wh = torch.zeros(num_envs, device=device)
    self.bus_voltage = torch.zeros(num_envs, device=device)
    self.reset(env_ids=None)

  def _ocv(self, soc: torch.Tensor) -> torch.Tensor:
    return self._ocv_at_soc0 + soc * (self._ocv_at_soc1 - self._ocv_at_soc0)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.soc[env_ids] = self._initial_soc
    self.cum_ah[env_ids] = 0.0
    self.cum_wh[env_ids] = 0.0
    self.bus_voltage[env_ids] = self._ocv(self.soc[env_ids])

  def step(
    self,
    per_actuator_current: torch.Tensor,
    mech_power: torch.Tensor,
    joule_heat: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance battery state by one control step.

    Args:
      per_actuator_current: Per-actuator current magnitude, shape (N, 12), A.
        Callers stepping at physics-substep resolution should pass the mean
        over substeps, not a single post-decimation sample.
      mech_power: Non-negative mechanical power (regenerative braking not
        credited), shape (N,), W. Must already be computed as
        ``(torque * vel).clamp(min=0).sum(-1)`` by the caller *before*
        averaging over physics substeps -- torque*vel is nonlinear, so
        averaging torque and vel separately and multiplying afterwards
        would give a different (wrong) answer than averaging the clamped
        per-substep power.
      joule_heat: Per-actuator I^2*Rd heat, shape (N, 12), W (shared with
        the thermal engine, not recomputed).
      dt: Control step duration (env.step_dt).

    Returns:
      Tuple of (energy_wh_step, bus_voltage, soc), each shape (N,).
    """
    elec_power = mech_power + joule_heat.sum(dim=-1)
    energy_wh_step = elec_power * dt / 3600.0

    i_pack = per_actuator_current.abs().sum(dim=-1)
    ah_step = i_pack * dt / 3600.0

    self.soc = (self.soc - ah_step / self._capacity_ah).clamp(min=0.0, max=1.0)
    self.cum_ah = self.cum_ah + ah_step
    self.cum_wh = self.cum_wh + energy_wh_step
    self.bus_voltage = self._ocv(self.soc) - i_pack * self._internal_resistance

    return energy_wh_step, self.bus_voltage, self.soc
