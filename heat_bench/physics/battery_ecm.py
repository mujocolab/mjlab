"""Two swappable battery equivalent-circuit models, same public interface.

``BatchedBatteryECM`` ("rint"): the original model. Fixed internal
resistance, linear OCV(SoC), no thermal coupling.

``AdvancedBatteryECM`` ("rint_soc_aging"): adds two upgrades identified by
researching published battery/legged-robot models (see heat_bench/README.md):
  1. SoC-dependent internal resistance (Rint literature: resistance is
     roughly flat over the mid-SoC range but rises substantially near both
     empty and full), instead of one fixed constant.
  2. A capacity-loss/aging term coupled to *thermal* state -- the
     semi-empirical model ``Q_loss = B*exp((-Ea+alpha*|I|)/(R*T))*(Ah)^z``
     (cited in Shu et al., "Learning-Based Model Predictive Control for
     Legged Robots with Battery-Supercapacitor Hybrid Energy Storage
     System," Appl. Sci. 2025, https://doi.org/10.3390/app15010382, their
     Eq. 18; that model, and B/Ea/alpha/z, originate from Petit, Prada,
     and Sauvant-Moynot, "Development of an empirical aging model for
     Li-ion batteries and application to assess the impact of Vehicle-to-
     Grid strategies on battery lifetime," Appl. Energy 2016, 172,
     398-407, Shu et al.'s reference [38]).
     Needs a battery temperature, which nothing in heat_bench tracks
     directly; this reuses the LPTN's chassis node as a proxy (the pack is
     chassis-mounted) instead of adding a 15th thermal node, which would
     break the 14-node topology match to the thermal-aware-locomotion
     literature already established for this package.

Both classes share one ``step()`` signature and 4-tuple return
(``energy_wh_step, bus_voltage, soc, capacity_loss_pct``) so callers never
need to branch on which model is selected -- ``BatchedBatteryECM`` simply
ignores ``chassis_temp_c`` and always reports ``capacity_loss_pct = 0``
(no aging tracked). Selected via ``battery.model`` in
``configs/go2_eval_config.yaml``.
"""

from __future__ import annotations

import torch


class BatchedBatteryECM:
  """Model 1 ("rint"): fixed internal resistance, linear OCV(SoC).

  No real Go2 pack datasheet gives a full OCV curve or internal-resistance
  values, so OCV is linear between two placeholder endpoints and resistance
  is one fixed constant -- the simplest model that still gives sensible
  Coulomb counting and voltage sag under load.
  """

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
    self.capacity_loss_pct = torch.zeros(num_envs, device=device)
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
    self.capacity_loss_pct[env_ids] = 0.0

  def step(
    self,
    per_actuator_current: torch.Tensor,
    mech_power: torch.Tensor,
    joule_heat: torch.Tensor,
    chassis_temp_c: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
      chassis_temp_c: Unused by this model (no thermal coupling); accepted
        so both battery models share one call signature.
      dt: Control step duration (env.step_dt).

    Returns:
      Tuple of (energy_wh_step, bus_voltage, soc, capacity_loss_pct), each
      shape (N,). capacity_loss_pct is always 0 for this model.
    """
    del chassis_temp_c

    elec_power = mech_power + joule_heat.sum(dim=-1)
    energy_wh_step = elec_power * dt / 3600.0

    i_pack = per_actuator_current.abs().sum(dim=-1)
    ah_step = i_pack * dt / 3600.0

    self.soc = (self.soc - ah_step / self._capacity_ah).clamp(min=0.0, max=1.0)
    self.cum_ah = self.cum_ah + ah_step
    self.cum_wh = self.cum_wh + energy_wh_step
    self.bus_voltage = self._ocv(self.soc) - i_pack * self._internal_resistance

    return energy_wh_step, self.bus_voltage, self.soc, self.capacity_loss_pct


class AdvancedBatteryECM:
  """Model 2 ("rint_soc_aging"): SoC-dependent resistance + thermal-coupled
  aging, on top of the same Coulomb-counting/energy bookkeeping as Model 1.
  """

  def __init__(self, cfg: dict, num_envs: int, device: str):
    self.num_envs = num_envs
    self.device = device

    self._capacity_ah = float(cfg["capacity_ah"])
    self._ocv_at_soc0 = float(cfg["ocv_at_soc0_v"])
    self._ocv_at_soc1 = float(cfg["ocv_at_soc1_v"])
    self._initial_soc = float(cfg["initial_soc"])

    r_cfg = cfg["soc_dependent_resistance"]
    self._r_mid = float(r_cfg["mid_ohm"])
    self._r_low_extra = float(r_cfg["low_soc_extra_ohm"])
    self._r_high_extra = float(r_cfg["high_soc_extra_ohm"])
    self._soc_breakpoint_low = float(r_cfg["soc_breakpoint_low"])
    self._soc_breakpoint_high = float(r_cfg["soc_breakpoint_high"])

    a_cfg = cfg["aging"]
    self._aging_b = float(a_cfg["B"])
    self._aging_ea = float(a_cfg["Ea_j_per_mol"])
    self._aging_alpha = float(a_cfg["alpha"])
    self._aging_r_gas = float(a_cfg["R_gas"])
    self._aging_z = float(a_cfg["z"])

    self.soc = torch.zeros(num_envs, device=device)
    self.cum_ah = torch.zeros(num_envs, device=device)
    self.cum_wh = torch.zeros(num_envs, device=device)
    self.bus_voltage = torch.zeros(num_envs, device=device)
    self.capacity_loss_pct = torch.zeros(num_envs, device=device)
    # Slow-moving averages of current/temperature "stress" feeding the aging
    # formula below -- the cited formula is meant to be evaluated against a
    # representative operating point, not a single noisy instantaneous
    # sample; using the raw per-step current directly would make
    # capacity_loss_pct fluctuate up and down with load instead of behaving
    # like monotonic aging.
    self._avg_current = torch.zeros(num_envs, device=device)
    self._avg_temp_k = torch.full((num_envs,), 298.15, device=device)
    self.reset(env_ids=None)

  def _ocv(self, soc: torch.Tensor) -> torch.Tensor:
    return self._ocv_at_soc0 + soc * (self._ocv_at_soc1 - self._ocv_at_soc0)

  def _internal_resistance(self, soc: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear R(SoC): flat at ``r_mid`` in the healthy mid-range,
    ramping up toward both extremes -- matches the qualitative shape
    reported for real Li-ion cells (roughly linear/flat 10-85% SoC, a
    "substantial increase" near the empty and full ends)."""
    low_ramp = (self._soc_breakpoint_low - soc).clamp(
      min=0.0
    ) / self._soc_breakpoint_low
    high_span = 1.0 - self._soc_breakpoint_high
    high_ramp = (soc - self._soc_breakpoint_high).clamp(min=0.0) / high_span
    return self._r_mid + self._r_low_extra * low_ramp + self._r_high_extra * high_ramp

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.soc[env_ids] = self._initial_soc
    self.cum_ah[env_ids] = 0.0
    self.cum_wh[env_ids] = 0.0
    self.bus_voltage[env_ids] = self._ocv(self.soc[env_ids])
    self.capacity_loss_pct[env_ids] = 0.0
    self._avg_current[env_ids] = 0.0
    self._avg_temp_k[env_ids] = 298.15

  def step(
    self,
    per_actuator_current: torch.Tensor,
    mech_power: torch.Tensor,
    joule_heat: torch.Tensor,
    chassis_temp_c: torch.Tensor,
    dt: float,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance battery state by one control step.

    Args:
      per_actuator_current: Per-actuator current magnitude, shape (N, 12), A.
      mech_power: Non-negative mechanical power, shape (N,), W (see Model 1
        for why this must be pre-averaged by the caller, not recomputed
        from separately-averaged torque/vel).
      joule_heat: Per-actuator I^2*Rd heat, shape (N, 12), W.
      chassis_temp_c: LPTN chassis node temperature, shape (N,), deg C --
        proxy for battery pack temperature (see module docstring).
      dt: Control step duration (env.step_dt).

    Returns:
      Tuple of (energy_wh_step, bus_voltage, soc, capacity_loss_pct), each
      shape (N,). capacity_loss_pct is cumulative since the last reset.
    """
    elec_power = mech_power + joule_heat.sum(dim=-1)
    energy_wh_step = elec_power * dt / 3600.0

    i_pack = per_actuator_current.abs().sum(dim=-1)
    ah_step = i_pack * dt / 3600.0

    self.soc = (self.soc - ah_step / self._capacity_ah).clamp(min=0.0, max=1.0)
    self.cum_ah = self.cum_ah + ah_step
    self.cum_wh = self.cum_wh + energy_wh_step

    # EMA decay ~0.99 at 50Hz gives a ~2s time constant -- smooths out
    # step-to-step noise so the aging formula below sees a representative
    # recent operating point instead of one instantaneous sample.
    ema_decay = 0.99
    self._avg_current = ema_decay * self._avg_current + (1.0 - ema_decay) * i_pack
    temp_kelvin_now = chassis_temp_c + 273.15
    self._avg_temp_k = (
      ema_decay * self._avg_temp_k + (1.0 - ema_decay) * temp_kelvin_now
    )

    r = self._internal_resistance(self.soc)
    self.bus_voltage = self._ocv(self.soc) - i_pack * r

    # Q_loss = B*exp((-Ea+alpha*|I|)/(R*T))*(Ah)^z, evaluated against the
    # smoothed recent operating point rather than this instant's current/
    # temperature (see EMA comment above) and the true monotonic cumulative
    # Ah throughput, so capacity_loss_pct behaves like aging (non-decreasing
    # in expectation) instead of tracking load moment to moment.
    self.capacity_loss_pct = (
      self._aging_b
      * torch.exp(
        (-self._aging_ea + self._aging_alpha * self._avg_current)
        / (self._aging_r_gas * self._avg_temp_k)
      )
      * self.cum_ah.clamp(min=1e-6) ** self._aging_z
    )

    return energy_wh_step, self.bus_voltage, self.soc, self.capacity_loss_pct
