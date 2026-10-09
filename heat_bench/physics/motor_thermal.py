"""Temperature dependence of a PMSM actuator's phase resistance and Kt.

Two distinct, reversible effects, both linear in temperature:

- **Phase resistance** ``Rd(T)``: copper resistivity rises with temperature,
  so the same current dissipates more Joule heat. Annealed copper's
  temperature coefficient is 0.00393 /°C referenced to 20°C (NBS Misc. Pub.
  17, *Copper Wire Card*, 1919).
- **Torque constant** ``Kt(T)``: NdFeB remanence ``Br`` falls with
  temperature, so the same torque needs more current. Arnold Magnetic
  Technologies' N42 datasheet gives a reversible coefficient of α(Br) =
  −0.12 %/°C, measured between 20 and 80°C (applied linearly beyond 80°C
  here; Go2's actual magnet grade is unknown). Since ``Kt ∝ Br``, the same
  coefficient applies to ``Kt``.

Only the *reversible* magnet fade is modeled. Permanent demagnetization
above a grade's maximum operating temperature is a separate, accumulating
effect (see ``heat_bench/PLAN.md``).

Each coefficient is defined relative to its own reference temperature
(``coeff_ref_c``, 20°C for both sources above), while the nominal Rd/Kt
values are specified at ``spec_ref_c``; ``temperature_scale`` converts
between the two exactly instead of assuming they coincide.

The motor-to-joint reduction is per joint (``joint_gear_ratios``): the
motor's own gearbox ``gear_ratio_N`` times an optional extra stage on some
joints, e.g. a quadruped knee's linkage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mjlab.utils.string import resolve_expr

# Floor on the Kt scale so a (non-physical) extrapolation far past the
# datasheet range can never divide by zero or flip sign.
_MIN_KT_SCALE = 0.05


def joint_gear_ratios(
  thermal_cfg: dict, joint_names: Sequence[str], device: str | torch.device
) -> torch.Tensor:
  """Per-joint motor-to-joint reduction ``N``, shape (J,).

  ``gear_ratio_N`` times the first matching entry of the optional
  ``joint_gear_ratio_scale`` regex map (1.0 for unmatched joints).
  """
  scales = resolve_expr(
    thermal_cfg.get("joint_gear_ratio_scale", {}), tuple(joint_names), 1.0
  )
  return float(thermal_cfg["gear_ratio_N"]) * torch.tensor(
    scales, device=device, dtype=torch.float32
  )


def temperature_scale(
  temps: torch.Tensor, coeff_per_c: float, coeff_ref_c: float, spec_ref_c: float
) -> torch.Tensor:
  """Return ``X(T) / X(spec_ref_c)`` for a property linear in temperature.

  ``X(T) = X(coeff_ref_c) * (1 + coeff_per_c * (T - coeff_ref_c))``.
  """
  spec = 1.0 + coeff_per_c * (spec_ref_c - coeff_ref_c)
  return (1.0 + coeff_per_c * (temps - coeff_ref_c)) / spec


def phase_resistance(
  temps: torch.Tensor,
  rd_spec: float,
  coeff_per_c: float,
  coeff_ref_c: float,
  spec_ref_c: float,
) -> torch.Tensor:
  """Per-joint phase resistance (ohm) at ``temps`` (°C)."""
  return rd_spec * temperature_scale(temps, coeff_per_c, coeff_ref_c, spec_ref_c)


def torque_constant(
  temps: torch.Tensor,
  kt_spec: float,
  coeff_per_c: float,
  coeff_ref_c: float,
  spec_ref_c: float,
) -> torch.Tensor:
  """Per-joint motor torque constant (N·m/A) at ``temps`` (°C)."""
  scale = temperature_scale(temps, coeff_per_c, coeff_ref_c, spec_ref_c)
  return kt_spec * scale.clamp(min=_MIN_KT_SCALE)


@dataclass(frozen=True)
class MotorThermalModel:
  """Rd(T)/Kt(T) for one motor type, built from the yaml ``thermal`` section.

  Setting both coefficients to 0 recovers constant ``Rd``/``Kt``.
  """

  rd_spec: float
  kt_spec: float
  rd_coeff_per_c: float
  kt_coeff_per_c: float
  coeff_ref_c: float
  spec_ref_c: float

  @classmethod
  def from_config(cls, thermal_cfg: dict) -> MotorThermalModel:
    return cls(
      rd_spec=float(thermal_cfg["phase_resistance_Rd"]),
      kt_spec=float(thermal_cfg["motor_torque_constant_Kt"]),
      rd_coeff_per_c=float(thermal_cfg.get("phase_resistance_temp_coeff_per_c", 0.0)),
      kt_coeff_per_c=float(thermal_cfg.get("torque_constant_temp_coeff_per_c", 0.0)),
      coeff_ref_c=float(thermal_cfg.get("motor_temp_coeff_reference_c", 20.0)),
      spec_ref_c=float(thermal_cfg.get("motor_constants_reference_temp_c", 25.0)),
    )

  def phase_resistance(self, temps: torch.Tensor) -> torch.Tensor:
    return phase_resistance(
      temps, self.rd_spec, self.rd_coeff_per_c, self.coeff_ref_c, self.spec_ref_c
    )

  def torque_constant(self, temps: torch.Tensor) -> torch.Tensor:
    return torque_constant(
      temps, self.kt_spec, self.kt_coeff_per_c, self.coeff_ref_c, self.spec_ref_c
    )

  def voltage_torque_bounds(
    self,
    temps: torch.Tensor,
    joint_vel: torch.Tensor,
    bus_voltage: torch.Tensor,
    gear_ratio: float | torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint-side (lower, upper) torque bounds allowed by the bus voltage.

    DC-equivalent motor voltage equation ``V = I·Rd + Ke·ω`` with
    ``Ke = Kt`` (SI units, energy-consistent: electrical power in equals
    mechanical power out plus I²R heat), motor speed ``ω = N·q̇`` and joint
    torque ``τ = N·Kt·I``. Back-EMF opposes torque in the direction of
    motion and adds to braking torque, so the bounds are asymmetric:

      upper = N·Kt·(+V − Kt·N·q̇) / Rd
      lower = N·Kt·(−V − Kt·N·q̇) / Rd

    Args:
      temps: Joint temperatures (°C), shape (N, J).
      joint_vel: Joint velocities (rad/s), shape (N, J).
      bus_voltage: Battery bus voltage (V), shape (N,).
      gear_ratio: Motor-to-joint reduction ``N``, scalar or per joint (J,).
    """
    kt = self.torque_constant(temps)
    rd = self.phase_resistance(temps)
    back_emf = kt * gear_ratio * joint_vel
    v = bus_voltage.unsqueeze(-1)
    scale = gear_ratio * kt / rd
    return scale * (-v - back_emf), scale * (v - back_emf)
