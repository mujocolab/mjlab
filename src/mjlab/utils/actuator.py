from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ElectricActuator:
  reflected_inertia: float
  velocity_limit: float
  effort_limit: float


def reflected_inertia(
  rotor_inertia: float,
  gear_ratio: float,
) -> float:
  return rotor_inertia * gear_ratio**2


def reflected_inertia_from_two_stage_planetary(
  rotor_inertia: tuple[float, float, float],
  gear_ratio: tuple[float, float, float],
) -> float:
  assert gear_ratio[0] == 1
  r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
  r2 = rotor_inertia[1] * gear_ratio[2] ** 2
  r3 = rotor_inertia[2]
  return r1 + r2 + r3


def rpm_to_rad(rpm: float) -> float:
  return (rpm * 2 * math.pi) / 60
