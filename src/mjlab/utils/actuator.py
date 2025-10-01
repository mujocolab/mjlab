# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Electric actuator utilities."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ElectricActuator:
  """Electric actuator parameters."""

  reflected_inertia: float
  velocity_limit: float
  effort_limit: float


def reflected_inertia(
  rotor_inertia: float,
  gear_ratio: float,
) -> float:
  """Compute reflected inertia of a single-stage gearbox."""
  return rotor_inertia * gear_ratio**2


def reflected_inertia_from_two_stage_planetary(
  rotor_inertia: tuple[float, float, float],
  gear_ratio: tuple[float, float, float],
) -> float:
  """Compute reflected inertia of a two-stage planetary gearbox."""
  assert gear_ratio[0] == 1
  r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
  r2 = rotor_inertia[1] * gear_ratio[2] ** 2
  r3 = rotor_inertia[2]
  return r1 + r2 + r3


def rpm_to_rad(rpm: float) -> float:
  """Convert revolutions per minute to radians per second."""
  return (rpm * 2 * math.pi) / 60
