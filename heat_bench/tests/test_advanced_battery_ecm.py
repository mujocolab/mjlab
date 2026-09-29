"""Pure-tensor tests for AdvancedBatteryECM, Model 2 (no simulation required).

Focuses on the two things this model adds over BatchedBatteryECM: SoC-
dependent internal resistance, and thermal-coupled capacity-loss/aging.
"""

import os

import torch
from heat_bench.physics.battery_ecm import AdvancedBatteryECM

CFG = {
  "capacity_ah": 8.0,
  "ocv_at_soc0_v": 24.0,
  "ocv_at_soc1_v": 33.6,
  "initial_soc": 1.0,
  "soc_dependent_resistance": {
    "mid_ohm": 0.03,
    "low_soc_extra_ohm": 0.05,
    "high_soc_extra_ohm": 0.01,
    "soc_breakpoint_low": 0.15,
    "soc_breakpoint_high": 0.9,
  },
  "aging": {
    "B": 130.0,
    "Ea_j_per_mol": 18461.0,
    "alpha": 32.0,
    "R_gas": 8.314,
    "z": 0.4,
  },
}
DT = 0.02


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


def make_battery(num_envs: int) -> AdvancedBatteryECM:
  return AdvancedBatteryECM(CFG, num_envs, get_test_device())


def _room_temp(num_envs: int, device: str) -> torch.Tensor:
  return torch.full((num_envs,), 25.0, device=device)


def test_resistance_is_higher_near_empty_than_mid_soc():
  """Sag for the same current should be larger near-empty than mid-range --
  matches the literature's "substantial increase near SoC extremes" shape."""
  low_soc_battery = make_battery(num_envs=1)
  mid_soc_battery = make_battery(num_envs=1)
  low_soc_battery.soc[:] = 0.05
  mid_soc_battery.soc[:] = 0.5

  current = torch.full((1, 12), 5.0 / 12, device=low_soc_battery.device)
  mech_power = torch.zeros(1, device=low_soc_battery.device)
  heat = torch.zeros(1, 12, device=low_soc_battery.device)
  chassis_temp = _room_temp(1, low_soc_battery.device)

  _, low_v, _, _ = low_soc_battery.step(current, mech_power, heat, chassis_temp, DT)
  _, mid_v, _, _ = mid_soc_battery.step(current, mech_power, heat, chassis_temp, DT)

  low_ocv = low_soc_battery._ocv(torch.tensor([0.05], device=low_soc_battery.device))
  mid_ocv = mid_soc_battery._ocv(torch.tensor([0.5], device=mid_soc_battery.device))
  low_sag = (low_ocv - low_v).item()
  mid_sag = (mid_ocv - mid_v).item()

  assert low_sag > mid_sag


def test_resistance_flat_across_mid_soc_range():
  """Within [breakpoint_low, breakpoint_high], R(SoC) should be exactly
  mid_ohm regardless of SoC (the flat middle segment)."""
  battery = make_battery(num_envs=1)
  soc = torch.tensor([0.3, 0.5, 0.7], device=battery.device)
  r = battery._internal_resistance(soc)
  mid_ohm: float = CFG["soc_dependent_resistance"]["mid_ohm"]  # type: ignore[index]
  expected = torch.full_like(r, mid_ohm)
  assert torch.allclose(r, expected)


def test_capacity_loss_increases_with_more_ah_throughput():
  """capacity_loss_pct should grow (not shrink) as cumulative Ah grows,
  for a fixed current/temperature operating point."""
  battery = make_battery(num_envs=1)
  current = torch.full((1, 12), 5.0 / 12, device=battery.device)
  mech_power = torch.zeros(1, device=battery.device)
  heat = torch.zeros(1, 12, device=battery.device)
  chassis_temp = _room_temp(1, battery.device)

  losses = []
  for _ in range(50):
    _, _, _, capacity_loss = battery.step(current, mech_power, heat, chassis_temp, DT)
    losses.append(capacity_loss.item())

  # EMA warm-up means the first few steps may not be monotonic; check the
  # overall trend instead of every single step.
  assert losses[-1] > losses[5]


def test_capacity_loss_higher_at_higher_temperature():
  """Same current/Ah throughput, hotter chassis -> more aging."""
  cool_battery = make_battery(num_envs=1)
  hot_battery = make_battery(num_envs=1)
  current = torch.full((1, 12), 5.0 / 12, device=cool_battery.device)
  mech_power = torch.zeros(1, device=cool_battery.device)
  heat = torch.zeros(1, 12, device=cool_battery.device)

  cool_temp = torch.full((1,), 25.0, device=cool_battery.device)
  hot_temp = torch.full((1,), 60.0, device=hot_battery.device)

  for _ in range(20):
    _, _, _, cool_loss = cool_battery.step(current, mech_power, heat, cool_temp, DT)
    _, _, _, hot_loss = hot_battery.step(current, mech_power, heat, hot_temp, DT)

  assert hot_loss.item() > cool_loss.item()


def test_reset_zeros_capacity_loss_and_averages():
  battery = make_battery(num_envs=2)
  current = torch.full((2, 12), 5.0 / 12, device=battery.device)
  mech_power = torch.zeros(2, device=battery.device)
  heat = torch.zeros(2, 12, device=battery.device)
  chassis_temp = _room_temp(2, battery.device)

  for _ in range(20):
    battery.step(current, mech_power, heat, chassis_temp, DT)

  assert (battery.capacity_loss_pct > 0).all()

  battery.reset(env_ids=torch.tensor([0], device=battery.device))
  assert battery.capacity_loss_pct[0].item() == 0.0
  assert battery.capacity_loss_pct[1].item() > 0.0
  assert battery._avg_current[0].item() == 0.0
