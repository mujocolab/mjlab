"""Pure-tensor tests for BatchedBatteryECM (no simulation required)."""

import os

import torch
from heat_bench.physics.battery_ecm import BatchedBatteryECM


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


CFG = {
  "capacity_ah": 15.0,
  "internal_resistance_ohm": 0.03,
  "ocv_at_soc0_v": 19.8,
  "ocv_at_soc1_v": 25.2,
  "initial_soc": 1.0,
}
DT = 0.02


def make_battery(num_envs: int) -> BatchedBatteryECM:
  return BatchedBatteryECM(CFG, num_envs, get_test_device())


def test_zero_current_leaves_soc_and_energy_unchanged():
  battery = make_battery(num_envs=1)
  current = torch.zeros(1, 12, device=battery.device)
  mech_power = torch.zeros(1, device=battery.device)
  heat = torch.zeros(1, 12, device=battery.device)

  for _ in range(10):
    energy, bus_v, soc = battery.step(current, mech_power, heat, DT)

  assert torch.allclose(soc, torch.ones_like(soc))
  assert torch.allclose(energy, torch.zeros_like(energy))
  assert torch.allclose(bus_v, torch.tensor([25.2], device=battery.device))


def test_constant_current_matches_closed_form_coulomb_count():
  battery = make_battery(num_envs=1)
  current = torch.zeros(1, 12, device=battery.device)
  current[0, 0] = 10.0  # 10 A pack draw.
  mech_power = torch.zeros(1, device=battery.device)
  heat = torch.zeros(1, 12, device=battery.device)

  num_steps = 100
  for _ in range(num_steps):
    battery.step(current, mech_power, heat, DT)

  expected_ah = 10.0 * num_steps * DT / 3600.0
  expected_soc = 1.0 - expected_ah / CFG["capacity_ah"]
  assert torch.allclose(
    battery.soc, torch.tensor([expected_soc], device=battery.device), atol=1e-6
  )
  assert torch.allclose(
    battery.cum_ah, torch.tensor([expected_ah], device=battery.device), atol=1e-6
  )


def test_higher_current_lowers_bus_voltage_at_fixed_soc():
  low_i = make_battery(num_envs=1)
  high_i = make_battery(num_envs=1)
  mech_power = torch.zeros(1, device=low_i.device)
  heat = torch.zeros(1, 12, device=low_i.device)

  low_current = torch.full((1, 12), 1.0 / 12, device=low_i.device)
  high_current = torch.full((1, 12), 10.0 / 12, device=high_i.device)

  _, low_bus_v, _ = low_i.step(low_current, mech_power, heat, DT)
  _, high_bus_v, _ = high_i.step(high_current, mech_power, heat, DT)

  assert high_bus_v.item() < low_bus_v.item()


def test_mech_power_adds_to_energy_without_double_clamping():
  """Regen isn't credited: callers pass an already-clamped mech_power, and
  step() must not re-derive or re-clamp it from anything else."""
  battery = make_battery(num_envs=1)
  current = torch.zeros(1, 12, device=battery.device)
  heat = torch.zeros(1, 12, device=battery.device)

  positive_power = torch.tensor([50.0], device=battery.device)
  energy, _, _ = battery.step(current, positive_power, heat, DT)
  expected_wh = 50.0 * DT / 3600.0
  assert torch.allclose(energy, torch.tensor([expected_wh], device=battery.device))


def test_cum_wh_accumulates_across_steps():
  """cum_wh is what the live-monitoring cumulative-energy metric reads."""
  battery = make_battery(num_envs=1)
  current = torch.zeros(1, 12, device=battery.device)
  heat = torch.zeros(1, 12, device=battery.device)
  power = torch.tensor([50.0], device=battery.device)

  for _ in range(10):
    battery.step(current, power, heat, DT)

  expected_cum_wh = 50.0 * DT / 3600.0 * 10
  assert torch.allclose(
    battery.cum_wh, torch.tensor([expected_cum_wh], device=battery.device), atol=1e-9
  )

  battery.reset(env_ids=torch.tensor([0], device=battery.device))
  assert battery.cum_wh[0].item() == 0.0


def test_reset_only_affects_indexed_envs():
  battery = make_battery(num_envs=4)
  current = torch.full((4, 12), 5.0 / 12, device=battery.device)
  mech_power = torch.zeros(4, device=battery.device)
  heat = torch.zeros(4, 12, device=battery.device)

  for _ in range(20):
    battery.step(current, mech_power, heat, DT)

  drained_soc = battery.soc.clone()
  battery.reset(env_ids=torch.tensor([1, 3], device=battery.device))

  assert torch.allclose(battery.soc[0], drained_soc[0])
  assert torch.allclose(battery.soc[2], drained_soc[2])
  assert battery.soc[1].item() == 1.0
  assert battery.soc[3].item() == 1.0
