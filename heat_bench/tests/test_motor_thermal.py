"""Pure-tensor tests for the Rd(T)/Kt(T) motor thermal model."""

import pytest
import torch
from heat_bench.envs_mjlab.go2_eval_env_cfg import load_heat_bench_config
from heat_bench.physics.motor_thermal import MotorThermalModel


@pytest.fixture
def model() -> MotorThermalModel:
  return MotorThermalModel.from_config(load_heat_bench_config()["thermal"])


def test_spec_values_at_spec_reference_temperature(model):
  t = torch.tensor([model.spec_ref_c])
  torch.testing.assert_close(model.phase_resistance(t), torch.tensor([model.rd_spec]))
  torch.testing.assert_close(model.torque_constant(t), torch.tensor([model.kt_spec]))


def test_known_ratios_at_80c(model):
  t = torch.tensor([80.0])
  # (1 + a*(80-20)) / (1 + a*(25-20)) with the cited coefficients.
  rd_ratio = (1 + 0.00393 * 60) / (1 + 0.00393 * 5)
  kt_ratio = (1 - 0.0012 * 60) / (1 - 0.0012 * 5)
  torch.testing.assert_close(
    model.phase_resistance(t) / model.rd_spec, t * 0 + rd_ratio
  )
  torch.testing.assert_close(model.torque_constant(t) / model.kt_spec, t * 0 + kt_ratio)
  assert 1.21 < rd_ratio < 1.22
  assert 0.93 < kt_ratio < 0.94


def test_zero_coefficients_give_constants():
  thermal_cfg = load_heat_bench_config()["thermal"]
  thermal_cfg["phase_resistance_temp_coeff_per_c"] = 0.0
  thermal_cfg["torque_constant_temp_coeff_per_c"] = 0.0
  model = MotorThermalModel.from_config(thermal_cfg)
  t = torch.tensor([0.0, 25.0, 150.0])
  torch.testing.assert_close(model.phase_resistance(t), torch.full((3,), model.rd_spec))
  torch.testing.assert_close(model.torque_constant(t), torch.full((3,), model.kt_spec))


def test_voltage_torque_bounds_shrink_with_speed_and_voltage(model):
  n = 6.22
  t = torch.full((1, 2), model.spec_ref_c)  # Kt/Rd at their spec values.
  v = torch.tensor([33.6])
  stall = n * model.kt_spec * 33.6 / model.rd_spec

  lo, hi = model.voltage_torque_bounds(t, torch.zeros(1, 2), v, n)
  torch.testing.assert_close(hi, torch.full((1, 2), stall))
  torch.testing.assert_close(lo, torch.full((1, 2), -stall))

  # Moving forward at 10 rad/s: back-EMF eats into forward torque but adds
  # to braking (reverse) torque by the same amount.
  qd = torch.tensor([[10.0, 0.0]])
  lo, hi = model.voltage_torque_bounds(t, qd, v, n)
  emf_torque = n * model.kt_spec * (model.kt_spec * n * 10.0) / model.rd_spec
  torch.testing.assert_close(hi[0, 0], torch.tensor(stall - emf_torque))
  torch.testing.assert_close(lo[0, 0], torch.tensor(-stall - emf_torque))

  # A depleted pack lowers the forward bound at the same speed.
  _, hi_low = model.voltage_torque_bounds(t, qd, torch.tensor([24.0]), n)
  assert hi_low[0, 0] < hi[0, 0]
