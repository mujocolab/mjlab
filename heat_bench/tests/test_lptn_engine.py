"""Pure-tensor tests for BatchedLPTNEngine (no simulation required)."""

import os

import torch
from heat_bench.physics.lptn_engine import AMBIENT_IDX, CHASSIS_IDX, BatchedLPTNEngine


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


CFG = {
  "gear_ratio_N": 6.22,
  "motor_torque_constant_Kt": 0.26,
  "phase_resistance_Rd": 0.66,
  "joint_thermal_capacitance_Cth": 400.0,
  "joint_thermal_resistance_Rth": 2.0,
  "chassis_thermal_capacitance_Cth_chassis": 2000.0,
  "chassis_convection_base_conductance": 0.8,
  "chassis_convection_velocity_gain": 1.5,
  "ambient_temperature_c": 25.0,
  "initial_joint_temperature_c": 25.0,
  "initial_chassis_temperature_c": 25.0,
}
DT = 0.02


def make_engine(num_envs: int) -> BatchedLPTNEngine:
  return BatchedLPTNEngine(CFG, num_envs, get_test_device())


def test_zero_heat_zero_velocity_stays_at_ambient():
  engine = make_engine(num_envs=2)
  heat = torch.zeros(2, 12, device=engine.device)
  vel = torch.zeros(2, 2, device=engine.device)
  for _ in range(50):
    engine.step(heat, vel, DT)
  assert torch.allclose(engine.T, torch.full_like(engine.T, 25.0), atol=1e-4)


def test_constant_heat_raises_joint_and_chassis_temp():
  engine = make_engine(num_envs=1)
  heat = torch.zeros(1, 12, device=engine.device)
  heat[0, 0] = 50.0
  vel = torch.zeros(1, 2, device=engine.device)

  prev_joint = engine.T[0, 0].item()
  prev_chassis = engine.T[0, CHASSIS_IDX].item()
  for _ in range(200):
    engine.step(heat, vel, DT)
    joint_t = engine.T[0, 0].item()
    chassis_t = engine.T[0, CHASSIS_IDX].item()
    assert joint_t >= prev_joint
    assert chassis_t >= prev_chassis
    assert engine.T[0, AMBIENT_IDX].item() == 25.0
    prev_joint, prev_chassis = joint_t, chassis_t

  assert prev_joint > 25.0
  assert prev_chassis > 25.0


def test_higher_velocity_lowers_steady_state_chassis_temp():
  low_v = make_engine(num_envs=1)
  high_v = make_engine(num_envs=1)
  heat = torch.zeros(1, 12, device=low_v.device)
  heat[0, :] = 5.0

  for _ in range(3000):
    low_v.step(heat, torch.zeros(1, 2, device=low_v.device), DT)
    high_v.step(heat, torch.full((1, 2), 1.0, device=high_v.device), DT)

  assert high_v.T[0, CHASSIS_IDX].item() < low_v.T[0, CHASSIS_IDX].item()


def test_reset_only_affects_indexed_envs():
  engine = make_engine(num_envs=4)
  heat = torch.full((4, 12), 20.0, device=engine.device)
  vel = torch.zeros(4, 2, device=engine.device)
  for _ in range(20):
    engine.step(heat, vel, DT)

  heated = engine.T.clone()
  engine.reset(env_ids=torch.tensor([1, 3], device=engine.device))

  assert torch.allclose(engine.T[0], heated[0])
  assert torch.allclose(engine.T[2], heated[2])
  assert engine.T[1, 0].item() == 25.0
  assert engine.T[3, 0].item() == 25.0
