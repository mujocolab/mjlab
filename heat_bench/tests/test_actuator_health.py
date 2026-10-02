"""Tests for the Phase 0 apply_actuator_health event scaffold."""

import importlib
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from heat_bench.envs_mjlab.actuator_health import (
  ActuatorState,
  apply_actuator_health,
  scripted_joint_fault,
)
from heat_bench.envs_mjlab.go2_eval_env_cfg import (
  go2_eval_env_cfg,
  load_heat_bench_config,
)

from mjlab.asset_zoo.robots import get_go1_robot_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


def _make_term(device: str, num_envs: int):
  entity = Entity(get_go1_robot_cfg())
  model = entity.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  sim.expand_model_fields(apply_actuator_health.model_fields)
  entity.initialize(model, sim.model, sim.data, device)

  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.scene = {"robot": entity}
  env.sim = sim

  params = {
    "asset_cfg": SceneEntityCfg(name="robot", joint_names=(".*",)),
    "obs_group": "thermal",
    "obs_term": "thermal_energy",
    "config": load_heat_bench_config(),
  }
  # Stand-in for ThermalEnergyObservation: joints at the Rd/Kt spec
  # reference temperature, i.e. no thermal derate.
  env.thermal_stub = SimpleNamespace(
    last_joint_temps=torch.full((num_envs, 12), 25.0, device=device)
  )
  env.observation_manager.get_term_cfg.return_value = Mock(func=env.thermal_stub)
  term_cfg = EventTermCfg(func=apply_actuator_health, mode="step", params=params)
  return apply_actuator_health(term_cfg, env), env, entity, params


def test_event_registered_by_default():
  cfg = go2_eval_env_cfg(play=True)
  event = cfg.events["actuator_health"]
  assert event.func is apply_actuator_health
  assert event.mode == "step"
  assert event.params["obs_group"] == "thermal"
  assert event.params["obs_term"] == "thermal_energy"


def test_event_absent_when_disabled(monkeypatch):
  hb_cfg = load_heat_bench_config()
  hb_cfg["actuator_health"] = {"enabled": False}
  # See test_payload_event_registered_when_enabled for why importlib is used.
  module = importlib.import_module("heat_bench.envs_mjlab.go2_eval_env_cfg")
  monkeypatch.setattr(module, "load_heat_bench_config", lambda: hb_cfg)
  assert "actuator_health" not in go2_eval_env_cfg(play=True).events


def test_ctrl_ids_map_each_joint_to_its_actuator():
  term, _, entity, _ = _make_term(get_test_device(), num_envs=2)
  assert len(term.ctrl_ids) == 12
  for joint_name, ctrl_id in zip(term.joint_names, term.ctrl_ids.tolist(), strict=True):
    actuator = next(a for a in entity.actuators if joint_name in a.target_names)
    idx = actuator.target_names.index(joint_name)
    assert actuator.global_ctrl_ids[idx].item() == ctrl_id


def test_buffers_init_healthy_and_reset_only_selected_envs():
  device = get_test_device()
  term, env, _, params = _make_term(device, num_envs=2)
  assert term.derate.shape == (2, 12)
  assert term.state.shape == (2, 12)
  assert (term.derate == 1.0).all()
  assert (term.state == ActuatorState.HEALTHY).all()

  # Simulate a reset-mode DR event halving env 1's limits, plus some
  # (future-phase) degraded state on both envs.
  forcerange = env.sim.model.actuator_forcerange
  forcerange[1, term.ctrl_ids] *= 0.5
  term.derate[:] = 0.3
  term.state[:] = ActuatorState.DEAD

  term.reset(torch.tensor([1], device=device))
  assert (term.derate[0] == 0.3).all()
  assert (term.state[0] == ActuatorState.DEAD).all()
  assert (term.derate[1] == 1.0).all()
  assert (term.state[1] == ActuatorState.HEALTHY).all()
  torch.testing.assert_close(term.baseline_forcerange[1], forcerange[1, term.ctrl_ids])

  # Identity write preserves the DR'd limits rather than clobbering them.
  term.reset(None)
  before = forcerange.clone()
  term(env, None, **params)
  torch.testing.assert_close(forcerange[:], before)


@pytest.mark.slow
def test_identity_write_in_real_env():
  cfg = go2_eval_env_cfg(play=True)
  cfg.scene.num_envs = 2
  # Zero temperature coefficients -> thermal_derate stays 1, so the write
  # must be an exact identity. (The hb_cfg dict is shared with the obs term.)
  thermal_cfg = cfg.events["actuator_health"].params["config"]["thermal"]
  thermal_cfg["torque_constant_temp_coeff_per_c"] = 0.0
  thermal_cfg["phase_resistance_temp_coeff_per_c"] = 0.0
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    term = env.event_manager.get_term_cfg("actuator_health").func
    before = env.sim.model.actuator_forcerange.clone()
    actions = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(3):
      env.step(actions)
    torch.testing.assert_close(env.sim.model.actuator_forcerange[:], before)
    assert (term.state == ActuatorState.HEALTHY).all()
    # The step event actually ran and resolved the thermal observation term.
    assert (
      term._thermal_term
      is env.observation_manager.get_term_cfg("thermal", "thermal_energy").func
    )

    term.derate[:] = 0.5
    term.state[:] = ActuatorState.DERATED
    env.reset()
    assert (term.derate == 1.0).all()
    assert (term.state == ActuatorState.HEALTHY).all()
  finally:
    env.close()


def test_reset_restores_joint_degraded_last_episode():
  """A joint derated/killed in one episode gets its full limit back after
  reset, rather than the degraded write being re-snapshotted as baseline."""
  device = get_test_device()
  term, env, _, params = _make_term(device, num_envs=2)
  forcerange = env.sim.model.actuator_forcerange
  full = forcerange[:, term.ctrl_ids].clone()

  term.derate[:, 0] = 0.0
  term(env, None, **params)
  assert (forcerange[:, term.ctrl_ids[0]] == 0.0).all()

  term.reset(torch.tensor([0], device=device))
  # Restored immediately, before the next step event runs.
  torch.testing.assert_close(forcerange[0, term.ctrl_ids], full[0])
  term(env, None, **params)
  torch.testing.assert_close(forcerange[0, term.ctrl_ids], full[0])
  assert (forcerange[1, term.ctrl_ids[0]] == 0.0).all()


def test_scripted_fault_follows_schedule_on_selected_joints_only():
  device = get_test_device()
  health, env, _, _ = _make_term(device, num_envs=3)
  env.event_manager.get_term_cfg.return_value = Mock(func=health)
  env.step_dt = 0.5
  joint_names = ("FR_calf_joint", "RL_calf_joint")
  params: dict[str, Any] = {
    "joint_names": joint_names,
    "health_term": "actuator_health",
    "derate_start_s": 1.0,
    "derate_end_s": 2.0,
    "derate_floor": 0.3,
    "dead_at_s": 3.0,
  }
  fault = scripted_joint_fault(
    EventTermCfg(func=scripted_joint_fault, mode="step", params=params), env
  )
  # Per-env episode times 0.5s (healthy), 1.5s (mid-ramp), 3.0s (dead).
  env.episode_length_buf = torch.tensor([1, 3, 6], device=device)
  fault(env, None, **params)

  idx = [health.joint_names.index(n) for n in joint_names]
  expected = torch.tensor([1.0, 0.65, 0.0], device=device)
  for j in idx:
    torch.testing.assert_close(health.derate[:, j], expected)
  assert health.state[:, idx].tolist() == [
    [ActuatorState.HEALTHY] * 2,
    [ActuatorState.DERATED] * 2,
    [ActuatorState.DEAD] * 2,
  ]
  others = [j for j in range(12) if j not in idx]
  assert (health.derate[:, others] == 1.0).all()
  assert (health.state[:, others] == ActuatorState.HEALTHY).all()


def test_hot_joint_torque_limit_follows_kt_fade_and_composes_with_derate():
  device = get_test_device()
  term, env, _, params = _make_term(device, num_envs=2)
  forcerange = env.sim.model.actuator_forcerange
  full = forcerange[:, term.ctrl_ids].clone()
  kt_ratio = (1 - 0.0012 * 60) / (1 - 0.0012 * 5)

  env.thermal_stub.last_joint_temps[0, 0] = 80.0  # One hot joint.
  env.thermal_stub.last_joint_temps[1] = 0.0  # Cold magnets.
  term.derate[0, 0] = 0.5  # Scripted fault on the same joint.
  term(env, None, **params)

  torch.testing.assert_close(term.thermal_derate[0, 0].item(), kt_ratio)
  torch.testing.assert_close(
    forcerange[0, term.ctrl_ids[0]], full[0, 0] * 0.5 * kt_ratio
  )
  torch.testing.assert_close(forcerange[0, term.ctrl_ids[1:]], full[0, 1:])
  # Colder than spec never raises the limit above the effort limit.
  torch.testing.assert_close(forcerange[1, term.ctrl_ids], full[1])
  # Thermal fade alone is not a fault state.
  assert (term.state == ActuatorState.HEALTHY).all()

  term.reset(torch.tensor([0], device=device))
  assert (term.thermal_derate[0] == 1.0).all()


@pytest.mark.slow
def test_hot_start_derates_torque_limit_in_real_env(monkeypatch):
  hb_cfg = load_heat_bench_config()
  hb_cfg["thermal"]["initial_joint_temperature_c"] = 80.0
  module = importlib.import_module("heat_bench.envs_mjlab.go2_eval_env_cfg")
  monkeypatch.setattr(module, "load_heat_bench_config", lambda: hb_cfg)
  cfg = go2_eval_env_cfg(play=True)
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    term = env.event_manager.get_term_cfg("actuator_health").func
    full = term.baseline_forcerange.clone()
    actions = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(3):
      env.step(actions)
    # Joints barely move from 80C in 3 steps: limit ~= Kt(80)/Kt(25).
    kt_ratio = (1 - 0.0012 * 60) / (1 - 0.0012 * 5)
    limit = env.sim.model.actuator_forcerange[:, term.ctrl_ids]
    torch.testing.assert_close(limit, full * kt_ratio, rtol=1e-3, atol=1e-3)
  finally:
    env.close()
