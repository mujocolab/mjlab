"""Tests for ThermalEnergyObservation and its go2_eval_env_cfg wiring."""

import importlib
import os
from unittest.mock import Mock

import torch
from heat_bench.envs_mjlab.eval_observations import ThermalEnergyObservation
from heat_bench.envs_mjlab.go2_eval_env_cfg import (
  CumulativeDistanceTraveled,
  add_impulse_disturbance,
  add_push_disturbance,
  go2_eval_env_cfg,
  load_heat_bench_config,
)
from heat_bench.physics.battery_ecm import AdvancedBatteryECM

from mjlab.asset_zoo.robots import get_go1_robot_cfg
from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_rough_env_cfg


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


def test_thermal_group_and_metrics_added_without_touching_actor():
  """The eval cfg adds a thermal group/metrics but leaves actor/critic obs
  and rewards exactly as the base Go1 task defines them -- a policy trained
  on the base task must see an identical actor observation width."""
  base_cfg = unitree_go1_rough_env_cfg(play=True)
  eval_cfg = go2_eval_env_cfg(play=True)

  assert set(eval_cfg.observations["actor"].terms) == set(
    base_cfg.observations["actor"].terms
  )
  assert set(eval_cfg.observations["critic"].terms) == set(
    base_cfg.observations["critic"].terms
  )
  assert set(eval_cfg.rewards) == set(base_cfg.rewards)

  assert "thermal" in eval_cfg.observations
  assert "thermal_energy" in eval_cfg.observations["thermal"].terms

  expected_metrics = {
    "thermal_max_joint_temp",
    "thermal_mean_joint_temp",
    "battery_energy_wh",
    "battery_final_soc",
    "distance_traveled",
    "battery_cumulative_energy_wh",
    "distance_traveled_cumulative",
    "battery_capacity_loss_pct",
  }
  assert expected_metrics <= set(eval_cfg.metrics)


def test_payload_event_absent_by_default():
  """Payload is opt-in via config -- disabled by default must not add an
  event, so existing eval/play runs are unaffected."""
  eval_cfg = go2_eval_env_cfg(play=True)
  assert "payload_mass" not in eval_cfg.events


def test_push_disturbance_absent_unless_added():
  """play=True strips push_robot for deterministic viewing by default;
  add_push_disturbance() opts back in without touching go2_eval_env_cfg()."""
  eval_cfg = go2_eval_env_cfg(play=True)
  assert "push_robot" not in eval_cfg.events

  add_push_disturbance(eval_cfg)
  assert "push_robot" in eval_cfg.events
  assert eval_cfg.events["push_robot"].mode == "interval"


def test_impulse_disturbance_absent_unless_added():
  """apply_body_impulse is force-based (writes xfrc_applied) and holds for
  a sampled duration -- distinct from add_push_disturbance's velocity kick."""
  eval_cfg = go2_eval_env_cfg(play=True)
  assert "impulse_disturbance" not in eval_cfg.events

  add_impulse_disturbance(eval_cfg)
  assert "impulse_disturbance" in eval_cfg.events
  event = eval_cfg.events["impulse_disturbance"]
  assert event.mode == "step"
  assert event.params["force_range"] == (-125.0, 125.0)
  assert event.params["asset_cfg"].body_names == ("trunk",)


def test_go2_eval_env_cfg_battery_model_override():
  """battery_model= lets scripts (e.g. play.py's --battery-model) A/B
  compare models without editing configs/go2_eval_config.yaml."""
  default_cfg = go2_eval_env_cfg(play=True)
  default_params = default_cfg.observations["thermal"].terms["thermal_energy"].params
  assert default_params["config"]["battery"]["model"] == "rint"

  overridden_cfg = go2_eval_env_cfg(play=True, battery_model="rint_soc_aging")
  overridden_params = (
    overridden_cfg.observations["thermal"].terms["thermal_energy"].params
  )
  assert overridden_params["config"]["battery"]["model"] == "rint_soc_aging"


def test_payload_event_registered_when_enabled(monkeypatch):
  hb_cfg = load_heat_bench_config()
  hb_cfg["payload"] = {
    "enabled": True,
    "alpha_range": [0.1, 0.3],
    "mode": "startup",
  }
  # `heat_bench.envs_mjlab.__init__` re-exports a function also named
  # `go2_eval_env_cfg`, shadowing the submodule attribute of the same name
  # on the package -- `import heat_bench.envs_mjlab.go2_eval_env_cfg as x`
  # would silently bind the function, not the module. importlib.import_module
  # goes through sys.modules directly and isn't affected.
  go2_eval_env_cfg_module = importlib.import_module(
    "heat_bench.envs_mjlab.go2_eval_env_cfg"
  )
  monkeypatch.setattr(go2_eval_env_cfg_module, "load_heat_bench_config", lambda: hb_cfg)

  eval_cfg = go2_eval_env_cfg(play=True)

  assert "payload_mass" in eval_cfg.events
  event = eval_cfg.events["payload_mass"]
  assert event.mode == "startup"
  assert event.params["alpha_range"] == (0.1, 0.3)
  assert event.params["asset_cfg"].body_names == ("trunk",)


def _build_go1_entity(device: str, num_envs: int) -> tuple[Entity, Simulation]:
  entity = Entity(get_go1_robot_cfg())
  model = entity.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


def test_call_steps_engines_once_and_returns_expected_shape():
  device = get_test_device()
  num_envs = 2
  entity, sim = _build_go1_entity(device, num_envs)

  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.step_dt = 0.02
  env.scene = {"robot": entity}

  asset_cfg = SceneEntityCfg(name="robot", joint_names=(".*",))
  term_cfg = ObservationTermCfg(
    func=ThermalEnergyObservation,
    params={"asset_cfg": asset_cfg, "config": load_heat_bench_config()},
  )
  term = ThermalEnergyObservation(term_cfg, env)
  assert len(term._joint_ids) == 12

  sim.data.qfrc_actuator[:] = 5.0
  sim.data.qvel[:] = 1.0

  obs = term(env, asset_cfg)
  assert obs.shape == (num_envs, 14)

  temps_after_one_step = term.thermal.T.clone()
  soc_after_one_step = term.battery.soc.clone()
  assert (soc_after_one_step < 1.0).all()

  obs2 = term(env, asset_cfg)
  assert obs2.shape == (num_envs, 14)
  assert not torch.allclose(term.thermal.T, temps_after_one_step)

  term.reset(env_ids=torch.tensor([0], device=device))
  assert term.battery.soc[0].item() == 1.0
  assert term.battery.soc[1].item() != 1.0


def test_battery_model_switch_via_config():
  """battery.model in go2_eval_config.yaml selects which battery_ecm class
  ThermalEnergyObservation instantiates -- same call signature either way,
  so this shouldn't require any branching in the observation term itself."""
  device = get_test_device()
  num_envs = 2
  entity, sim = _build_go1_entity(device, num_envs)

  hb_cfg = load_heat_bench_config()
  hb_cfg["battery"]["model"] = "rint_soc_aging"

  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.step_dt = 0.02
  env.scene = {"robot": entity}

  asset_cfg = SceneEntityCfg(name="robot", joint_names=(".*",))
  term_cfg = ObservationTermCfg(
    func=ThermalEnergyObservation,
    params={"asset_cfg": asset_cfg, "config": hb_cfg},
  )
  term = ThermalEnergyObservation(term_cfg, env)
  assert isinstance(term.battery, AdvancedBatteryECM)

  sim.data.qfrc_actuator[:] = 20.0
  sim.data.qvel[:] = 5.0
  for _ in range(30):
    obs = term(env, asset_cfg)

  assert obs.shape == (num_envs, 14)
  assert (term.last_capacity_loss_pct >= 0).all()


def test_substep_heat_is_averaged_not_last_sample():
  """Joule heat must be the mean over physics substeps, not a sample of
  whichever substep happened to run last -- see the module docstring in
  eval_observations.py for why (foot-impact torque spikes on rough terrain
  can be 3x the post-decimation sample at the tail)."""
  device = get_test_device()
  num_envs = 1
  entity, sim = _build_go1_entity(device, num_envs)

  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.step_dt = 0.02
  env.scene = {"robot": entity}
  env.sim = Mock()
  # A no-op step: real physics would recompute qfrc_actuator from ctrl on
  # every step(), overwriting the values this test sets manually.
  env.sim.step = lambda: None

  asset_cfg = SceneEntityCfg(name="robot", joint_names=(".*",))
  term_cfg = ObservationTermCfg(
    func=ThermalEnergyObservation,
    params={"asset_cfg": asset_cfg, "config": load_heat_bench_config()},
  )
  term = ThermalEnergyObservation(term_cfg, env)
  assert env.sim.step == term._accumulate_substep  # Bound methods, not `is`.

  sim.data.qvel[:] = 0.0  # Isolate the check to heat; no mechanical power.

  # entity.data.qfrc_actuator[:, term._joint_ids] indexes into raw sim DOF
  # space via entity.indexing.joint_v_adr (the free-base root joint occupies
  # the first several DOFs), so writing sim.data.qfrc_actuator directly
  # requires the corresponding raw DOF column, not joint index 0.
  raw_dof_col = entity.indexing.joint_v_adr[term._joint_ids[0]].item()

  # Three "physics substeps" on the first actuated joint: two quiet, one
  # foot-impact-style spike, mimicking what the smoke-test measurement
  # found on rough terrain.
  torques = [1.0, 1.0, 10.0]
  for tau in torques:
    sim.data.qfrc_actuator[:] = 0.0
    sim.data.qfrc_actuator[:, raw_dof_col] = tau
    term._accumulate_substep()

  obs = term(env, asset_cfg)
  assert obs.shape == (num_envs, 14)
  assert term._substep_count == 0  # Drained after __call__.

  assert term.last_current.shape == (num_envs, 12)
  assert term.last_torque.shape == (num_envs, 12)
  expected_mean_torque = sum(torques) / len(torques)
  assert torch.allclose(
    term.last_torque[0, 0],
    torch.tensor(expected_mean_torque, device=device),
    rtol=1e-4,
  )

  cfg = load_heat_bench_config()["thermal"]
  gear_ratio_n, kt, rd = (
    cfg["gear_ratio_N"],
    cfg["motor_torque_constant_Kt"],
    cfg["phase_resistance_Rd"],
  )
  heat_per_substep = [(tau / (gear_ratio_n * kt)) ** 2 * rd for tau in torques]
  expected_mean_heat = sum(heat_per_substep) / len(heat_per_substep)
  last_sample_heat = heat_per_substep[-1]
  assert expected_mean_heat < last_sample_heat  # The spike dominates I^2.

  # mech_power is 0 (qvel=0), so energy_wh_step is exactly heat*dt/3600 --
  # an exact check that averaging, not last-sample, produced this value.
  expected_energy_wh = expected_mean_heat * env.step_dt / 3600.0
  last_sample_energy_wh = last_sample_heat * env.step_dt / 3600.0
  assert torch.allclose(
    term.last_energy_wh_step,
    torch.tensor([expected_energy_wh], device=device),
    rtol=1e-4,
  )
  assert not torch.allclose(
    term.last_energy_wh_step,
    torch.tensor([last_sample_energy_wh], device=device),
    rtol=1e-4,
  )


def test_cumulative_distance_traveled_accumulates_and_resets():
  device = get_test_device()
  num_envs = 2

  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.step_dt = 0.02

  asset = Mock()
  asset.data.root_link_lin_vel_b = torch.tensor(
    [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], device=device
  )  # env 0: speed 5 m/s in xy; env 1: stationary.
  env.scene = {"robot": asset}

  asset_cfg = SceneEntityCfg(name="robot")
  dummy_cfg = MetricsTermCfg(func=CumulativeDistanceTraveled)
  metric = CumulativeDistanceTraveled(cfg=dummy_cfg, env=env)

  for _ in range(10):
    total = metric(env, asset_cfg)

  expected = torch.tensor([5.0 * env.step_dt * 10, 0.0], device=device)
  assert torch.allclose(total, expected, atol=1e-5)

  metric.reset(env_ids=torch.tensor([0], device=device))
  assert metric._cum_distance[0].item() == 0.0
  assert metric._cum_distance[1].item() == 0.0
