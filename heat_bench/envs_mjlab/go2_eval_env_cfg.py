"""Passive thermal/battery evaluation config, built on the Go1 rough-terrain task.

mjlab has no Go2 MJCF asset yet, so this config simulates the existing Go1
robot body while modeling Go2's electro-thermal hardware constants (see
``heat_bench/configs/go2_eval_config.yaml``). It adds a read-only "thermal"
observation group and episode-summary metrics on top of the unmodified Go1
rough-terrain task; no reward or action terms are touched.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from heat_bench.envs_mjlab.eval_observations import (
  ThermalEnergyObservation,
  battery_cumulative_energy_wh,
  battery_energy_wh_step,
  battery_soc,
  thermal_max_joint_temp,
  thermal_mean_joint_temp,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_rough_env_cfg

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "go2_eval_config.yaml"


def load_heat_bench_config() -> dict:
  with open(CONFIG_PATH) as f:
    return yaml.safe_load(f)


def distance_traveled(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Per-step distance delta. Reduced with ``sum`` for the episode-end total."""
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, :2].norm(dim=-1) * env.step_dt


class CumulativeDistanceTraveled:
  """Running distance total since the last reset, for live monitoring.

  A plain ``sum``-reduce metric (like ``distance_traveled`` above) only
  exposes its running total at episode end, not per step -- this keeps its
  own persistent buffer so an interactive viewer can plot the trend live."""

  def __init__(self, cfg: MetricsTermCfg, env):
    del cfg
    self._cum_distance = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids) -> None:
    self._cum_distance[env_ids] = 0.0

  def __call__(self, env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    step_distance = asset.data.root_link_lin_vel_b[:, :2].norm(dim=-1) * env.step_dt
    self._cum_distance += step_distance
    return self._cum_distance


def go2_eval_env_cfg(play: bool = True) -> ManagerBasedRlEnvCfg:
  """Create the passive Go2-hardware-modeled thermal/battery eval config."""
  cfg = unitree_go1_rough_env_cfg(play=play)
  hb_cfg = load_heat_bench_config()
  robot_asset_cfg = SceneEntityCfg("robot", joint_names=(".*",))

  cfg.observations["thermal"] = ObservationGroupCfg(
    terms={
      "thermal_energy": ObservationTermCfg(
        func=ThermalEnergyObservation,
        params={"asset_cfg": robot_asset_cfg, "config": hb_cfg},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.metrics["thermal_max_joint_temp"] = MetricsTermCfg(
    func=thermal_max_joint_temp,
    reduce="max",
    params={"obs_group": "thermal", "obs_term": "thermal_energy"},
  )
  cfg.metrics["thermal_mean_joint_temp"] = MetricsTermCfg(
    func=thermal_mean_joint_temp,
    reduce="mean",
    params={"obs_group": "thermal", "obs_term": "thermal_energy"},
  )
  cfg.metrics["battery_energy_wh"] = MetricsTermCfg(
    func=battery_energy_wh_step,
    reduce="sum",
    params={"obs_group": "thermal", "obs_term": "thermal_energy"},
  )
  cfg.metrics["battery_final_soc"] = MetricsTermCfg(
    func=battery_soc,
    reduce="last",
    params={"obs_group": "thermal", "obs_term": "thermal_energy"},
  )
  cfg.metrics["distance_traveled"] = MetricsTermCfg(
    func=distance_traveled,
    reduce="sum",
    params={"asset_cfg": robot_asset_cfg},
  )
  cfg.metrics["battery_cumulative_energy_wh"] = MetricsTermCfg(
    func=battery_cumulative_energy_wh,
    reduce="last",
    params={"obs_group": "thermal", "obs_term": "thermal_energy"},
  )
  cfg.metrics["distance_traveled_cumulative"] = MetricsTermCfg(
    func=CumulativeDistanceTraveled,
    reduce="last",
    params={"asset_cfg": robot_asset_cfg},
  )

  return cfg
