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
  battery_capacity_loss_pct,
  battery_cumulative_energy_wh,
  battery_energy_wh_step,
  battery_soc,
  thermal_max_joint_temp,
  thermal_mean_joint_temp,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
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


def go2_eval_env_cfg(
  play: bool = True, battery_model: str | None = None
) -> ManagerBasedRlEnvCfg:
  """Create the passive Go2-hardware-modeled thermal/battery eval config.

  Args:
    play: Forwarded to unitree_go1_rough_env_cfg.
    battery_model: If given, overrides the yaml's `battery.model` ("rint"
      or "rint_soc_aging") -- lets scripts A/B compare battery models
      without editing configs/go2_eval_config.yaml.
  """
  cfg = unitree_go1_rough_env_cfg(play=play)
  hb_cfg = load_heat_bench_config()
  if battery_model is not None:
    hb_cfg["battery"]["model"] = battery_model
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
  cfg.metrics["battery_capacity_loss_pct"] = MetricsTermCfg(
    func=battery_capacity_loss_pct,
    reduce="last",
    params={"obs_group": "thermal", "obs_term": "thermal_energy"},
  )

  payload_cfg = hb_cfg.get("payload", {})
  if payload_cfg.get("enabled", False):
    cfg.events["payload_mass"] = EventTermCfg(
      func=dr.pseudo_inertia,
      mode=payload_cfg.get("mode", "reset"),
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
        "alpha_range": tuple(payload_cfg.get("alpha_range", (0.0, 0.0))),
      },
    )

  return cfg


def add_push_disturbance(cfg: ManagerBasedRlEnvCfg) -> None:
  """Re-add the standard velocity-kick push event, mutating ``cfg`` in place.

  ``unitree_go1_rough_env_cfg(play=True)`` strips "push_robot"
  (src/mjlab/tasks/velocity/config/go1/env_cfgs.py) so interactive viewing
  is deterministic by default. Call this after ``go2_eval_env_cfg(play=True)``
  to opt back into it for watching how the tracked thermal/battery/torque
  state reacts to an external disturbance -- same params mjlab's own
  training config uses (src/mjlab/tasks/velocity/velocity_env_cfg.py).
  """
  cfg.events["push_robot"] = EventTermCfg(
    func=envs_mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={
      "velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
      },
    },
  )


def add_impulse_disturbance(
  cfg: ManagerBasedRlEnvCfg, hb_cfg: dict | None = None
) -> None:
  """Add a real force-based push/kick event, mutating ``cfg`` in place.

  Unlike ``add_push_disturbance`` (an instantaneous, mass-independent qvel
  overwrite -- no force is ever computed), this uses
  ``mjlab.envs.mdp.apply_body_impulse``: a real force+torque wrench written
  to ``xfrc_applied`` and held for a sampled duration, so it respects the
  robot's own mass/inertia/contacts and renders as a visible arrow in the
  Viser/native debug-vis overlay (``apply_body_impulse.debug_vis``,
  auto-discovered by ``EventManager.debug_vis``, no extra viewer code
  needed). Force/torque/duration/cooldown magnitudes come from
  ``configs/go2_eval_config.yaml``'s ``impulse_disturbance`` section.
  """
  hb_cfg = hb_cfg if hb_cfg is not None else load_heat_bench_config()
  impulse_cfg = hb_cfg["impulse_disturbance"]
  cfg.events["impulse_disturbance"] = EventTermCfg(
    func=envs_mdp.apply_body_impulse,
    mode="step",
    params={
      "force_range": tuple(impulse_cfg["force_range"]),
      "torque_range": tuple(impulse_cfg["torque_range"]),
      "duration_s": tuple(impulse_cfg["duration_s"]),
      "cooldown_s": tuple(impulse_cfg["cooldown_s"]),
      "asset_cfg": SceneEntityCfg("robot", body_names=tuple(impulse_cfg["body_names"])),
    },
  )
