"""Headless thermal/battery benchmark: run a trained checkpoint, dump results.

Runs one episode per parallel environment (num_envs = number of episodes)
against ``go2_eval_env_cfg``, accumulating per-env thermal/battery summaries
directly from the live ``ThermalEnergyObservation`` instance every step
(``extras["log"]`` from mjlab's MetricsManager only gives a mean over the
envs that happen to reset together in one call, not one row per env -- see
the module docstring in ``heat_bench/envs_mjlab/eval_observations.py``).
Modeled on ``src/mjlab/tasks/tracking/scripts/evaluate.py``.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from heat_bench.envs_mjlab.go2_eval_env_cfg import go2_eval_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

REFERENCE_TASK_ID = "Mjlab-Velocity-Rough-Unitree-Go1"


@dataclass(frozen=True)
class RunEvalConfig:
  """Configuration for a headless heat_bench evaluation run."""

  checkpoint_file: str | None = None
  """Path to a local checkpoint file. Mutually exclusive with wandb_run_path."""
  wandb_run_path: str | None = None
  """W&B run path ('entity/project/run_id') to resolve a checkpoint from."""
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run (e.g. 'model_4000.pt')."""
  num_envs: int = 256
  """Number of parallel environments (= number of episodes evaluated)."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  output_file: str = "heat_bench_results.csv"
  """Where to dump per-episode results. '.csv' or '.json' by suffix."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""


def run_eval(cfg: RunEvalConfig) -> list[dict[str, float]]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # play=False: finite episode_length_s and terrain-bounds termination stay
  # enabled, so episodes actually end. play=True is for the interactive
  # viewer (infinite episode length) and would hang this headless loop.
  env_cfg = go2_eval_env_cfg(play=False)
  agent_cfg = load_rl_cfg(REFERENCE_TASK_ID)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
  elif cfg.wandb_run_path is not None:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    resume_path, _ = get_wandb_checkpoint_path(
      log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
    )
  else:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    resume_path = get_checkpoint_path(log_root_path)
  print(f"[INFO] Loading checkpoint: {resume_path}")

  runner_cls = load_runner_cls(REFERENCE_TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  thermal_term = env.unwrapped.observation_manager.get_term_cfg(
    "thermal", "thermal_energy"
  ).func
  # None when actuator_health is disabled: no joint can die.
  health_term = (
    env.unwrapped.event_manager.get_term_cfg("actuator_health").func
    if "actuator_health" in env.unwrapped.event_manager.active_terms.get("step", [])
    else None
  )

  done_envs = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  max_temp = torch.zeros(cfg.num_envs, device=device)
  sum_temp = torch.zeros(cfg.num_envs, device=device)
  step_count = torch.zeros(cfg.num_envs, device=device)
  energy_wh = torch.zeros(cfg.num_envs, device=device)
  distance_m = torch.zeros(cfg.num_envs, device=device)
  final_soc = torch.ones(cfg.num_envs, device=device)
  dead_joints = torch.zeros(cfg.num_envs, device=device)
  first_death_s = torch.full((cfg.num_envs,), float("nan"), device=device)

  obs = env.get_observations()
  print(f"[INFO] Running {cfg.num_envs} evaluation episodes...")

  step = 0
  while not done_envs.all():
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)

    active = ~done_envs
    vel_xy = env.unwrapped.scene["robot"].data.root_link_lin_vel_b[:, :2]
    max_temp = torch.where(
      active,
      torch.maximum(max_temp, thermal_term.last_joint_temps.max(dim=-1).values),
      max_temp,
    )
    sum_temp = torch.where(
      active, sum_temp + thermal_term.last_joint_temps.mean(dim=-1), sum_temp
    )
    step_count = torch.where(active, step_count + 1, step_count)
    energy_wh = torch.where(
      active, energy_wh + thermal_term.last_energy_wh_step, energy_wh
    )
    distance_m = torch.where(
      active,
      distance_m + vel_xy.norm(dim=-1) * env.unwrapped.step_dt,
      distance_m,
    )
    final_soc = torch.where(active, thermal_term.last_soc, final_soc)
    if health_term is not None:
      # The death latch is monotonic within an episode, so the max is the
      # episode's dead-joint count.
      dead_now = health_term.dead.sum(dim=-1).float()
      first_death_s = torch.where(
        active & (dead_now > 0) & first_death_s.isnan(),
        step_count * env.unwrapped.step_dt,
        first_death_s,
      )
      dead_joints = torch.where(
        active, torch.maximum(dead_joints, dead_now), dead_joints
      )

    newly_done = dones.bool() & ~done_envs
    done_envs = done_envs | newly_done
    if newly_done.any():
      print(
        f"[INFO] {done_envs.sum().item()}/{cfg.num_envs} episodes completed (step {step})"
      )
    step += 1

  safe_steps = step_count.clamp(min=1)
  records = []
  for env_id in range(cfg.num_envs):
    energy = energy_wh[env_id].item()
    dist = distance_m[env_id].item()
    records.append(
      {
        "env_id": env_id,
        "max_joint_temp_c": max_temp[env_id].item(),
        "mean_joint_temp_c": (sum_temp[env_id] / safe_steps[env_id]).item(),
        "energy_wh": energy,
        "distance_m": dist,
        "wh_per_m": energy / dist if dist > 1e-6 else float("nan"),
        "final_soc": final_soc[env_id].item(),
        "dead_joints": int(dead_joints[env_id].item()),
        "first_death_s": first_death_s[env_id].item(),
      }
    )

  output_path = Path(cfg.output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  if output_path.suffix == ".json":
    with open(output_path, "w") as f:
      json.dump(records, f, indent=2)
  else:
    with open(output_path, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
      writer.writeheader()
      writer.writerows(records)
  print(f"[INFO] Results saved to {output_path}")

  env.close()
  return records


def main():
  args = tyro.cli(RunEvalConfig)
  if args.checkpoint_file is None and args.wandb_run_path is None:
    print("Must pass --checkpoint-file or --wandb-run-path.")
    sys.exit(1)
  run_eval(args)


if __name__ == "__main__":
  main()
