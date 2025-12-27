"""Measure environment throughput for regression tracking.

This script measures physics and environment step throughput across canonical tasks
to catch performance regressions in the manager-based API.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import tyro
import wandb

import mjlab.tasks  # noqa: F401 - registers tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg


@dataclass
class BenchmarkResult:
  """Results from a single benchmark run."""

  task: str
  num_envs: int
  num_steps: int
  decimation: int
  physics_sps: float
  env_sps: float
  overhead_pct: float

  def __str__(self) -> str:
    return (
      f"{self.task} (dec={self.decimation}):\n"
      f"  Physics SPS: {self.physics_sps:,.0f}\n"
      f"  Env SPS:     {self.env_sps:,.0f}\n"
      f"  Overhead:    {self.overhead_pct:.1f}%"
    )

  def to_dict(self) -> dict:
    return asdict(self)


@dataclass
class ThroughputConfig:
  """Configuration for throughput benchmarking."""

  num_envs: int = 4096
  """Number of parallel environments."""

  num_steps: int = 200
  """Number of steps to measure (after warmup)."""

  warmup_steps: int = 50
  """Number of warmup steps before measuring."""

  device: str = "cuda:0"
  """Device to run on."""

  tasks: list[str] = field(
    default_factory=lambda: [
      "Mjlab-Velocity-Flat-Unitree-Go1",
      "Mjlab-Tracking-Flat-Unitree-G1",
      "Mjlab-Lift-Cube-Yam",
    ]
  )
  """Tasks to benchmark."""

  tracking_motion: str = "rll_humanoid/wandb-registry-Motions/lafan_cartwheel:latest"
  """W&B artifact path for tracking task motion (entity/project/name:alias)."""

  output_dir: Path | None = None
  """Output directory for JSON results. If None, results are only printed."""


def measure_physics_sps(env: ManagerBasedRlEnv, num_steps: int) -> float:
  """Measure raw physics stepping in env steps per second.

  Runs num_steps worth of physics (i.e., num_steps * decimation sim.step calls)
  and reports throughput in env steps/sec for direct comparison with env.step().
  """
  decimation = env.cfg.decimation
  total_physics_steps = num_steps * decimation

  torch.cuda.synchronize()
  start = time.perf_counter()

  for _ in range(total_physics_steps):
    env.sim.step()

  torch.cuda.synchronize()
  elapsed = time.perf_counter() - start

  # Report in env steps/sec (not physics steps/sec) for fair comparison.
  return (num_steps * env.num_envs) / elapsed


def measure_env_sps(env: ManagerBasedRlEnv, num_steps: int) -> float:
  """Measure full environment step throughput in env steps per second."""
  action_dim = sum(env.action_manager.action_term_dim)
  action = torch.zeros(env.num_envs, action_dim, device=env.device)

  torch.cuda.synchronize()
  start = time.perf_counter()

  for _ in range(num_steps):
    env.step(action)

  torch.cuda.synchronize()
  elapsed = time.perf_counter() - start

  return (num_steps * env.num_envs) / elapsed


def benchmark_task(task: str, cfg: ThroughputConfig) -> BenchmarkResult:
  """Benchmark a single task."""
  print(f"\nBenchmarking {task}...")

  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = cfg.num_envs

  # Handle tracking task motion file.
  if env_cfg.commands is not None:
    motion_cmd = env_cfg.commands.get("motion")
    if isinstance(motion_cmd, MotionCommandCfg):
      api = wandb.Api()
      artifact = api.artifact(cfg.tracking_motion)
      motion_dir = artifact.download()
      motion_cmd.motion_file = str(Path(motion_dir) / "motion.npz")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env.reset()

  # Warmup.
  action_dim = sum(env.action_manager.action_term_dim)
  action = torch.zeros(env.num_envs, action_dim, device=env.device)
  for _ in range(cfg.warmup_steps):
    env.step(action)
  torch.cuda.synchronize()

  decimation = env.cfg.decimation
  physics_sps = measure_physics_sps(env, cfg.num_steps)

  env.reset()
  torch.cuda.synchronize()

  env_sps = measure_env_sps(env, cfg.num_steps)

  overhead_pct = 100 * (1 - env_sps / physics_sps)

  env.close()

  return BenchmarkResult(
    task=task,
    num_envs=cfg.num_envs,
    num_steps=cfg.num_steps,
    decimation=decimation,
    physics_sps=physics_sps,
    env_sps=env_sps,
    overhead_pct=overhead_pct,
  )


def get_git_commit() -> str:
  """Get current git commit SHA."""
  try:
    result = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      capture_output=True,
      text=True,
      check=True,
    )
    return result.stdout.strip()[:7]
  except subprocess.CalledProcessError:
    return "unknown"


def save_results(results: list[BenchmarkResult], output_dir: Path) -> None:
  """Save benchmark results to JSON, appending to existing data."""
  output_dir.mkdir(parents=True, exist_ok=True)
  data_file = output_dir / "throughput_data.json"

  # Load existing data.
  existing: list[dict] = []
  if data_file.exists():
    with open(data_file) as f:
      existing = json.load(f)

  # Create new run entry.
  run_entry = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "commit": get_git_commit(),
    "results": [r.to_dict() for r in results],
  }

  existing.append(run_entry)

  with open(data_file, "w") as f:
    json.dump(existing, f, indent=2)

  print(f"\nResults saved to {data_file}")


def main(cfg: ThroughputConfig) -> list[BenchmarkResult]:
  """Run throughput benchmarks on all configured tasks."""
  print("Throughput Benchmark")
  print(f"  Envs: {cfg.num_envs}")
  print(f"  Steps: {cfg.num_steps} (+ {cfg.warmup_steps} warmup)")
  print(f"  Device: {cfg.device}")

  results = []
  for task in cfg.tasks:
    result = benchmark_task(task, cfg)
    results.append(result)
    print(result)

  print("\n" + "=" * 74)
  print("Summary (all values in env steps per second):")
  print("  Physics SPS: sim.step() only (×decimation per env step)")
  print("  Env SPS: full env.step() including managers")
  print("  Overhead: time spent on non-physics work (observations, rewards, etc.)")
  print("=" * 74)
  print(f"{'Task':<35} {'Dec':>4} {'Physics SPS':>12} {'Env SPS':>12} {'Overhead':>8}")
  print("-" * 74)
  for r in results:
    task_short = r.task.replace("Mjlab-", "").replace("-Unitree-", "-")
    print(
      f"{task_short:<35} {r.decimation:>4} {r.physics_sps:>12,.0f} {r.env_sps:>12,.0f} {r.overhead_pct:>7.1f}%"
    )

  if cfg.output_dir:
    save_results(results, cfg.output_dir)

  return results


if __name__ == "__main__":
  cfg = tyro.cli(ThroughputConfig)
  main(cfg)
