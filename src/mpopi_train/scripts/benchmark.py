"""Benchmark: PPO vs naive replay + PPO vs MPOPI + PPO vs MPC-guided PPO.

Runs every arm for several seeds through the real ``MpopiOnPolicyRunner`` /
mode switch and reports a per-iteration score, its area under the curve
(sample efficiency), the final score, and Welch / Mann-Whitney tests between
arms.

Two environments are supported:

- default: the pure-torch ``PointMassVecEnv`` (CPU, no MuJoCo). The score is the
  return of the deterministic policy from fixed starts, evaluated after every
  update.
- ``--task <id>``: a registered mjlab task (e.g. ``Mjlab-Cartpole-Balance``),
  with the task's own PPO hyperparameters. The score is the mean per-step
  reward of the deterministic policy over ``eval_steps`` on a separate
  ``play`` env from a fixed seed, every ``eval_every`` iterations. The
  stochastic training reward is also recorded but is confounded by the
  exploration noise level.

Examples::

  uv run --extra cpu python -m mpopi_train.scripts.benchmark \\
    --seeds 10 --iterations 150 --out-dir logs/mpopi_bench/toy
  uv run --extra cpu python -m mpopi_train.scripts.benchmark \\
    --task Mjlab-Cartpole-Balance --num-envs 64 --iterations 150 \\
    --out-dir logs/mpopi_bench/cartpole
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import tyro
from rsl_rl.env import VecEnv
from scipy import stats

import mjlab.tasks  # noqa: F401  (populates the registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
  RslRlVecEnvWrapper,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mpopi_train.algorithms import MpopiCfg
from mpopi_train.algorithms.config import MpcDataCfg
from mpopi_train.algorithms.toy_env import PointMassVecEnv
from mpopi_train.config import MpopiRunnerCfg, with_mpopi
from mpopi_train.mpc import SamplingMpcCfg
from mpopi_train.runner import MpopiOnPolicyRunner


@dataclass(frozen=True)
class Arm:
  mpopi: MpopiCfg
  minibatch_divisor: int = 1
  """Divides ``num_mini_batches``. 2 gives PPO the same samples per gradient
  step as replay ratio 1.0 (a compute-matched control), with half the steps."""
  mpc_overrides: tuple[tuple[str, object], ...] = ()
  """``mpc_ppo`` arms: fields of ``BenchmarkCfg.mpc`` changed for this arm."""


ARMS: dict[str, Arm] = {
  "A_ppo": Arm(MpopiCfg(mode="ppo")),
  "A_ppo_bigmb": Arm(MpopiCfg(mode="ppo"), minibatch_divisor=2),
  "B_naive_replay": Arm(MpopiCfg(mode="naive_replay_ppo")),
  "C_mpopi": Arm(MpopiCfg(mode="mpopi_ppo")),
  "C_mpopi_noclip": Arm(MpopiCfg(mode="mpopi_ppo", importance_weight_clip_max=None)),
  # MPC-generated data (mjlab tasks only).
  "D_mpc_ppo": Arm(MpopiCfg(mode="mpc_ppo")),
  "D_mpc_naive": Arm(MpopiCfg(mode="mpc_ppo"), mpc_overrides=(("correction", False),)),
  "D_mpc_bc_only": Arm(
    MpopiCfg(mode="mpc_ppo"), mpc_overrides=(("use_in_ppo", False),)
  ),
  "D_mpc_no_bc": Arm(MpopiCfg(mode="mpc_ppo"), mpc_overrides=(("bc_coef", 0.0),)),
}

TESTS = (
  ("C_mpopi", "A_ppo"),
  ("C_mpopi", "A_ppo_bigmb"),
  ("C_mpopi", "B_naive_replay"),
  ("B_naive_replay", "A_ppo"),
  ("A_ppo_bigmb", "A_ppo"),
  ("C_mpopi_noclip", "C_mpopi"),
  ("D_mpc_ppo", "A_ppo"),
  ("D_mpc_ppo", "D_mpc_naive"),
  ("D_mpc_ppo", "D_mpc_bc_only"),
  ("D_mpc_ppo", "D_mpc_no_bc"),
  ("D_mpc_bc_only", "A_ppo"),
  ("D_mpc_no_bc", "A_ppo"),
)

LOGGED_KEYS = (
  "kl",
  "clip_fraction",
  "mpopi/accepted",
  "mpopi/ess",
  "mpopi/weight_mean",
  "mpopi/weight_std",
  "mpopi/weight_max",
  "mpopi/raw_ratio_max",
  "mpopi/clipped_frac",
  "mpopi/behavior_kl",
  "mpopi/policy_age_mean",
  "mpc/bc_loss",
  "mpc/collect_reward",
  "mpc/collect_seconds",
)


@dataclass
class BenchmarkCfg:
  task: str | None = None
  """mjlab task id. None uses the pure-torch point-mass toy env."""
  device: str = "cpu"
  """Torch / simulation device, e.g. "cpu" or "cuda:0"."""
  seeds: int = 10
  seed_offset: int = 0
  """First seed; use fresh seeds for confirmation runs."""
  iterations: int = 60
  num_envs: int = 32
  num_steps_per_env: int | None = None
  """Rollout length. None uses 16 for the toy env and the task's own value."""
  episode_length: int = 50
  """Toy env only."""
  eval_episodes: int = 16
  """Toy env: episodes per evaluation. Task: parallel eval envs."""
  eval_every: int = 1
  """Task only: evaluate the deterministic policy every N iterations."""
  eval_steps: int = 200
  """Task only: control steps per evaluation rollout."""
  eval_seed: int = 10_000
  progress_every: int = 10
  """Print a progress line every N iterations (0 disables). Logging only;
  results are unaffected."""
  """Task only: fixed seed so every arm and run sees the same eval starts."""
  replay_buffer_size: int = 4
  replay_ratio: float = 1.0
  arms: tuple[str, ...] = ("A_ppo", "B_naive_replay", "C_mpopi", "C_mpopi_noclip")
  out_dir: Path = Path("logs/mpopi_bench")
  ppo: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      num_learning_epochs=5, num_mini_batches=4, learning_rate=1e-3
    )
  )
  """PPO hyperparameters for the toy env (tasks use their registered ones)."""
  mpc: MpcDataCfg = field(
    default_factory=lambda: MpcDataCfg(
      num_envs=8,
      num_steps=16,
      collect_iterations=20,
      buffer_segments=8,
      max_age=10,
      bc_iterations=30,
      planner=SamplingMpcCfg(num_samples=32, horizon=20),
    )
  )
  """MPC data source of the ``D_*`` arms. The planner defaults score 0.92 as a
  controller on Cartpole (``src/mpopi_train/scripts/eval_mpc.py``, 8 envs)."""


def evaluate_toy(runner: MpopiOnPolicyRunner, cfg: BenchmarkCfg) -> float:
  """Mean undiscounted return of the deterministic policy from fixed starts."""
  env = PointMassVecEnv(num_envs=cfg.eval_episodes, device=cfg.device)
  env.pos = torch.linspace(-2.0, 2.0, cfg.eval_episodes, device=cfg.device).view(-1, 1)
  policy = runner.alg.get_policy()
  total = torch.zeros(cfg.eval_episodes, device=cfg.device)
  alive = torch.ones(cfg.eval_episodes, dtype=torch.bool, device=cfg.device)
  with torch.inference_mode():
    for _ in range(cfg.episode_length):
      obs = env.get_observations()
      action = policy(obs)  # Deterministic mean; consumes no RNG.
      _, reward, dones, _ = env.step(action)
      total += reward * alive
      alive &= dones == 0
  return float(total.mean())


@contextlib.contextmanager
def _preserve_rng():
  """Restore Python, NumPy and torch (CPU and CUDA) global RNG states on exit.

  mjlab env construction calls ``seed_rng``, which reseeds them globally.
  """
  states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
  cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
  try:
    yield
  finally:
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if cuda_states is not None:
      torch.cuda.set_rng_state_all(cuda_states)


class TaskEvaluator:
  """Deterministic-policy evaluation on a separate ``play`` env of a task.

  Reports the mean per-step reward over ``eval_steps`` from a fixed seed. The
  global torch RNG state is saved and restored so evaluation cannot perturb the
  training run (including the reseed done while building the eval env).
  """

  def __init__(self, task: str, cfg: BenchmarkCfg) -> None:
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = cfg.eval_episodes
    env_cfg.seed = cfg.eval_seed
    self.eval_seed = cfg.eval_seed
    with _preserve_rng():
      self.env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device))
    self.steps = cfg.eval_steps

  def __call__(self, runner: MpopiOnPolicyRunner) -> float:
    policy = runner.alg.get_policy()
    total = 0.0
    with _preserve_rng(), torch.inference_mode():
      torch.manual_seed(self.eval_seed)  # Same reset states at every eval.
      obs, _ = self.env.reset()
      for _ in range(self.steps):
        action = policy(obs)  # Deterministic mean.
        obs, reward, _, _ = self.env.step(action)
        total += float(reward.mean())
    return total / self.steps

  def close(self) -> None:
    self.env.close()


def _build(arm: Arm, seed: int, cfg: BenchmarkCfg) -> tuple[VecEnv, MpopiRunnerCfg]:
  mpopi = replace(
    arm.mpopi,
    replay_buffer_size=cfg.replay_buffer_size,
    replay_ratio=cfg.replay_ratio,
    mpc=replace(cfg.mpc, **dict(arm.mpc_overrides)),  # type: ignore[arg-type]
  )
  if cfg.task is None:
    env = PointMassVecEnv(
      num_envs=cfg.num_envs,
      max_episode_length=cfg.episode_length,
      device=cfg.device,
      seed=seed,
    )
    agent = RslRlOnPolicyRunnerCfg(
      num_steps_per_env=cfg.num_steps_per_env or 16,
      actor=RslRlModelCfg(
        hidden_dims=(32, 32),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
      ),
      critic=RslRlModelCfg(hidden_dims=(32, 32)),
      algorithm=cfg.ppo,
    )
  else:
    env_cfg = load_env_cfg(cfg.task)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = seed
    loaded = load_rl_cfg(cfg.task)
    assert isinstance(loaded, RslRlOnPolicyRunnerCfg)
    agent = loaded
    if cfg.num_steps_per_env is not None:
      agent.num_steps_per_env = cfg.num_steps_per_env
    env = RslRlVecEnvWrapper(
      ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device),
      clip_actions=agent.clip_actions,
    )
  agent.seed = seed
  agent.logger = "tensorboard"
  num_mini_batches = max(1, agent.algorithm.num_mini_batches // arm.minibatch_divisor)
  agent.algorithm = replace(agent.algorithm, num_mini_batches=num_mini_batches)
  return env, with_mpopi(agent, mpopi)


def run_one(arm_name: str, seed: int, cfg: BenchmarkCfg) -> list[dict]:
  torch.manual_seed(seed)
  env, agent = _build(ARMS[arm_name], seed, cfg)
  runner = MpopiOnPolicyRunner(env, asdict(agent), log_dir=None, device=cfg.device)
  env_steps_per_it = env.num_envs * agent.num_steps_per_env

  # Accumulate the per-step training reward of each rollout.
  reward_sum = torch.zeros(())
  reward_count = 0
  env_step = env.step

  def recording_step(actions: torch.Tensor):
    nonlocal reward_sum, reward_count
    out = env_step(actions)
    reward_sum = reward_sum + out[1].detach().float().sum().cpu()
    reward_count += out[1].numel()
    return out

  env.step = recording_step  # type: ignore[method-assign]

  evaluator = TaskEvaluator(cfg.task, cfg) if cfg.task is not None else None
  rows: list[dict] = []

  def evaluate(it: int) -> float:
    if evaluator is None:
      return evaluate_toy(runner, cfg)
    last = it == cfg.iterations - 1
    return evaluator(runner) if it % cfg.eval_every == 0 or last else math.nan

  def on_log(**kw) -> None:
    nonlocal reward_sum, reward_count
    it = kw["it"]
    loss = kw["loss_dict"]
    train_reward = float(reward_sum) / max(1, reward_count)
    reward_sum, reward_count = torch.zeros(()), 0
    rewbuffer = runner.logger.rewbuffer
    row = {
      "arm": arm_name,
      "seed": seed,
      "iteration": it,
      "env_steps": (it + 1) * env_steps_per_it,
      "train_reward": train_reward,
      "episode_reward": sum(rewbuffer) / len(rewbuffer) if rewbuffer else math.nan,
      "eval_return": evaluate(it),
      "gradient_samples": loss.get("mpopi/gradient_samples", env_steps_per_it),
    }
    row.update({k: loss.get(k, math.nan) for k in LOGGED_KEYS})
    rows.append(row)
    if cfg.progress_every > 0 and (
      it % cfg.progress_every == 0 or it == cfg.iterations - 1
    ):
      _print_progress(arm_name, seed, it, cfg.iterations, rows)

  runner.logger.log = on_log  # type: ignore[method-assign]
  start = time.time()
  runner.learn(num_learning_iterations=cfg.iterations)
  seconds = time.time() - start
  for row in rows:
    row["run_seconds"] = seconds
  collector = getattr(runner.alg, "mpc_collector", None)
  if collector is not None:
    collector.close()
  if isinstance(env, RslRlVecEnvWrapper):
    env.close()
  if evaluator is not None:
    evaluator.close()
  return rows


def _print_progress(arm: str, seed: int, it: int, total: int, rows: list[dict]) -> None:
  """One-line training progress: latest eval score plus PPO/MPOPI diagnostics."""
  row = rows[-1]
  evals = [r["eval_return"] for r in rows if not math.isnan(r["eval_return"])]
  parts = [
    f"[{arm} seed {seed}] it {it + 1:>4}/{total}",
    f"eval {evals[-1]:9.4f}" if evals else "eval       n/a",
    f"train_r {row['train_reward']:8.4f}",
  ]
  for key, label in (
    ("kl", "kl"),
    ("clip_fraction", "clip"),
    ("mpc/bc_loss", "bc"),
    ("mpc/collect_reward", "mpc_r"),
    ("mpopi/ess", "ess"),
    ("mpopi/weight_mean", "w"),
    ("mpopi/behavior_kl", "kl_mu"),
  ):
    if not math.isnan(row[key]):
      parts.append(f"{label} {row[key]:.3f}")
  print("  ".join(parts), flush=True)


def _ci95(x: list[float]) -> tuple[float, float]:
  n = len(x)
  mean = sum(x) / n
  if n < 2:
    return mean, math.nan
  sd = math.sqrt(sum((v - mean) ** 2 for v in x) / (n - 1))
  return mean, float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)


def summarize(rows: list[dict], cfg: BenchmarkCfg) -> dict:
  score_key = "eval_return"
  per_run: dict[str, dict[int, list[dict]]] = {}
  for r in rows:
    per_run.setdefault(r["arm"], {}).setdefault(r["seed"], []).append(r)

  def run_stats(runs: list[dict]) -> dict:
    score = [r[score_key] for r in runs if not math.isnan(r[score_key])]
    tail = max(1, len(score) // 10)
    return {
      "auc": sum(score) / len(score),  # Mean score over training.
      "final": sum(score[-tail:]) / tail,  # Mean of the last 10% of iterations.
    }

  summary: dict = {
    "score": score_key,
    "config": {k: str(v) for k, v in asdict(cfg).items()},
  }
  values: dict[str, dict[str, list[float]]] = {}
  for arm, seeds in per_run.items():
    s = [run_stats(v) for v in seeds.values()]
    values[arm] = {m: [x[m] for x in s] for m in ("auc", "final")}
    diag = {}
    for key in LOGGED_KEYS:
      vals = [r[key] for v in seeds.values() for r in v if not math.isnan(r[key])]
      diag[key] = sum(vals) / len(vals) if vals else math.nan
    mpc_seconds = [
      sum(
        r["mpc/collect_seconds"] for r in v if not math.isnan(r["mpc/collect_seconds"])
      )
      for v in seeds.values()
    ]
    summary[arm] = {
      m: dict(zip(("mean", "ci95"), _ci95(v), strict=True))
      for m, v in values[arm].items()
    } | {
      "per_seed": values[arm],
      "diagnostics_mean": diag,
      "run_seconds_mean": sum(v[0]["run_seconds"] for v in seeds.values()) / len(seeds),
      "mpc_collect_seconds_mean": sum(mpc_seconds) / len(mpc_seconds),
    }

  tests = {}
  for a, b in TESTS:
    if a not in values or b not in values:
      continue
    for m in ("auc", "final"):
      xa, xb = values[a][m], values[b][m]
      welch = stats.ttest_ind(xa, xb, equal_var=False)
      mwu = stats.mannwhitneyu(xa, xb, alternative="two-sided")
      tests[f"{a} vs {b} [{m}]"] = {
        "diff": sum(xa) / len(xa) - sum(xb) / len(xb),
        "welch_p": float(welch.pvalue),  # type: ignore[attr-defined]
        "mwu_p": float(mwu.pvalue),
      }
  summary["tests"] = tests
  return summary


def main(cfg: BenchmarkCfg) -> None:
  cfg.out_dir.mkdir(parents=True, exist_ok=True)
  rows: list[dict] = []
  for arm in cfg.arms:
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.seeds):
      start = time.time()
      run_rows = run_one(arm, seed, cfg)
      rows += run_rows
      last = run_rows[-1]
      score = last["eval_return"]
      print(
        f"{arm:>16} seed {seed}: final score {score:9.4f}"
        f"  ({time.time() - start:.1f}s)",
        flush=True,
      )

  with open(cfg.out_dir / "curves.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  summary = summarize(rows, cfg)
  (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

  print(f"\nScore = {summary['score']} (higher is better), mean ± 95% CI")
  print(f"{'arm':>16} {'AUC':>20} {'final':>20} {'run s':>8} {'MPC s':>8}")
  for arm in cfg.arms:
    s = summary[arm]
    print(
      f"{arm:>16} {s['auc']['mean']:10.4f} ± {s['auc']['ci95']:7.4f}"
      f" {s['final']['mean']:10.4f} ± {s['final']['ci95']:7.4f}"
      f" {s['run_seconds_mean']:8.1f} {s['mpc_collect_seconds_mean']:8.1f}"
    )
  print("\nTests (two-sided)")
  for name, t in summary["tests"].items():
    print(
      f"  {name:<40} diff {t['diff']:+9.4f}  Welch p {t['welch_p']:.3g}"
      f"  MWU p {t['mwu_p']:.3g}"
    )
  print(f"\nWrote {cfg.out_dir / 'curves.csv'} and {cfg.out_dir / 'summary.json'}")


if __name__ == "__main__":
  main(tyro.cli(BenchmarkCfg))
