"""Evaluate sampling MPC (MPPI / MPOPI) as a controller on an mjlab task.

Runs the task's ``play`` env from a fixed seed and reports the mean per-step
reward of the MPC controller next to zero-action and random-action baselines,
all from the same initial states. No learning is involved.

Example::

  uv run --extra cpu python -m mpopi_train.scripts.eval_mpc --num-envs 16 --steps 200 \\
    --mpc.num-samples 64 --mpc.horizon 20
"""

import time
from dataclasses import dataclass, field
from typing import Literal

import torch
import tyro

import mjlab.tasks  # noqa: F401  (populates the registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mpopi_train.mpc import SamplingMpc, SamplingMpcCfg


@dataclass
class EvalMpcCfg:
  task: str = "Mjlab-Cartpole-Balance"
  controllers: tuple[Literal["mpc", "zero", "random"], ...] = ("zero", "random", "mpc")
  num_envs: int = 16
  steps: int = 200
  seed: int = 10_000
  """Seed of the evaluated env; the same for every controller."""
  device: str = "cpu"
  max_step_reward: float | None = 0.05
  """Per-step reward upper bound used to print a normalized score
  (0.05 for Cartpole: weight 1.0 x dt 0.05). None prints raw rewards only."""
  mpc: SamplingMpcCfg = field(default_factory=SamplingMpcCfg)


def run(cfg: EvalMpcCfg, controller: str) -> dict:
  env_cfg = load_env_cfg(cfg.task, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env.reset()
  action_dim = env.action_manager.total_action_dim
  mpc = (
    SamplingMpc(load_env_cfg(cfg.task), cfg.num_envs, cfg.mpc, device=cfg.device)
    if controller == "mpc"
    else None
  )
  gen = torch.Generator(device=cfg.device).manual_seed(0)
  total, ess_sum, plan_time = 0.0, 0.0, 0.0
  for step in range(cfg.steps):
    if mpc is not None:
      start = time.time()
      plan = mpc.plan(env)
      plan_time += time.time() - start
      action = plan.action
      ess_sum += float(plan.ess.mean())
    elif controller == "zero":
      action = torch.zeros(cfg.num_envs, action_dim, device=cfg.device)
    else:
      action = torch.rand(cfg.num_envs, action_dim, device=cfg.device, generator=gen)
      action = 2 * action - 1
    env.step(action)
    total += float(env.reward_buf.mean())
    if mpc is not None and (step + 1) % 20 == 0:
      score = total / (step + 1)
      norm = f" ({score / cfg.max_step_reward:.3f})" if cfg.max_step_reward else ""
      print(
        f"  [mpc] step {step + 1:>4}/{cfg.steps}  mean reward {score:.4f}{norm}"
        f"  plan {plan_time / (step + 1):.2f}s/step  ess {ess_sum / (step + 1):.2f}",
        flush=True,
      )
  env.close()
  if mpc is not None:
    mpc.close()
  result = {"controller": controller, "mean_reward": total / cfg.steps}
  if mpc is not None:
    result["plan_seconds_per_step"] = plan_time / cfg.steps
  return result


def main(cfg: EvalMpcCfg) -> None:
  results = [run(cfg, c) for c in cfg.controllers]
  print(f"\n{cfg.task}: {cfg.num_envs} envs x {cfg.steps} steps, seed {cfg.seed}")
  for r in results:
    line = f"  {r['controller']:>6}: mean per-step reward {r['mean_reward']:.4f}"
    if cfg.max_step_reward:
      line += f"  normalized {r['mean_reward'] / cfg.max_step_reward:.3f}"
    if "plan_seconds_per_step" in r:
      line += f"  ({r['plan_seconds_per_step']:.2f} s planning per control step)"
    print(line)


if __name__ == "__main__":
  main(tyro.cli(EvalMpcCfg))
