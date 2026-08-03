"""Measure physical ball-visibility transitions in a football rollout."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path)
  return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, float | int | str | bool]:
  env_cfg = load_env_cfg(args.task)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(env_cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task)
  if runner_cls is None:
    raise ValueError(f"Task {args.task!r} has no runner")
  runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  wrapped.seed(args.seed)
  observations, _ = wrapped.reset()
  visible = observations["actor"][:, -1] > 0.5
  seen_visible = visible.clone()
  left_after_visible = torch.zeros_like(visible)
  recovered = torch.zeros_like(visible)
  completed_episodes = 0
  recovered_episodes = 0
  invisible_frames = 0
  total_frames = 0

  try:
    with torch.inference_mode():
      for _ in range(args.steps):
        actions = policy(observations)
        observations, _, dones, _ = wrapped.step(actions)
        next_visible = observations["actor"][:, -1] > 0.5
        done = dones.bool()

        active = ~done
        newly_out = active & seen_visible & ~next_visible
        left_after_visible |= newly_out
        recovered |= active & left_after_visible & next_visible
        seen_visible |= active & next_visible

        completed_episodes += int(done.sum().item())
        recovered_episodes += int((done & recovered).sum().item())
        invisible_frames += int((~next_visible).sum().item())
        total_frames += args.num_envs

        seen_visible = torch.where(done, next_visible, seen_visible)
        left_after_visible = torch.where(
          done, torch.zeros_like(left_after_visible), left_after_visible
        )
        recovered = torch.where(done, torch.zeros_like(recovered), recovered)

    sampled_episodes = completed_episodes + args.num_envs
    recovered_episodes += int(recovered.sum().item())
    invisible_fraction = invisible_frames / max(total_frames, 1)
    recovery_episode_fraction = recovered_episodes / max(sampled_episodes, 1)
    result: dict[str, float | int | str | bool] = {
      "task": args.task,
      "checkpoint": str(args.checkpoint.resolve()),
      "seed": args.seed,
      "num_envs": args.num_envs,
      "steps": args.steps,
      "sampled_episodes": sampled_episodes,
      "completed_episodes": completed_episodes,
      "recovered_episodes": recovered_episodes,
      "invisible_frame_fraction": invisible_fraction,
      "recovery_episode_fraction": recovery_episode_fraction,
      "passes_invisible_frame_gate": invisible_fraction >= 0.05,
      "passes_recovery_episode_gate": recovery_episode_fraction >= 0.20,
    }
    return result
  finally:
    wrapped.close()


def main() -> None:
  args = parse_args()
  if args.num_envs <= 0 or args.steps <= 0:
    raise ValueError("num-envs and steps must be positive")
  torch.manual_seed(args.seed)
  result = run(args)
  serialized = json.dumps(result, indent=2, sort_keys=True)
  print(serialized)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")


if __name__ == "__main__":
  main()
