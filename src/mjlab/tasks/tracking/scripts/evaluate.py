"""Evaluate a trained tracking policy and compute metrics."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import tyro
import wandb

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.metrics import (
  compute_ee_orientation_error,
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvaluateConfig:
  """Configuration for policy evaluation."""

  wandb_run_path: str
  """W&B run path in format 'entity/project/run_id'."""
  num_envs: int = 1024
  """Number of parallel environments (= number of episodes to evaluate)."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  output_file: str | None = None
  """Optional path to save metrics as JSON."""


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, float]:
  """Run policy evaluation and compute metrics."""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load configs.
  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)

  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"Task {task_id} is not a tracking task.")

  # Load motion file from W&B run.
  api = wandb.Api()
  run = api.run(cfg.wandb_run_path)
  art = next((a for a in run.used_artifacts() if a.type == "motions"), None)
  if art is None:
    raise RuntimeError("No motion artifact found in the run.")
  motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  # Evaluation config.
  motion_cmd.sampling_mode = "start"
  env_cfg.observations["actor"].enable_corruption = True
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
  resume_path, _ = get_wandb_checkpoint_path(log_root_path, Path(cfg.wandb_run_path))
  print(f"[INFO] Loading checkpoint: {resume_path}")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  ee_body_names = env_cfg.terminations["ee_body_pos"].params["body_names"]
  print(f"[INFO] End effector bodies: {ee_body_names}")

  # Metric accumulators.
  all_mpkpe: list[torch.Tensor] = []
  all_r_mpkpe: list[torch.Tensor] = []
  all_joint_vel_error: list[torch.Tensor] = []
  all_ee_pos_error: list[torch.Tensor] = []
  all_ee_ori_error: list[torch.Tensor] = []

  done_envs = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  success = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)

  obs = env.get_observations()
  env.unwrapped.command_manager.compute(dt=env.unwrapped.step_dt)

  print(f"[INFO] Running {cfg.num_envs} evaluation episodes...")

  step = 0
  while not done_envs.all():
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)

    # Compute metrics for active envs.
    active = ~done_envs
    if active.any():
      all_mpkpe.append(torch.where(active, compute_mpkpe(command), 0.0))
      all_r_mpkpe.append(torch.where(active, compute_root_relative_mpkpe(command), 0.0))
      all_joint_vel_error.append(
        torch.where(active, compute_joint_velocity_error(command), 0.0)
      )
      all_ee_pos_error.append(
        torch.where(active, compute_ee_position_error(command, ee_body_names), 0.0)
      )
      all_ee_ori_error.append(
        torch.where(active, compute_ee_orientation_error(command, ee_body_names), 0.0)
      )

    # Track completions.
    terminated = env.unwrapped.termination_manager.terminated
    truncated = env.unwrapped.termination_manager.time_outs
    newly_done = dones.bool() & ~done_envs

    if newly_done.any():
      success = success | (newly_done & truncated & ~terminated)
      done_envs = done_envs | newly_done
      print(
        f"[INFO] {done_envs.sum().item()}/{cfg.num_envs} episodes completed "
        f"(step {step}, truncated={(newly_done & truncated).sum().item()}, "
        f"terminated={(newly_done & terminated).sum().item()})"
      )
    step += 1

  # Compute mean metrics.
  stacks = [
    all_mpkpe,
    all_r_mpkpe,
    all_joint_vel_error,
    all_ee_pos_error,
    all_ee_ori_error,
  ]
  stacks = [torch.stack(s, dim=0) for s in stacks]
  active_steps = (stacks[0] != 0).sum(dim=0).float().clamp(min=1)
  means = [s.sum(dim=0) / active_steps for s in stacks]

  metrics = {
    "success_rate": success.float().mean().item(),
    "mpkpe": means[0].mean().item(),
    "r_mpkpe": means[1].mean().item(),
    "joint_vel_error": means[2].mean().item(),
    "ee_pos_error": means[3].mean().item(),
    "ee_ori_error": means[4].mean().item(),
  }

  print("\n" + "=" * 50)
  print("Evaluation Results")
  print("=" * 50)
  for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
  print("=" * 50)

  if cfg.output_file:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
      json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {output_path}")

  env.close()
  return metrics


def main():
  import mjlab.tasks  # noqa: F401

  tracking_tasks = [t for t in list_tasks() if "Tracking" in t]
  if not tracking_tasks:
    print("No tracking tasks found.")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(tracking_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    EvaluateConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  run_evaluate(chosen_task, args)


if __name__ == "__main__":
  main()
