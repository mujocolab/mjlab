"""Evaluate controllers on a velocity-tracking task at fixed forward speeds.

For each commanded forward speed (straight ahead, no turning), a ``play`` env
of the task is driven from a fixed seed by the sampling MPC, by a trained
policy checkpoint, or by zero actions as a baseline. Reported per speed: the
achieved forward speed in the base frame, the tracking error, falls per env,
and the mean per-step reward.

Example (GPU)::

  uv run python scripts/mpc/eval_mpc_velocity.py --device cuda:0 \\
    --speeds 0.5 1.0 1.5 --num-envs 8 --steps 250 --mpc.num-samples 32

  uv run python scripts/mpc/eval_mpc_velocity.py \\
    --task Mjlab-Velocity-Flat-Unitree-G1-2k --controllers policy \\
    --checkpoint logs/rsl_rl/g1_velocity_2k/<run>/model_1999.pt
"""

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import imageio_ffmpeg
import numpy as np
import torch
import tyro
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401  (populates the registry)
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.mpc import SamplingMpc, SamplingMpcCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


@dataclass
class EvalVelocityCfg:
  task: str = "Mjlab-Velocity-Flat-Unitree-G1"
  speeds: tuple[float, ...] = (0.5, 1.0, 1.5)
  """Commanded forward speeds in m/s (base frame, no lateral or yaw command)."""
  controllers: tuple[Literal["zero", "mpc", "policy"], ...] = ("zero", "mpc")
  num_envs: int = 8
  steps: int = 250
  """Control steps per speed (250 = 5 s at 50 Hz)."""
  settle_steps: int = 50
  """Steps ignored when averaging the achieved speed (acceleration phase)."""
  seed: int = 10_000
  device: str = "cuda:0"
  mpc: SamplingMpcCfg = field(
    default_factory=lambda: SamplingMpcCfg(num_samples=32, horizon=20, noise_std=0.3)
  )
  checkpoint: Path | None = None
  """Policy checkpoint (``model_*.pt``) for the ``policy`` controller."""
  out: Path | None = None
  """Optional JSON file for the results."""
  video_dir: Path | None = None
  """If set, save an MP4 of each controller and speed (first env) here."""


def fixed_speed_cfg(
  task: str, speed: float, num_envs: int, seed: int
) -> ManagerBasedRlEnvCfg:
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (speed, speed)  # type: ignore[attr-defined]
  twist.ranges.lin_vel_y = (0.0, 0.0)  # type: ignore[attr-defined]
  twist.ranges.ang_vel_z = (0.0, 0.0)  # type: ignore[attr-defined]
  twist.ranges.heading = None  # type: ignore[attr-defined]
  for name, value in (
    ("heading_command", False),
    ("rel_standing_envs", 0.0),
    ("rel_heading_envs", 0.0),
    ("rel_world_envs", 0.0),
    ("rel_forward_envs", 0.0),
    ("init_velocity_prob", 0.0),
  ):
    setattr(twist, name, value)
  twist.resampling_time_range = (1e9, 1e9)
  return cfg


def run(cfg: EvalVelocityCfg, controller: str, speed: float) -> dict:
  env_cfg = fixed_speed_cfg(cfg.task, speed, cfg.num_envs, cfg.seed)
  render_mode = "rgb_array" if cfg.video_dir is not None else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode=render_mode)
  env.reset()
  frames: list[np.ndarray] = []
  robot = env.scene["robot"]
  n, dim = cfg.num_envs, env.action_manager.total_action_dim
  planner = (
    SamplingMpc(copy.deepcopy(env_cfg), n, cfg.mpc, device=cfg.device)
    if controller == "mpc"
    else None
  )
  policy, clip, obs = None, None, None
  if controller == "policy":
    policy, clip = _load_policy(cfg, env)
    obs = env.observation_manager.compute()
  speeds, rewards = [], []
  falls = torch.zeros(n, device=cfg.device)
  plan_time = 0.0
  for step in range(cfg.steps):
    if planner is not None:
      t0 = time.time()
      action = planner.plan(env).action
      plan_time += time.time() - t0
    elif policy is not None:
      with torch.inference_mode():
        action = policy(TensorDict(obs, batch_size=[n]))
      if clip is not None:
        action = action.clamp(-clip, clip)
    else:
      action = torch.zeros(n, dim, device=cfg.device)
    obs, reward, terminated, truncated, _ = env.step(action)
    fell = terminated & ~truncated
    falls += fell.float()
    if planner is not None and bool(fell.any()):
      planner.reset(fell.nonzero().flatten())
    rewards.append(float(reward.mean()))
    if render_mode is not None:
      frame = env.render()
      if frame is not None:
        frames.append(np.ascontiguousarray(frame))
    if step >= cfg.settle_steps:
      speeds.append(robot.data.root_link_lin_vel_b[:, 0].clone())
    if planner is not None and (step + 1) % 25 == 0:
      v = torch.stack(speeds).mean() if speeds else torch.tensor(float("nan"))
      print(
        f"  [mpc {speed} m/s] step {step + 1}/{cfg.steps}  speed {float(v):.2f}"
        f"  falls {float(falls.sum()):.0f}  plan {plan_time / (step + 1):.2f} s/step",
        flush=True,
      )
  env.close()
  if planner is not None:
    planner.close()
  if frames and cfg.video_dir is not None:
    _write_video(cfg.video_dir / f"{controller}_{speed:.1f}mps.mp4", frames)
  v = torch.stack(speeds)  # [T, N]
  result = {
    "controller": controller,
    "command": speed,
    "speed": float(v.mean()),
    "abs_error": float((v - speed).abs().mean()),
    "falls_per_env": float(falls.mean()),
    "reward_per_step": sum(rewards) / len(rewards),
  }
  if planner is not None:
    result["plan_seconds_per_step"] = plan_time / cfg.steps
  return result


def _load_policy(cfg: EvalVelocityCfg, env: ManagerBasedRlEnv):
  """Deterministic policy of ``cfg.checkpoint`` and its action clipping."""
  if cfg.checkpoint is None:
    raise ValueError("The policy controller needs --checkpoint.")
  agent_cfg = load_rl_cfg(cfg.task)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  # The wrapper resets the env; the caller starts from the reset state.
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
  runner.load(
    str(cfg.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=cfg.device,
  )
  return runner.get_inference_policy(device=cfg.device), agent_cfg.clip_actions


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 50) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  h, w = frames[0].shape[:2]
  writer = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, macro_block_size=1)
  writer.send(None)
  for frame in frames:
    writer.send(frame[..., :3].astype(np.uint8))
  writer.close()
  print(f"Wrote {path}")


def main(cfg: EvalVelocityCfg) -> None:
  results = [run(cfg, c, s) for s in cfg.speeds for c in cfg.controllers]
  seconds = cfg.steps * 0.02
  print(
    f"\n{cfg.task}: {cfg.num_envs} envs x {cfg.steps} steps ({seconds:.0f} s), "
    f"seed {cfg.seed}, speed averaged after step {cfg.settle_steps}"
  )
  print(
    f"{'command':>8} {'controller':>10} {'speed':>7} {'|err|':>7} {'falls/env':>10} "
    f"{'reward/step':>12} {'plan s/step':>12}"
  )
  for r in results:
    plan = f"{r['plan_seconds_per_step']:.2f}" if "plan_seconds_per_step" in r else "-"
    print(
      f"{r['command']:>8.2f} {r['controller']:>10} {r['speed']:>7.2f} "
      f"{r['abs_error']:>7.2f} {r['falls_per_env']:>10.2f} {r['reward_per_step']:>12.4f} "
      f"{plan:>12}"
    )
  if cfg.out is not None:
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(
      json.dumps({"config": asdict(cfg), "results": results}, indent=2, default=str)
    )
    print(f"Wrote {cfg.out}")


if __name__ == "__main__":
  main(tyro.cli(EvalVelocityCfg))
