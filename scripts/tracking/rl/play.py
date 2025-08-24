from dataclasses import asdict
from pathlib import Path
from typing import cast

import gymnasium as gym
import torch
import tyro

import wandb
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.viewer import NativeMujocoViewer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

KEY_ENTER = 257
FRAME_TIME = 1.0 / 60.0

_HERE = Path(__file__).parent


def main(
  task: str,
  wandb_run_path: str,
  motion_file: str | None = None,
  num_envs: int | None = None,
  device: str | None = None,
  video: bool = False,
  video_length: int = 200,
  video_height: int | None = None,
  video_width: int | None = None,
  camera: int | str | None = -1,
  render_all_envs: bool = False,
):
  env_cfg = cast(TrackingEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point"))
  agent_cfg = cast(
    RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
  )

  env_cfg.sim.num_envs = num_envs or env_cfg.sim.num_envs
  env_cfg.sim.device = device or env_cfg.sim.device
  env_cfg.sim.render.camera = camera or -1
  env_cfg.sim.render.height = video_height or env_cfg.sim.render.height
  env_cfg.sim.render.width = video_width or env_cfg.sim.render.width

  log_root_path = _HERE / "logs" / "rsl_rl" / agent_cfg.experiment_name
  log_root_path = log_root_path.resolve()
  print(f"[INFO]: Loading experiment from: {log_root_path}")

  resume_path = get_wandb_checkpoint_path(log_root_path, Path(wandb_run_path))
  print(f"[INFO]: Loading checkpoint: {resume_path}")

  if motion_file is not None:
    print(f"[INFO]: Using motion file from CLI: {motion_file}")
    env_cfg.commands.motion.motion_file = motion_file
  else:
    api = wandb.Api()
    wandb_run = api.run(str(wandb_run_path))
    art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
    if art is None:
      raise RuntimeError("No motion artifact found in the run.")
    env_cfg.commands.motion.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir = resume_path.parent

  env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if video else None)
  if video:
    video_kwargs = {
      "video_folder": log_dir / "videos" / "play",
      "step_trigger": lambda step: step == 0,
      "video_length": video_length,
      "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = MotionTrackingOnPolicyRunner(
    env, asdict(agent_cfg), log_dir=str(log_dir), device=agent_cfg.device
  )
  runner.load(str(resume_path), map_location=agent_cfg.device)

  policy = runner.get_inference_policy(device=env.device)

  viewer = NativeMujocoViewer(env, policy, 60.0, render_all_envs)
  viewer.run()

  env.close()


if __name__ == "__main__":
  tyro.cli(main)
