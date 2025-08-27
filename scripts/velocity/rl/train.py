"""Script to train RL agent with RSL-RL."""

import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.os import dump_yaml, get_checkpoint_path

# TODO(kevin): Make sure this does not interfere with seed_rng call in env.seed().
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

_HERE = Path(__file__).parent


def main(
  task: str,
  num_envs: int | None = None,
  seed: int | None = None,
  max_iterations: int | None = None,
  device: str | None = None,
  video: bool = False,
  video_length: int = 200,
  video_interval: int = 2000,
):
  env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(task, "rl_cfg_entry_point")

  env_cfg.sim.num_envs = num_envs or env_cfg.sim.num_envs
  agent_cfg.max_iterations = max_iterations or agent_cfg.max_iterations
  agent_cfg.seed = seed or agent_cfg.seed

  # Set the environment seed.
  env_cfg.seed = agent_cfg.seed
  env_cfg.sim.device = device or env_cfg.sim.device

  # Specify directory for logging experiments.
  log_root_path = _HERE.parents[2] / "logs" / "rsl_rl" / agent_cfg.experiment_name
  log_root_path.resolve()
  print(f"[INFO] Logging experiment in directory: {log_root_path}")
  log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if agent_cfg.run_name:
    log_dir += f"_{agent_cfg.run_name}"
  log_dir = log_root_path / log_dir

  # Create env.
  env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if video else None)

  # Save resume path before creating a new log_dir.
  if agent_cfg.resume:
    resume_path = get_checkpoint_path(
      log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )

  # Wrap for video recording.
  if video:
    video_kwargs = {
      "video_folder": os.path.join(log_dir, "videos", "train"),
      "step_trigger": lambda step: step % video_interval == 0,
      "video_length": video_length,
      "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = OnPolicyRunner(
    env,
    asdict(agent_cfg),
    log_dir=log_dir,
    device=agent_cfg.device,
  )
  runner.add_git_repo_to_log(__file__)

  if agent_cfg.resume:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
  )

  env.close()


if __name__ == "__main__":
  tyro.cli(main)
