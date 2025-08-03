"""Script to train RL agent with RSL-RL."""

from dataclasses import asdict
from pathlib import Path
import torch
from tqdm import tqdm
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from mjlab.tasks.utils.parse_cfg import load_cfg_from_registry
import gymnasium as gym

import utils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

_HERE = Path(__file__).parent
# _TASK = "Mjlab-Velocity-Flat-Unitree-Go1-v0"
_TASK = "Tracking-Flat-G1-Play-v0"


def main(
  task: str = _TASK,
  num_envs: int | None = None,
  device: str | None = None,
  video: bool = False,
  video_length: int = 200,
  video_height: int | None = None,
  video_width: int | None = None,
  camera: int | str | None = -1,
  checkpoint: Path | None = None,
  wandb_run_path: Path | None = None,
):
  env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(task, "rl_cfg_entry_point")

  env_cfg.sim.num_envs = num_envs or env_cfg.sim.num_envs
  env_cfg.sim.device = device or env_cfg.sim.device
  env_cfg.sim.render.camera = camera or -1
  env_cfg.sim.render.height = video_height or env_cfg.sim.render.height
  env_cfg.sim.render.width = video_width or env_cfg.sim.render.width

  log_root_path = _HERE / "logs" / "rsl_rl" / agent_cfg.experiment_name
  log_root_path = log_root_path.resolve()
  print(f"[INFO]: Loading experiment from: {log_root_path}")
  if checkpoint is not None:
    resume_path = checkpoint.resolve()
  elif wandb_run_path is not None:
    resume_path = utils.get_wandb_checkpoint_path(log_root_path, wandb_run_path)
  else:
    resume_path = utils.get_checkpoint_path(
      log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
  print(f"[INFO]: Loading checkpoint: {resume_path}")

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

  runner = OnPolicyRunner(
    env, asdict(agent_cfg), log_dir=log_dir, device=agent_cfg.device
  )
  runner.load(resume_path, map_location=agent_cfg.device)

  policy = runner.get_inference_policy(device=env.device)

  obs = env.get_observations()
  for _ in tqdm(range(video_length)):
    actions = policy(obs)
    obs, _, _, _ = env.step(actions)

  env.close()


if __name__ == "__main__":
  tyro.cli(main)
