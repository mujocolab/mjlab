"""Script to train RL agent with RSL-RL."""

import os
import pathlib
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import tyro
import wandb

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class TrainConfig:
  registry_name: str
  env: TrackingEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  device: str = "cuda:0"
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000


def main(task: str, cfg: TrainConfig) -> None:
  configure_torch_backends()

  # Check if the registry name includes alias, if not, append ":latest".
  registry_name = cfg.registry_name
  if ":" not in registry_name:
    registry_name += ":latest"

  api = wandb.Api()
  artifact = api.artifact(registry_name)
  cfg.env.commands.motion.motion_file = str(
    pathlib.Path(artifact.download()) / "motion.npz"
  )

  # Specify directory for logging experiments.
  log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
  log_root_path.resolve()
  print(f"[INFO] Logging experiment in directory: {log_root_path}")
  log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    log_dir += f"_{cfg.agent.run_name}"
  log_dir = log_root_path / log_dir

  # Create env.
  env = gym.make(
    task, cfg=cfg.env, device=cfg.device, render_mode="rgb_array" if cfg.video else None
  )

  # Save resume path before creating a new log_dir.
  resume_path = None
  if cfg.agent.resume:
    resume_path = get_checkpoint_path(
      log_root_path,
      cfg.agent.load_run,
      cfg.agent.load_checkpoint,
    )

  # Wrap for video recording.
  if cfg.video:
    video_kwargs = {
      "video_folder": os.path.join(log_dir, "videos", "train"),
      "step_trigger": lambda step: step % cfg.video_interval == 0,
      "video_length": cfg.video_length,
      "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  runner = MotionTrackingOnPolicyRunner(
    env,
    asdict(cfg.agent),
    log_dir=str(log_dir),
    device=cfg.device,
    registry_name=registry_name,
  )
  runner.add_git_repo_to_log(__file__)

  if cfg.agent.resume:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


if __name__ == "__main__":
  # Parse first argument to choose the task.
  task_prefix = "Mjlab-Tracking-"
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(
      [k for k in gym.registry.keys() if k.startswith(task_prefix)]
    ),
    add_help=False,
    return_unknown_args=True,
  )
  del task_prefix

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(chosen_task, "rl_cfg_entry_point")
  assert isinstance(env_cfg, TrackingEnvCfg)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig(
      registry_name=tyro.MISSING,
      env=env_cfg,
      agent=agent_cfg,
    ),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del env_cfg, agent_cfg, remaining_args

  main(chosen_task, args)
