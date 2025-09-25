"""Script to play RL agent with RSL-RL."""

from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional, cast

import gymnasium as gym
import tyro
from rsl_rl.runners import OnPolicyRunner
from typing_extensions import assert_never

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.tasks.velocity.rl import attach_onnx_metadata, export_velocity_policy_as_onnx
from mjlab.tasks.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserViewer


def main(
  task: str,
  wandb_run_path: Path,
  motion_file: Optional[str] = None,
  num_envs: int | None = None,
  device: str = "cuda:0",
  video: bool = False,
  video_length: int = 200,
  video_height: int | None = None,
  video_width: int | None = None,
  camera: int | str | None = None,
  render_all_envs: bool = False,
  viewer: Literal["native", "viser"] = "native",
):
  configure_torch_backends()

  # Load configs
  env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
  agent_cfg = cast(
    RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
  )

  # Detect task type by config class
  is_tracking = isinstance(env_cfg, TrackingEnvCfg)
  is_velocity = isinstance(env_cfg, LocomotionVelocityEnvCfg)

  if not (is_tracking or is_velocity):
    raise RuntimeError(
      f"Unsupported env cfg type: {type(env_cfg).__name__}. "
      "Expected TrackingEnvCfg or LocomotionVelocityEnvCfg."
    )

  if num_envs is not None:
    env_cfg.scene.num_envs = num_envs
  if camera is not None:
    env_cfg.sim.render.camera = camera
  if video_height is not None:
    env_cfg.sim.render.height = video_height
  if video_width is not None:
    env_cfg.sim.render.width = video_width

  log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
  print(f"[INFO]: Loading experiment from: {log_root_path}")
  resume_path = get_wandb_checkpoint_path(log_root_path, wandb_run_path)
  print(f"[INFO]: Loading checkpoint: {resume_path}")
  log_dir = resume_path.parent

  # Check if tracking and resolves motion file
  if is_tracking:
    env_cfg = cast(TrackingEnvCfg, env_cfg)
    if motion_file is not None:
      print(f"[INFO]: Using motion file from CLI: {motion_file}")
      env_cfg.commands.motion.motion_file = motion_file
    else:
      import wandb

      api = wandb.Api()
      wandb_run = api.run(str(wandb_run_path))
      art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
      if art is None:
        raise RuntimeError("No motion artifact found in the run.")
      env_cfg.commands.motion.motion_file = str(Path(art.download()) / "motion.npz")

  env = gym.make(
    task, cfg=env_cfg, device=device, render_mode="rgb_array" if video else None
  )
  if video:
    print("[INFO] Recording videos during play")
    env = gym.wrappers.RecordVideo(
      env,
      video_folder=str(log_dir / "videos" / "play"),
      step_trigger=lambda step: step == 0,
      video_length=video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if is_velocity:
    runner = OnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)

  else:
    runner = MotionTrackingOnPolicyRunner(
      env, asdict(agent_cfg), log_dir=str(log_dir), device=device
    )

  runner.load(str(resume_path), map_location=device)

  if is_velocity:
    export_model_dir = log_dir / "exported"
    export_velocity_policy_as_onnx(
      runner.alg.policy,
      normalizer=runner.alg.policy.actor_obs_normalizer,
      path=str(export_model_dir),
      filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, str(wandb_run_path), str(export_model_dir))

  policy = runner.get_inference_policy(device=device)

  if viewer == "native":
    NativeMujocoViewer(env, policy, render_all_envs=render_all_envs).run()
  elif viewer == "viser":
    ViserViewer(env, policy, render_all_envs=render_all_envs).run()
  else:
    assert_never(viewer)

  env.close()


if __name__ == "__main__":
  tyro.cli(main)
