"""Script to play RL agent with RSL-RL."""

from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

import gymnasium as gym
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner
from typing_extensions import assert_never

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserViewer


def run_play(
  task: str,
  wandb_run_path: str | None = None,
  checkpoint_file: str | None = None,
  motion_file: str | None = None,
  num_envs: int | None = None,
  device: str | None = None,
  video: bool = False,
  video_length: int = 200,
  video_height: int | None = None,
  video_width: int | None = None,
  camera: int | str | None = None,
  render_all_envs: bool = False,
  viewer: Literal["native", "viser"] = "native",
  export: bool = False,
  export_format: Literal["jit", "onnx", "both"] = "both",
  output_dir: str | None = None,
  verbose: bool = False,
):
  configure_torch_backends()

  if device is None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f"[INFO]: Using device: {device}")

  if checkpoint_file is not None and motion_file is None:
    raise ValueError("Must provide `motion_file` if using `checkpoint_file`.")

  env_cfg = cast(
    ManagerBasedRlEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point")
  )
  agent_cfg = cast(
    RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
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

  if checkpoint_file is not None:
    resume_path = Path(checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
  else:
    assert wandb_run_path is not None
    resume_path = get_wandb_checkpoint_path(log_root_path, Path(wandb_run_path))
  print(f"[INFO]: Loading checkpoint: {resume_path}")
  log_dir = resume_path.parent

  if isinstance(env_cfg, TrackingEnvCfg):
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

  if isinstance(env_cfg, TrackingEnvCfg):
    runner = MotionTrackingOnPolicyRunner(
      env, asdict(agent_cfg), log_dir=str(log_dir), device=device
    )
  else:
    runner = OnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)
  runner.load(str(resume_path), map_location=device)

  policy = runner.get_inference_policy(device=device)

  if export:
    print("[INFO]: Exporting policy...")

    try:
      policy_nn = runner.alg.policy  # pyright: ignore [reportAttributeAccessIssue]
    except AttributeError:
      policy_nn = runner.alg.actor_critic  # pyright: ignore [reportAttributeAccessIssue]

    if hasattr(policy_nn, "actor_obs_normalizer"):
      normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
      normalizer = policy_nn.student_obs_normalizer
    else:
      normalizer = None

    if output_dir is None:
      export_model_dir = str(log_dir / "exported")
    else:
      export_model_dir = output_dir

    print(f"[INFO]: Exporting models to: {export_model_dir}")

    log_dir_name = log_dir.name
    model_filename = resume_path.stem
    export_filename = f"{log_dir_name}_{model_filename}.onnx"

    try:
      if isinstance(env_cfg, TrackingEnvCfg):
        from mjlab.tasks.tracking.rl.exporter import (
          attach_onnx_metadata as attach_tracking_metadata,
        )
        from mjlab.tasks.tracking.rl.exporter import export_motion_policy_as_onnx

        # Export tracking policy
        print(f"[INFO]: Exporting motion tracking policy as ONNX: {export_filename}")
        export_motion_policy_as_onnx(
          env.unwrapped,
          policy_nn,
          normalizer=normalizer,
          path=export_model_dir,
          filename=export_filename,
          verbose=verbose,
        )
        attach_tracking_metadata(
          env.unwrapped, str(resume_path), export_model_dir, export_filename
        )
        print("[INFO]: Motion tracking policy export completed.")

      elif "Velocity" in task:
        from mjlab.tasks.velocity.rl.exporter import (
          attach_onnx_metadata,
          export_velocity_policy_as_onnx,
        )

        # Export velocity policy
        print(f"[INFO]: Exporting velocity policy as ONNX: {export_filename}")
        export_velocity_policy_as_onnx(
          policy_nn,
          normalizer=normalizer,
          path=export_model_dir,
          filename=export_filename,
          verbose=verbose,
        )
        attach_onnx_metadata(
          env.unwrapped, str(resume_path), export_model_dir, export_filename
        )
        print("[INFO]: Velocity policy export completed.")

      else:
        print("[WARNING]: No specialized exporter available for this task type.")

    except Exception as e:
      print(f"[ERROR]: Export failed: {e}")
      if not export:  # Only return if export was the main purpose
        return

    print(f"[INFO]: Export completed. Models saved to: {export_model_dir}")

  if viewer == "native":
    NativeMujocoViewer(env, policy, render_all_envs=render_all_envs).run()
  elif viewer == "viser":
    ViserViewer(env, policy, render_all_envs=render_all_envs).run()
  else:
    assert_never(viewer)

  env.close()


def main():
  """Entry point for the CLI."""
  tyro.cli(run_play)


if __name__ == "__main__":
  main()
