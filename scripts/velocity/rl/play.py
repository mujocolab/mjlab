from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

import gymnasium as gym
import tyro
from rsl_rl.runners import OnPolicyRunner
from typing_extensions import assert_never

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.rl import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserViewer


def main(
  task: str,
  wandb_run_path: Path,
  num_envs: int | None = None,
  device: str = "cuda:0",
  video: bool = False,
  video_length: int = 200,
  video_height: int | None = None,
  video_width: int | None = None,
  camera: int | str | None = -1,
  render_all_envs: bool = False,
  viewer: Literal["native", "viser"] = "native",
):
  configure_torch_backends()

  env_cfg = cast(
    LocomotionVelocityEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point")
  )
  agent_cfg = cast(
    RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
  )

  env_cfg.scene.num_envs = num_envs or env_cfg.scene.num_envs
  env_cfg.sim.render.camera = camera or -1
  env_cfg.sim.render.height = video_height or env_cfg.sim.render.height
  env_cfg.sim.render.width = video_width or env_cfg.sim.render.width

  log_root_path = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
  log_root_path = log_root_path.resolve()
  print(f"[INFO]: Loading experiment from: {log_root_path}")

  resume_path = get_wandb_checkpoint_path(log_root_path, wandb_run_path)
  print(f"[INFO]: Loading checkpoint: {resume_path}")

  log_dir = resume_path.parent

  env = gym.make(
    task, cfg=env_cfg, device=device, render_mode="rgb_array" if video else None
  )
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

  runner = OnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)
  runner.load(str(resume_path), map_location=device)

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
