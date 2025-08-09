from dataclasses import asdict
from pathlib import Path
import torch
import mujoco
import time
import mujoco.viewer
import tyro
import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
import gymnasium as gym

from mjlab.utils.os import get_wandb_checkpoint_path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

FRAME_TIME = 1.0 / 60.0
KEY_BACKSPACE = 259
KEY_ENTER = 257

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

  resume_path = get_wandb_checkpoint_path(log_root_path, wandb_run_path)
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
    env, asdict(agent_cfg), log_dir=log_dir, device=agent_cfg.device
  )
  runner.load(resume_path, map_location=agent_cfg.device)

  policy = runner.get_inference_policy(device=env.device)

  obs = env.get_observations()

  mjm = env.unwrapped.sim.mj_model
  mjd = env.unwrapped.sim.mj_data

  vd = mujoco.MjData(mjm)
  pert = mujoco.MjvPerturb()
  vopt = mujoco.MjvOption()
  catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

  def copy_env_to_viewer() -> None:
    mjd.qpos[:] = env.unwrapped.sim.data.qpos[0].cpu().numpy()
    mjd.qvel[:] = env.unwrapped.sim.data.qvel[0].cpu().numpy()
    mujoco.mj_forward(mjm, mjd)

  def copy_viewer_to_env() -> None:
    xfrc_applied = torch.tensor(mjd.xfrc_applied, dtype=torch.float, device=env.device)
    env.unwrapped.sim.data.xfrc_applied[:] = xfrc_applied[None]

  def key_callback(key: int) -> None:
    if key == KEY_ENTER:
      print("RESET KEY DETECTED")
      env.reset()

  viewer = mujoco.viewer.launch_passive(mjm, mjd, key_callback=key_callback)
  with viewer:
    last_frame_time = time.perf_counter()

    step = 0
    while viewer.is_running():
      frame_start = time.perf_counter()

      copy_viewer_to_env()

      actions = policy(obs)
      obs = env.step(actions)[0]

      viewer.user_scn.ngeom = 0
      env.unwrapped.update_visualizers(viewer.user_scn)
      for i in range(1, env.unwrapped.num_envs):
        vd.qpos[:] = env.unwrapped.sim.data.qpos[i].cpu().numpy()
        vd.qvel[:] = env.unwrapped.sim.data.qvel[i].cpu().numpy()
        mujoco.mj_forward(mjm, vd)
        mujoco.mjv_addGeoms(mjm, vd, vopt, pert, catmask, viewer.user_scn)

      copy_env_to_viewer()
      viewer.sync(state_only=True)

      elapsed = time.perf_counter() - frame_start
      remaining_time = FRAME_TIME - elapsed
      if remaining_time > 0.005:
        time.sleep(remaining_time - 0.003)
      while (time.perf_counter() - frame_start) < FRAME_TIME:
        pass

      step += 1
      current_time = time.perf_counter()
      if step % 60 == 0 and step > 0:
        actual_fps = 1.0 / (current_time - last_frame_time)
        print(f"Step {step}: FPS={actual_fps:.1f}")
      last_frame_time = current_time

  env.close()


if __name__ == "__main__":
  tyro.cli(main)
