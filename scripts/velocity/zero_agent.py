import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import torch
import tyro

import mjlab.tasks  # noqa: F401
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

FRAME_TIME = 1.0 / 60.0
KEY_BACKSPACE = 259
KEY_ENTER = 257


def main(
  task: str,
  num_envs: int | None = None,
  device: str | None = None,
):
  env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")

  env_cfg.sim.num_envs = num_envs or env_cfg.sim.num_envs
  env_cfg.sim.device = device or env_cfg.sim.device

  env = gym.make(task, cfg=env_cfg)

  print(f"[INFO]: Gym observation space: {env.observation_space}")
  print(f"[INFO]: Gym action space: {env.action_space}")

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
    xfrc_applied = torch.tensor(
      mjd.xfrc_applied, dtype=torch.float, device=env.unwrapped.device
    )
    env.unwrapped.sim.data.xfrc_applied[:] = xfrc_applied[None]

  def key_callback(key: int) -> None:
    if key == KEY_ENTER:
      print("RESET KEY DETECTED")
      env.reset()

  viewer = mujoco.viewer.launch_passive(mjm, mjd, key_callback=key_callback)
  env.reset()

  with viewer:
    last_frame_time = time.perf_counter()

    step = 0
    while viewer.is_running():
      frame_start = time.perf_counter()

      copy_viewer_to_env()

      actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
      env.step(actions)

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
