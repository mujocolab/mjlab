from dataclasses import asdict
from pathlib import Path
from mjlab.rl import RslRlVecEnvWrapper
import torch
import time
import mujoco
import tyro
import mujoco.viewer

from mjlab.tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import gymnasium as gym

from rl import utils

_HERE = Path(__file__).parent
_TASK_NAME = "Tracking-Flat-T1-Play-v0"
FRAME_TIME = 1.0 / 60.0
KEY_BACKSPACE = 259
KEY_ENTER = 257


def main(
  wandb_run_path: Path,
  task: str = _TASK_NAME,
) -> None:
  env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(task, "rl_cfg_entry_point")

  env = gym.make(task, cfg=env_cfg)
  env = RslRlVecEnvWrapper(env)

  log_root_path = _HERE / "rl" / "logs" / "rsl_rl" / agent_cfg.experiment_name
  log_root_path = log_root_path.resolve()
  resume_path = utils.get_wandb_checkpoint_path(log_root_path, wandb_run_path)

  log_dir = resume_path.parent
  runner = OnPolicyRunner(
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
