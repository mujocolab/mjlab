import onnxruntime as rt
import numpy as np
from mjlab.rl import RslRlVecEnvWrapper
import torch
from pathlib import Path
import time
import mujoco
import mujoco.viewer as viewer

from mjlab.tasks.utils.parse_cfg import load_cfg_from_registry
import gymnasium as gym

TASK_NAME = "Mjlab-Velocity-Flat-Unitree-Go1-v0"
FRAME_TIME = 1.0 / 60.0
KEY_BACKSPACE = 259
KEY_ENTER = 257


class Controller:
  def __init__(self):
    self._output_names = ["continuous_actions"]
    policy_path = str(Path(__file__).parent / "go1_policy.onnx")
    self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

  @torch.inference_mode()
  def __call__(self, obs: dict) -> np.ndarray:
    ret = []
    policy_obs = obs["policy"].cpu().numpy()
    for i in range(policy_obs.shape[0]):
      onnx_input = {"obs": policy_obs[i][None]}
      ret.append(self._policy.run(self._output_names, onnx_input)[0])
    return np.concatenate(ret)


if __name__ == "__main__":
  env_cfg = load_cfg_from_registry(TASK_NAME, "env_cfg_entry_point")
  env_cfg.sim.num_envs = 1
  env_cfg.observations.policy.enable_corruption = False

  env = gym.make(TASK_NAME, cfg=env_cfg)

  env = RslRlVecEnvWrapper(env)

  policy = Controller()

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

  controller = Controller()

  viewer = viewer.launch_passive(mjm, mjd, key_callback=key_callback)
  with viewer:
    last_frame_time = time.perf_counter()

    step = 0
    while viewer.is_running():
      frame_start = time.perf_counter()

      copy_viewer_to_env()

      action = controller(obs)
      obs, reward, *_ = env.step(torch.tensor(action, device=env.device))

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
