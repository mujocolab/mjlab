import onnxruntime as rt
import numpy as np
import mujoco
import mujoco.viewer
import torch
import time

from mjlab.tasks.utils.parse_cfg import load_cfg_from_registry
import gymnasium as gym
# from mjlab.tasks.locomotion.velocity.config.go1.flat_env_cfg import UnitreeGo1FlatEnvCfg


class Controller:
  def __init__(self):
    self._output_names = ["continuous_actions"]
    policy_path = "/home/kevin/dev/mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/go1_policy.onnx"
    self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

  def __call__(self, obs: dict) -> np.ndarray:
    ret = []
    policy_obs = obs["policy"].cpu().numpy()
    for i in range(policy_obs.shape[0]):
      onnx_input = {"obs": policy_obs[i][None]}
      ret.append(self._policy.run(self._output_names, onnx_input)[0])
    return np.concatenate(ret)


if __name__ == "__main__":
  task_name = "Mjlab-Velocity-Flat-Unitree-Go1-v0"

  env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
  env_cfg.sim.num_envs = 1
  env_cfg.observations.policy.enable_corruption = False

  env = gym.make(task_name, cfg=env_cfg)

  mjm = env.unwrapped.sim.mj_model
  mjd = env.unwrapped.sim.mj_data

  vd = mujoco.MjData(mjm)
  pert = mujoco.MjvPerturb()
  vopt = mujoco.MjvOption()
  catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

  def copy_env_to_viewer():
    mjd.qpos[:] = env.unwrapped.sim.data.qpos[0].cpu().numpy()
    mjd.qvel[:] = env.unwrapped.sim.data.qvel[0].cpu().numpy()
    mujoco.mj_forward(mjm, mjd)

  def copy_viewer_to_env():
    xfrc_applied = torch.tensor(
      mjd.xfrc_applied, dtype=torch.float, device=env.unwrapped.device
    )
    env.unwrapped.sim.data.xfrc_applied[:] = xfrc_applied[None]

  @torch.no_grad()
  def get_zero_action():
    return torch.rand(
      (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
      device="cuda:0",
    )

  FRAME_TIME = 1.0 / 60.0

  KEY_BACKSPACE = 259
  KEY_ENTER = 257

  def key_callback(key: int) -> None:
    if key == KEY_ENTER:
      print("RESET KEY DETECTED")
      env.unwrapped.reset()

  controller = Controller()

  viewer = mujoco.viewer.launch_passive(mjm, mjd, key_callback=key_callback)
  with viewer:
    obs, extras = env.unwrapped.reset()
    copy_env_to_viewer()

    last_frame_time = time.perf_counter()

    step = 0
    while viewer.is_running():
      frame_start = time.perf_counter()

      copy_viewer_to_env()

      action = controller(obs)
      obs, reward, *_ = env.unwrapped.step(
        torch.tensor(action, device=env.unwrapped.device)
      )

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

      # Debug FPS every 60 frames
      step += 1
      current_time = time.perf_counter()
      if step % 60 == 0 and step > 0:
        actual_fps = 1.0 / (current_time - last_frame_time)
        print(f"Step {step}: FPS={actual_fps:.1f}")
      last_frame_time = current_time
