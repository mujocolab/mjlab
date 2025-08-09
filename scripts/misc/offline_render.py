import onnxruntime as rt
import numpy as np
from mjlab.rl import RslRlVecEnvWrapper
import torch
from pathlib import Path

from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
import gymnasium as gym

TASK_NAME = "Mjlab-Velocity-Flat-Unitree-Go1-v0"


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
  env_cfg.sim.render.camera = "robot/tracking"
  env_cfg.sim.render.enable_shadows = True
  env_cfg.observations.policy.enable_corruption = False

  env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array")

  video_kwargs = {
    "video_folder": "videos",
    "step_trigger": lambda step: step == 0,
    "video_length": 200,
    "disable_logger": True,
  }
  env = gym.wrappers.RecordVideo(env, **video_kwargs)

  env = RslRlVecEnvWrapper(env)
  policy = Controller()

  obs = env.get_observations()
  for _ in range(200):
    actions = policy(obs)
    obs, _, _, _ = env.step(torch.tensor(actions, device=env.device))

  env.close()
