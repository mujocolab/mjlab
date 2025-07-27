"""Script to train RL agent with RSL-RL."""

from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import os
import torch
import tyro
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from mjlab.tasks.utils.parse_cfg import load_cfg_from_registry
import gymnasium as gym

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

_HERE = Path(__file__).parent
_TASK = "Mjlab-Velocity-Flat-Unitree-Go1-v0"


def main(
  task: str = _TASK,
  num_envs: int | None = None,
  agent_cfg: RslRlOnPolicyRunnerCfg = RslRlOnPolicyRunnerCfg(),
  seed: int | None = None,
  max_iterations: int | None = None,
  device: str | None = None,
):
  env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
  env_cfg.sim.num_envs = num_envs if num_envs is not None else env_cfg.sim.num_envs
  agent_cfg.max_iterations = (
    max_iterations if max_iterations is not None else agent_cfg.max_iterations
  )
  agent_cfg.seed = seed if seed is not None else agent_cfg.seed

  env_cfg.seed = agent_cfg.seed
  env_cfg.sim.device = device if device is not None else env_cfg.sim.device

  log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
  log_root_path = os.path.abspath(log_root_path)
  print(f"[INFO] Logging experiment in directory: {log_root_path}")
  log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  print(f"Exact experiment name requested from command line: {log_dir}")
  if agent_cfg.run_name:
    log_dir += f"_{agent_cfg.run_name}"
  log_dir = os.path.join(log_root_path, log_dir)

  env = gym.make(task, cfg=env_cfg)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = OnPolicyRunner(
    env, asdict(agent_cfg), log_dir=log_dir, device=agent_cfg.device
  )
  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
  )
  env.close()


if __name__ == "__main__":
  tyro.cli(main)
