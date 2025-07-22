"""Script to train RL agent with RSL-RL."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import tyro

from rsl_rl.runners import OnPolicyRunner

# from mjlab.envs.manager_based_rl_env_config import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

_HERE = Path(__file__).parent


def main(
  num_envs: int,
  # agent_cfg: RslRlOnPolicyRunnerCfg,
):
  # env_cfg = Go1LocomotionFlatEnvCfg()
  # env_cfg.sim.num_envs = num_envs
  # env_cfg.seed = agent_cfg.seed

  agent_cfg = RslRlOnPolicyRunnerCfg()
  # from ipdb import set_trace; set_trace()

  log_root_path = _HERE / "logs" / "rsl_rl" / agent_cfg.experiment_name
  log_root_path = log_root_path.resolve()
  log_root_path.mkdir(parents=True, exist_ok=True)
  print(f"[INFO] Logging experiment in directory: {log_root_path}")

  dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  print(f"Exact experiment name requested from command line: {dirname}")
  if agent_cfg.run_name:
    dirname += f"_{agent_cfg.run_name}"
  log_dir = log_root_path / dirname

  from mjlab.tasks.go1_locomotion import Go1LocomotionFlatEnvCfg
  from mjlab.envs.manager_based_rl_env import ManagerBasedRLEnv

  env_cfg = Go1LocomotionFlatEnvCfg()
  env_cfg.sim.num_envs = num_envs
  env_cfg.seed = agent_cfg.seed
  env = ManagerBasedRLEnv(cfg=env_cfg)
  #
  # if agent_cfg.resume:
  #   resume_path = utils.get_checkpoint_path(
  #     log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
  #   )

  # Wrap the env to be compatible with RSL-RL.
  env = RslRlVecEnvWrapper(env)

  runner = OnPolicyRunner(
    env, asdict(agent_cfg), log_dir=str(log_dir), device=agent_cfg.device
  )
  # runner.add_git_repo_to_log(str(MJLAB_ROOT_PATH))
  # if agent_cfg.resume:
  #   print(f"[INFO]: Loading model checkpoint from: {resume_path}")
  #   runner.load(resume_path)

  # Dump the configuration into the log dir.
  # utils.dump_yaml(log_dir / "params" / "env.yaml", asdict(task_cfg))
  # utils.dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  # Run!
  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
  )


if __name__ == "__main__":
  tyro.cli(main, config=(tyro.conf.ConsolidateSubcommandArgs,))
