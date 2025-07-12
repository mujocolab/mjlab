"""Script to train RL agent with RSL-RL."""

import jax
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import tyro

from mjlab.rl.on_policy_runner import MjlabOnPolicyRunner as OnPolicyRunner

from mjlab import MJLAB_ROOT_PATH
from mjlab.tasks import TaskConfigUnion
from mjlab.tasks import registry
from mjlab.rl import config, utils, wrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

_HERE = Path(__file__).parent


def main(
  task_cfg: TaskConfigUnion,
  num_envs: int,
  agent_cfg: config.OnPolicyRunnerConfig,
):
  """Train an RL agent with RSL-RL.

  Args:
    task: The task to train on.
    num_envs: The number of environments to run in parallel.
    agent_cfg: The configuration for the RL agent.
  """
  log_root_path = _HERE / "logs" / "rsl_rl" / agent_cfg.experiment_name
  log_root_path = log_root_path.resolve()
  log_root_path.mkdir(parents=True, exist_ok=True)
  print(f"[INFO] Logging experiment in directory: {log_root_path}")

  dirname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  print(f"Exact experiment name requested from command line: {dirname}")
  if agent_cfg.run_name:
    dirname += f"_{agent_cfg.run_name}"
  log_dir = log_root_path / dirname

  task_name = registry.get_task_name_by_config_class_name(task_cfg.__class__.__name__)
  env = registry.make(task_name, task_cfg)

  if agent_cfg.resume:
    resume_path = utils.get_checkpoint_path(
      log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )

  # Wrap the env to be compatible with RSL-RL.
  env = wrapper.RslRlVecEnvWrapper(
    env,
    num_envs=num_envs,
    seed=agent_cfg.seed,
    clip_actions=agent_cfg.clip_actions,
    device=agent_cfg.device,
  )

  runner = OnPolicyRunner(
    env=env,
    train_cfg=asdict(agent_cfg),
    log_dir=log_dir,
    device=agent_cfg.device,
  )
  runner.add_git_repo_to_log(str(MJLAB_ROOT_PATH))
  if agent_cfg.resume:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

  # Dump the configuration into the log dir.
  utils.dump_yaml(log_dir / "params" / "env.yaml", asdict(task_cfg))
  utils.dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

  # Run!
  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
  )


if __name__ == "__main__":
  jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
  # jax.config.update("jax_debug_nans", True)  # Throw error on NaN.

  tyro.cli(main, config=(tyro.conf.ConsolidateSubcommandArgs,))
