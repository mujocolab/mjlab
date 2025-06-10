"""Script to train RL agent with RSL-RL."""

# fmt: off
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_debug_nans", True)  # Throw error on NaN.
# fmt: on

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated

import torch
import tyro

from rsl_rl.runners import OnPolicyRunner

from mjlab import TaskConfigUnion
from mjlab._src import registry
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

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = log_root_path / log_dir

    task_name = registry.get_task_name_by_config_class_name(task_cfg.__class__.__name__)
    env = registry.make(task_name, task_cfg)

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

    # Write git state to logs.
    runner.add_git_repo_to_log(__file__)

    # Dump the configuration into the log dir.
    utils.dump_yaml(log_dir / "params" / "env.yaml", asdict(task_cfg))
    utils.dump_yaml(log_dir / "params" / "agent.yaml", asdict(agent_cfg))

    # Run!
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.ConsolidateSubcommandArgs,))
