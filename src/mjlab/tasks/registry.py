"""Task registry system for managing environment registration and creation."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

# Private module-level registry: task_id -> (env_cfg, rl_cfg)
EnvRlCfgPair = tuple[ManagerBasedRlEnvCfg, RslRlOnPolicyRunnerCfg]
_REGISTRY: dict[str, EnvRlCfgPair] = {}


def register_mjlab_task(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlOnPolicyRunnerCfg,
) -> None:
  """Register an environment task.

  Args:
      task_id: Unique task identifier (e.g., "Mjlab-Velocity-Rough-Unitree-Go1").
      env_cfg: Environment configuration (instance or callable that returns one).
      rl_cfg: RL runner configuration.
  """
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered")
  _REGISTRY[task_id] = (env_cfg, rl_cfg)


def list_tasks() -> list[str]:
  """List all registered task IDs.

  Returns:
      Sorted list of task identifiers.
  """
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str) -> ManagerBasedRlEnvCfg:
  """Load environment configuration for a task.

  Args:
      task_name: Task identifier.

  Returns:
      Environment configuration.
  """
  env_cfg, _ = _REGISTRY[task_name]

  if callable(env_cfg):
    return env_cfg()
  return env_cfg


def load_rl_cfg(task_name: str) -> RslRlOnPolicyRunnerCfg:
  """Load RL configuration for a task.

  Args:
      task_name: Task identifier.

  Returns:
      RL runner configuration.
  """
  _, rl_cfg = _REGISTRY[task_name]
  return rl_cfg
