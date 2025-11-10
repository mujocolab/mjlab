"""Task registry system for managing environment registration and creation."""

from dataclasses import dataclass
from typing import Any

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg


@dataclass(frozen=True)
class RegisteredTask:
    """Metadata about a registered task.

    Attributes:
        task_id: Unique task identifier (e.g., "Mjlab-Velocity-Rough-Unitree-Go1")
        env_cfg: Environment config class or callable that returns config
        rl_cfg_entry_point: Module path to RL runner config (e.g., "package.module:ConfigClass")
    """

    task_id: str
    env_cfg: Any
    rl_cfg_entry_point: str


# Private module-level registry
_REGISTRY: dict[str, RegisteredTask] = {}


def register(
    task_id: str,
    env_cfg: Any,
    rl_cfg_entry_point: str,
) -> None:
    """Register an environment task.

    Args:
        task_id: Unique task identifier (e.g., "Mjlab-Velocity-Rough-Unitree-Go1")
        env_cfg: Environment config class or callable that returns config
        rl_cfg_entry_point: Module path to RL runner config (e.g., "package.module:ConfigClass")

    Raises:
        ValueError: If task_id is already registered
    """
    if task_id in _REGISTRY:
        raise ValueError(f"Task '{task_id}' is already registered")
    _REGISTRY[task_id] = RegisteredTask(task_id, env_cfg, rl_cfg_entry_point)


def make_env(
    task_id: str,
    cfg: ManagerBasedRlEnvCfg | None = None,
    device: str = "cuda:0",
    render_mode: str | None = None,
) -> ManagerBasedRlEnv:
    """Create an environment instance.

    Args:
        task_id: Registered task identifier
        cfg: Environment config (if None, uses registered cfg)
        device: Device for computation
        render_mode: Rendering mode (None, "rgb_array", etc.)

    Returns:
        ManagerBasedRlEnv instance

    Raises:
        KeyError: If task_id is not registered
    """
    if task_id not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"Task '{task_id}' not found. Available tasks: {available}"
        )

    task = _REGISTRY[task_id]

    # Use provided config or resolve registered config
    if cfg is None:
        if callable(task.env_cfg):
            resolved_cfg: Any = task.env_cfg()
        else:
            resolved_cfg = task.env_cfg

        if isinstance(resolved_cfg, ManagerBasedRlEnvCfg):
            cfg = resolved_cfg
        else:
            raise TypeError(
                f"Expected ManagerBasedRlEnvCfg, got {type(resolved_cfg).__name__}"
            )

    return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=render_mode)


def get_task(task_id: str) -> RegisteredTask:
    """Get registered task metadata.

    Args:
        task_id: Task identifier

    Returns:
        RegisteredTask metadata

    Raises:
        KeyError: If task_id is not registered
    """
    if task_id not in _REGISTRY:
        raise KeyError(f"Task '{task_id}' not found")
    return _REGISTRY[task_id]


def list_tasks() -> list[str]:
    """List all registered task IDs.

    Returns:
        Sorted list of task identifiers
    """
    return sorted(_REGISTRY.keys())


def get_rl_cfg_entry_point(task_id: str) -> str:
    """Get the RL config entry point for a task.

    Args:
        task_id: Task identifier

    Returns:
        Entry point string for RL config

    Raises:
        KeyError: If task_id is not registered
    """
    return get_task(task_id).rl_cfg_entry_point


def load_cfg_from_registry(
    task_name: str,
    entry_point_key: str,
) -> Any:
    """Load configuration from mjlab registry.

    Replaces the gymnasium-based registry lookup with mjlab's custom registry.

    Args:
        task_name: Task identifier (may include version suffix with ":")
        entry_point_key: Configuration key to retrieve:
            - "env_cfg_entry_point" for environment config
            - "rl_cfg_entry_point" for RL runner config

    Returns:
        The loaded configuration object

    Raises:
        ValueError: If entry_point_key is not valid
        KeyError: If task is not registered
    """
    # Import here to avoid circular imports
    from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
        load_cfg_from_registry as _load_cfg_base,
    )

    # Handle version suffix (e.g., "Mjlab-Velocity-Rough-Unitree-Go1:v1" -> "Mjlab-Velocity-Rough-Unitree-Go1")
    task_id = task_name.split(":")[0]

    if entry_point_key == "env_cfg_entry_point":
        task = get_task(task_id)
        env_cfg = task.env_cfg
        if callable(env_cfg):
            return env_cfg()
        return env_cfg

    elif entry_point_key == "rl_cfg_entry_point":
        entry_point_str = get_rl_cfg_entry_point(task_id)
        # Delegate to the base implementation for parsing module paths
        return _load_cfg_base(task_id, entry_point_key)

    else:
        raise ValueError(
            f"Unknown entry_point_key: '{entry_point_key}'. "
            f"Valid options: 'env_cfg_entry_point', 'rl_cfg_entry_point'"
        )
