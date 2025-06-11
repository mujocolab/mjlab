from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

from mjlab._src import mjx_env, mjx_task


@dataclass(frozen=True)
class Task:
  name: str
  task_cls: Type[mjx_task.MjxTask]
  config_cls: Type[mjx_task.TaskConfig]


class TaskRegistry:
  def __init__(self):
    self._tasks: Dict[str, Task] = {}

  def register_task(
    self,
    task_name: str,
    task_cls: Type[mjx_task.MjxTask],
    config_cls: Type[mjx_task.TaskConfig],
  ) -> None:
    """Register a task in the registry.

    Args:
      task_name: Unique identifier for the task
      task_cls: The task class that inherits from MjxTask
      config_cls: The config class that inherits from TaskConfig
    """
    if task_name in self._tasks:
      raise ValueError(f"Task '{task_name}' is already registered!")

    task = Task(name=task_name, task_cls=task_cls, config_cls=config_cls)
    self._tasks[task_name] = task

  def get_task(self, task_name: str) -> Task:
    """Get a registered task by name.

    Args:
      task_name: Name of the task to retrieve

    Returns:
      Task object containing task metadata

    Raises:
      KeyError: If task is not registered
    """
    if task_name not in self._tasks:
      available_tasks = list(self._tasks.keys())
      raise KeyError(
        f"Task '{task_name}' not found. Available tasks: {available_tasks}"
      )
    return self._tasks[task_name]

  def list_task_names(self) -> Tuple[str, ...]:
    """List all registered task names.

    Returns:
      List of all registered task names.
    """
    return tuple(self._tasks.keys())

  def task_exists(self, task_name: str) -> bool:
    """Check if a task is registered.

    Args:
      task_name: Name of the task to check

    Returns:
      True if task exists, False otherwise
    """
    return task_name in self._tasks

  def get_task_name_by_config_class_name(self, config_class_name: str) -> str:
    """Get task name by config class name.

    Args:
      config_class_name: Name of the config class (e.g., "CartPoleConfig")

    Returns:
      Task name string

    Raises:
      KeyError: If no task with matching config class name is found
    """
    for task_name, task in self._tasks.items():
      if task.config_cls.__name__ == config_class_name:
        return task_name

    available_configs = [task.config_cls.__name__ for task in self._tasks.values()]
    raise KeyError(
      f"No task found with config class name '{config_class_name}'. "
      f"Available config class names: {available_configs}"
    )


_REGISTRY = TaskRegistry()


def make(
  task_name: str, config: Optional[mjx_task.TaskConfig] = None, **kwargs
) -> mjx_env.MjxEnv:
  """Create an environment from a registered task (similar to gym.make).

  Args:
    task_name: Name of the registered task
    config: Optional task configuration. If None, uses default config
    **kwargs: Additional keyword arguments passed to the task constructor

  Returns:
    MjxEnv instance wrapping the created task

  Raises:
    KeyError: If task is not registered
  """
  task_info = _REGISTRY.get_task(task_name)
  if config is None:
    config = task_info.config_cls()
  task_instance = task_info.task_cls(config, **kwargs)
  return mjx_env.MjxEnv(task=task_instance)


def register(
  task_name: str,
  task_class: Type[mjx_task.MjxTask],
  task_config: Type[mjx_task.TaskConfig],
) -> None:
  """Register a task (convenience function).

  Args:
    task_name: Unique identifier for the task
    task_class: The task class that inherits from MjxTask
    task_config: The config class that inherits from TaskConfig
  """
  _REGISTRY.register_task(task_name, task_class, task_config)


def list_task_names() -> Tuple[str, ...]:
  """List all registered task names (convenience function).

  Returns:
    Tuple of all registered task names.
  """
  return _REGISTRY.list_task_names()


def task_exists(task_name: str) -> bool:
  """Check if a task is registered (convenience function).

  Args:
    task_name: Name of the task to check

  Returns:
    True if task exists, False otherwise
  """
  return _REGISTRY.task_exists(task_name)


def get_task_name_by_config_class_name(config_class_name: str) -> str:
  """Get task name by config class name (convenience function).

  Args:
    config_class_name: Name of the config class (e.g., "CartPoleConfig")

  Returns:
    Task name string

  Raises:
    KeyError: If no task with matching config class name is found
  """
  return _REGISTRY.get_task_name_by_config_class_name(config_class_name)
