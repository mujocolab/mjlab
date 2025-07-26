# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with utilities for parsing and loading configurations."""

import collections
import gymnasium as gym
import importlib
import inspect
import os
import yaml

from mjlab.envs import ManagerBasedRlEnvCfg


def load_cfg_from_registry(task_name: str, entry_point_key: str) -> dict | object:
  """Load default configuration given its entry point from the gym registry.

  This function loads the configuration object from the gym registry for the given task name.
  It supports both YAML and Python configuration files.

  It expects the configuration to be registered in the gym registry as:

  .. code-block:: python

      gym.register(
          id="My-Awesome-Task-v0",
          ...
          kwargs={"env_entry_point_cfg": "path.to.config:ConfigClass"},
      )

  The parsed configuration object for above example can be obtained as:

  .. code-block:: python

      from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

      cfg = load_cfg_from_registry("My-Awesome-Task-v0", "env_entry_point_cfg")

  Args:
      task_name: The name of the environment.
      entry_point_key: The entry point key to resolve the configuration file.

  Returns:
      The parsed configuration object. If the entry point is a YAML file, it is parsed into a dictionary.
      If the entry point is a Python class, it is instantiated and returned.

  Raises:
      ValueError: If the entry point key is not available in the gym registry for the task.
  """
  # obtain the configuration entry point
  cfg_entry_point = gym.spec(task_name.split(":")[-1]).kwargs.get(entry_point_key)
  # check if entry point exists
  if cfg_entry_point is None:
    # get existing agents and algorithms
    agents = collections.defaultdict(list)
    for k in gym.spec(task_name.split(":")[-1]).kwargs:
      if k.endswith("_cfg_entry_point") and k != "env_cfg_entry_point":
        spec = (
          k.replace("_cfg_entry_point", "")
          .replace("rl_games", "rl-games")
          .replace("rsl_rl", "rsl-rl")
          .split("_")
        )
        agent = spec[0].replace("-", "_")
        algorithms = [item.upper() for item in (spec[1:] if len(spec) > 1 else ["PPO"])]
        agents[agent].extend(algorithms)
    msg = "\nExisting RL library (and algorithms) config entry points: "
    for agent, algorithms in agents.items():
      msg += f"\n  |-- {agent}: {', '.join(algorithms)}"
    # raise error
    raise ValueError(
      f"Could not find configuration for the environment: '{task_name}'."
      f"\nPlease check that the gym registry has the entry point: '{entry_point_key}'."
      f"{msg if agents else ''}"
    )
  # parse the default config file
  if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
    if os.path.exists(cfg_entry_point):
      # absolute path for the config file
      config_file = cfg_entry_point
    else:
      # resolve path to the module location
      mod_name, file_name = cfg_entry_point.split(":")
      mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
      # obtain the configuration file path
      config_file = os.path.join(mod_path, file_name)
    # load the configuration
    print(f"[INFO]: Parsing configuration from: {config_file}")
    with open(config_file, encoding="utf-8") as f:
      cfg = yaml.full_load(f)
  else:
    if callable(cfg_entry_point):
      # resolve path to the module location
      mod_path = inspect.getfile(cfg_entry_point)
      # load the configuration
      cfg_cls = cfg_entry_point()
    elif isinstance(cfg_entry_point, str):
      # resolve path to the module location
      mod_name, attr_name = cfg_entry_point.split(":")
      mod = importlib.import_module(mod_name)
      cfg_cls = getattr(mod, attr_name)
    else:
      cfg_cls = cfg_entry_point
    # load the configuration
    print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")
    if callable(cfg_cls):
      cfg = cfg_cls()
    else:
      cfg = cfg_cls
  return cfg


def parse_env_cfg(
  task_name: str,
  device: str = "cuda:0",
  num_envs: int | None = None,
  use_fabric: bool | None = None,
) -> ManagerBasedRlEnvCfg:
  """Parse configuration for an environment and override based on inputs.

  Args:
      task_name: The name of the environment.
      device: The device to run the simulation on. Defaults to "cuda:0".
      num_envs: Number of environments to create. Defaults to None, in which case it is left unchanged.
      use_fabric: Whether to enable/disable fabric interface. If false, all read/write operations go through USD.
          This slows down the simulation but allows seeing the changes in the USD through the USD stage.
          Defaults to None, in which case it is left unchanged.

  Returns:
      The parsed configuration object.

  Raises:
      RuntimeError: If the configuration for the task is not a class. We assume users always use a class for the
          environment configuration.
  """
  # load the default configuration
  cfg = load_cfg_from_registry(task_name.split(":")[-1], "env_cfg_entry_point")

  # check that it is not a dict
  # we assume users always use a class for the configuration
  if isinstance(cfg, dict):
    raise RuntimeError(
      f"Configuration for the task: '{task_name}' is not a class. Please provide a class."
    )

  # simulation device
  cfg.sim.device = device
  # disable fabric to read/write through USD
  if use_fabric is not None:
    cfg.sim.use_fabric = use_fabric
  # number of environments
  if num_envs is not None:
    cfg.scene.num_envs = num_envs

  return cfg
