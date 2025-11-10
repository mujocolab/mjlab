# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by MjLab developers:
#   - 2025-11-01: Removed verbose print statements from load_cfg_from_registry()
#     (lines 91, 108) to reduce startup output noise during configuration parsing.
#   - 2025-11-09: Replaced gymnasium registry with mjlab custom registry.

"""Sub-module with utilities for parsing and loading configurations."""

import collections
import importlib
import inspect
import os
import yaml

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import get_rl_cfg_entry_point


def load_cfg_from_registry(task_name: str, entry_point_key: str) -> object:
  """Load default configuration given its entry point from mjlab registry.

  This function loads the configuration object from the mjlab registry for the given task name.
  It supports both YAML and Python configuration files.

  Args:
      task_name: The name of the environment.
      entry_point_key: The entry point key to resolve the configuration file.
          - "env_cfg_entry_point" for environment config
          - "rl_cfg_entry_point" for RL config

  Returns:
      The parsed configuration object. If the entry point is a YAML file, it is parsed into a dictionary.
      If the entry point is a Python class, it is instantiated and returned.

  Raises:
      ValueError: If the entry point key is not available in the mjlab registry for the task.
  """
  # Handle task name with version suffix (e.g., "Mjlab-Velocity-Rough-Unitree-Go1:v1")
  task_id = task_name.split(":")[0]

  # Get the entry point string from mjlab registry
  if entry_point_key == "env_cfg_entry_point":
    # This is handled directly in mjlab.tasks.registry.make_env()
    raise ValueError(
      f"Use mjlab.tasks.registry.make_env() for environment creation, not load_cfg_from_registry()"
    )
  elif entry_point_key == "rl_cfg_entry_point":
    cfg_entry_point = get_rl_cfg_entry_point(task_id)
  else:
    raise ValueError(
      f"Unknown entry_point_key: '{entry_point_key}'. "
      f"Valid options: 'env_cfg_entry_point', 'rl_cfg_entry_point'"
    )

  # check if entry point exists
  if cfg_entry_point is None:
    raise ValueError(
      f"Could not find configuration for the environment: '{task_name}'."
      f"\nPlease check that the task is registered in mjlab.tasks.registry"
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
    if callable(cfg_cls):
      cfg = cfg_cls()
    else:
      cfg = cfg_cls
  return cfg
