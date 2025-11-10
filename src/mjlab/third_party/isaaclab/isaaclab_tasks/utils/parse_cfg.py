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

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def load_cfg_from_registry(task_name: str, entry_point_key: str) -> object:
  """Load default configuration given its entry point from mjlab registry.

  This function loads the configuration object from the mjlab registry for the given task name.

  Args:
      task_name: The name of the environment.
      entry_point_key: The entry point key to resolve the configuration.
          - "env_cfg_entry_point" for environment config
          - "rl_cfg_entry_point" or "rl_cfg" for RL config

  Returns:
      The configuration object.
  """
  # Handle environment config.
  if entry_point_key == "env_cfg_entry_point":
    return load_env_cfg(task_name)
  # Handle RL config.
  elif entry_point_key in ("rl_cfg_entry_point", "rl_cfg"):
    return load_rl_cfg(task_name)
  else:
    raise ValueError(
      f"Unknown entry_point_key: '{entry_point_key}'. "
      f"Valid options: 'env_cfg_entry_point', 'rl_cfg_entry_point', 'rl_cfg'"
    )
