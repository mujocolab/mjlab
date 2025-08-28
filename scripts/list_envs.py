"""Script to list MJLab environments."""

import gymnasium as gym
from prettytable import PrettyTable

import mjlab.tasks  # noqa: F401 to register environments - NEED TO BE REMOVED LATER (see PR #18)


def main():
  """Print all environments registered whose id contains `Mjlab-`."""
  prefix_substring = "Mjlab-"

  table = PrettyTable(["#", "Task ID", "Entry Point", "env_cfg_entry_point"])
  table.title = "Available Environments in Mjlab"
  table.align["Task ID"] = "l"
  table.align["Entry Point"] = "l"
  table.align["env_cfg_entry_point"] = "l"

  idx = 0
  for spec in gym.registry.values():
    try:
      if prefix_substring in spec.id:
        env_cfg_ep = spec.kwargs.get("env_cfg_entry_point", "")
        table.add_row([idx + 1, spec.id, spec.entry_point, env_cfg_ep])
        idx += 1
    except Exception:
      continue

  print(table)
  if idx == 0:
    print(f"[INFO] No tasks matched filter: '{prefix_substring}'")
  return idx


if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    raise RuntimeError(f"Error listing environments: {e}") from e
