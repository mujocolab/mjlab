"""Script to list MJLab environments."""

import gymnasium as gym
import tyro
from prettytable import PrettyTable

import mjlab.tasks  # noqa: F401


def list_environments(keyword: str | None = None):
  """List all environments registered whose id contains `Mjlab-`.

  Args:
    keyword: Optional filter to only show environments containing this keyword.
  """
  prefix_substring = "Mjlab-"

  table = PrettyTable(["#", "Task ID", "Env Cfg Entry Point"])
  table.title = "Available Environments in Mjlab"
  table.align["Task ID"] = "l"
  table.align["Env Cfg Entry Point"] = "l"

  idx = 0
  for spec in gym.registry.values():
    try:
      # Check if prefix matches and optionally filter by keyword.
      if prefix_substring not in spec.id:
        continue
      if keyword and keyword.lower() not in spec.id.lower():
        continue

      env_cfg_ep = spec.kwargs.get("env_cfg_entry_point", "")
      table.add_row([idx + 1, spec.id, env_cfg_ep])
      idx += 1
    except Exception:
      continue

  print(table)
  if idx == 0:
    msg = f"[INFO] No tasks matched filter: '{prefix_substring}'"
    if keyword:
      msg += f" and keyword '{keyword}'"
    print(msg)
  return idx


def main():
  return tyro.cli(list_environments)


if __name__ == "__main__":
  main()
