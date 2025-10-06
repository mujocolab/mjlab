"""Run tracking task with zero-action policy and report timing breakdowns."""

from __future__ import annotations

import cProfile
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import tyro
from prettytable import PrettyTable
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.logging import print_info

GROUP_LABEL_OVERRIDES: dict[str, str] = {
  "reward_manager.compute": "Reward Manager Terms",
  "termination_manager.compute": "Termination Manager Terms",
  "observation_manager.compute": "Observation Manager Terms",
  "command_manager.compute": "Command Manager Terms",
  "action_manager": "Action Manager Steps",
  "event_manager": "Event Manager Steps",
  "reset": "Reset Operations",
}


# ANSI color codes for better table readability
class Colors:
  RESET = "\033[0m"
  BOLD = "\033[1m"
  DIM = "\033[2m"

  # Component type colors
  MANAGER = "\033[94m"  # Blue
  COMPUTE = "\033[92m"  # Green
  STEP = "\033[93m"  # Yellow
  RESET_OP = "\033[95m"  # Magenta

  # Performance colors
  HIGH = "\033[91m"  # Red (high time)
  MEDIUM = "\033[93m"  # Yellow (medium time)
  LOW = "\033[92m"  # Green (low time)

  # Tree structure colors
  TREE = "\033[90m"  # Gray for tree symbols


def _get_component_color(component_name: str) -> str:
  """Get color code based on component type."""
  if "manager" in component_name.lower():
    return Colors.MANAGER
  elif "compute" in component_name.lower():
    return Colors.COMPUTE
  elif "step" in component_name.lower():
    return Colors.STEP
  elif "reset" in component_name.lower():
    return Colors.RESET_OP
  return Colors.RESET


def _get_performance_color(percent: float) -> str:
  """Get color code based on performance impact."""
  if percent >= 20.0:
    return Colors.HIGH
  elif percent >= 5.0:
    return Colors.MEDIUM
  else:
    return Colors.LOW


def _supports_color() -> bool:
  """Check if terminal supports color output."""
  return (
    hasattr(os, "getenv")
    and os.getenv("TERM") != "dumb"
    and os.getenv("NO_COLOR") is None
  )


def _print_performance_summary(
  average_timings: dict[str, float], total_step: float
) -> None:
  """Print a compact performance summary table showing only top-level components."""
  if not total_step or total_step <= 0:
    return

  # Extract only top-level manager operations (no sub-components)
  key_metrics = {}
  for key, value in average_timings.items():
    # Only include top-level manager compute operations
    if key in [
      "reward_manager.compute",
      "termination_manager.compute",
      "observation_manager.compute",
      "command_manager.compute",
      "action_manager.apply_action",
      "sim",
      "scene",
    ]:
      key_metrics[key] = value

  if not key_metrics:
    return

  _print_section("Top Performance Bottlenecks")

  table = PrettyTable()
  table.field_names = ["Component", "Time (ms)", "Percent", "Impact"]
  table.align["Component"] = "l"
  table.align["Time (ms)"] = "r"
  table.align["Percent"] = "r"
  table.align["Impact"] = "c"
  table.border = True

  use_color = _supports_color()

  # Sort by time descending
  sorted_metrics = sorted(key_metrics.items(), key=lambda x: x[1], reverse=True)

  for key, value in sorted_metrics:
    percent = (value / total_step) * 100.0

    # Clean up component names for readability
    component = (
      key.replace("_manager.compute", "").replace("_manager.", " ").replace("_", " ")
    )

    # Determine impact level
    if percent >= 20.0:
      impact = "🔴 Critical"
      if use_color:
        impact = f"{Colors.HIGH}🔴 Critical{Colors.RESET}"
    elif percent >= 10.0:
      impact = "🟠 High"
      if use_color:
        impact = f"{Colors.MEDIUM}🟠 High{Colors.RESET}"
    elif percent >= 5.0:
      impact = "🟡 Medium"
      if use_color:
        impact = f"{Colors.MEDIUM}🟡 Medium{Colors.RESET}"
    else:
      impact = "🟢 Low"
      if use_color:
        impact = f"{Colors.LOW}🟢 Low{Colors.RESET}"

    # Apply colors
    if use_color:
      perf_color = _get_performance_color(percent)
      component_color = _get_component_color(component)
      time_str = f"{perf_color}{value * 1e3:.3f}{Colors.RESET}"
      percent_str = f"{perf_color}{percent:5.1f}%{Colors.RESET}"
      component_str = f"{component_color}{component}{Colors.RESET}"
    else:
      time_str = f"{value * 1e3:.3f}"
      percent_str = f"{percent:5.1f}%"
      component_str = component

    table.add_row([component_str, time_str, percent_str, impact])

  print_info(table.get_string())


def _format_seconds(value: float) -> str:
  return f"{value * 1e3:.3f} ms"


def _print_heading(title: str) -> None:
  print_info("=" * 72)
  print_info(f"{title:^72}")
  print_info("=" * 72)


def _print_section(title: str) -> None:
  print_info("-" * 72)
  print_info(f"{title:^72}")
  print_info("-" * 72)


def _format_tree_symbols(depth: int, is_last: bool, is_leaf: bool) -> str:
  """Generate tree symbols for hierarchical display."""
  if depth == 0:
    return ""

  symbols = []
  for _i in range(depth - 1):
    symbols.append("│   ")

  if is_last:
    symbols.append("└── ")
  else:
    symbols.append("├── ")

  return "".join(symbols)


def _print_table(
  rows: list[tuple[str, float]],
  total: float | None = None,
  highlight_top: int = 3,
) -> None:
  if not rows:
    print_info("(no entries)")
    return

  if total is None:
    total = sum(value for _, value in rows) or 1.0

  tree_rows = _build_timing_tree_rows(dict(rows))
  use_color = _supports_color()

  # Create table with better formatting
  table = PrettyTable()
  table.field_names = ["Component", "Time (ms)", "Percent", "Bar"]
  table.align["Component"] = "l"
  table.align["Time (ms)"] = "r"
  table.align["Percent"] = "r"
  table.align["Bar"] = "l"

  # Set column widths for better readability
  table._max_width = {"Component": 50, "Time (ms)": 12, "Percent": 8, "Bar": 20}
  table.float_format = "0.3"

  for _idx, (depth, name, value, is_leaf, is_last) in enumerate(tree_rows):
    percent = (value / total) * 100.0 if total > 0 else 0.0

    # Create tree structure with symbols
    tree_symbols = _format_tree_symbols(depth, is_last, is_leaf)

    # Apply colors if supported
    if use_color:
      component_color = _get_component_color(name)
      perf_color = _get_performance_color(percent)
      tree_color = Colors.TREE

      display_name = f"{tree_color}{tree_symbols}{component_color}{name}{Colors.RESET}"
      time_str = f"{perf_color}{value * 1e3:.3f}{Colors.RESET}"
      percent_str = f"{perf_color}{percent:5.1f}%{Colors.RESET}"
    else:
      display_name = f"{tree_symbols}{name}"
      time_str = f"{value * 1e3:.3f}"
      percent_str = f"{percent:5.1f}%"

    # Create visual bar
    bar_length = min(20, int(percent * 0.2))  # Scale to 20 chars max
    if use_color:
      bar_color = _get_performance_color(percent)
      bar = f"{bar_color}{'█' * bar_length}{'░' * (20 - bar_length)}{Colors.RESET}"
    else:
      bar = f"{'█' * bar_length}{'░' * (20 - bar_length)}"

    # No highlighting

    table.add_row([display_name, time_str, percent_str, bar])

  # Print table with custom border
  table.border = True

  print_info(table.get_string())


def _format_timings(timings: dict[str, float]) -> list[tuple[str, float]]:
  sorted_items = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
  return sorted_items


def _filter_timings(
  timings: dict[str, float], prefixes_to_exclude: set[str]
) -> dict[str, float]:
  return {
    key: value
    for key, value in timings.items()
    if not any(key.startswith(prefix) for prefix in prefixes_to_exclude)
  }


def _build_timing_tree_rows(
  timings: dict[str, float],
) -> list[tuple[int, str, float, bool, bool]]:
  """Build tree rows with enhanced structure information.

  Returns:
    List of tuples: (depth, name, value, is_leaf, is_last_sibling)
  """
  if not timings:
    return []

  def new_node(label: str) -> dict[str, Any]:
    return {"label": label, "value": None, "children": {}, "total": 0.0}

  root = new_node("")

  for key, value in timings.items():
    path = _split_path(key)
    node = root
    for label in path:
      node = node["children"].setdefault(label, new_node(label))
    node["value"] = value

  def compute_totals(node: dict[str, Any]) -> float:
    for child in node["children"].values():
      compute_totals(child)
    if node["value"] is not None:
      node["total"] = float(node["value"])
    else:
      node["total"] = sum(child["total"] for child in node["children"].values())
    return node["total"]

  compute_totals(root)

  rows: list[tuple[int, str, float, bool, bool]] = []

  def traverse(node: dict[str, Any], depth: int, is_last: bool = True) -> None:
    sorted_children = sorted(  # type: ignore[call-arg]
      node["children"].values(),
      key=lambda child: child["total"],
      reverse=True,
    )
    for i, child in enumerate(sorted_children):
      is_last_sibling = i == len(sorted_children) - 1
      is_leaf = len(child["children"]) == 0
      value = (
        float(child["value"]) if child["value"] is not None else float(child["total"])
      )
      rows.append((depth, child["label"], value, is_leaf, is_last_sibling))
      traverse(child, depth + 1, is_last_sibling)

  traverse(root, 0)
  return rows


def _split_path(key: str) -> list[str]:
  parts = key.split(".")
  if len(parts) >= 2 and parts[0].endswith("_manager"):
    return [".".join(parts[:2])] + parts[2:]
  return parts


def _group_timings_by_root(timings: dict[str, float]) -> dict[str, dict[str, float]]:
  groups: dict[str, dict[str, float]] = {}
  for key, value in timings.items():
    path = _split_path(key)
    if not path:
      continue
    root = path[0]
    groups.setdefault(root, {})[key] = value
  return groups


def run_tracking_timing(
  task: str = "Mjlab-Tracking-Flat-Unitree-G1-Play",
  motion_file: str | None = None,
  pretrained: bool = False,
  num_steps: int = 200,
  device: str | None = None,
  num_envs: int | None = None,
  reset_stats: bool = True,
  output_json: str | None = None,
  output_markdown: str | None = None,
  profile_output: str | None = None,
) -> None:
  """Run tracking task using zero actions and print timing statistics."""

  if device is None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print_info(f"[INFO]: Using device: {device}")

  env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
  assert isinstance(env_cfg, ManagerBasedRlEnvCfg)
  env_cfg.enable_timing = True

  if isinstance(env_cfg, TrackingEnvCfg):
    if motion_file is not None:
      env_cfg.commands.motion.motion_file = motion_file
    else:
      default_motion = ensure_default_motion()
      env_cfg.commands.motion.motion_file = default_motion
  if num_envs is not None:
    env_cfg.scene.num_envs = num_envs

  # Create environment
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  # Load policy if requested
  policy = None
  wrapped_env = None
  if pretrained:
    print_info("[INFO]: Using default pretrained checkpoint")
    checkpoint_file = ensure_default_checkpoint()

    # Load agent config
    agent_cfg = load_cfg_from_registry(task, "rl_cfg_entry_point")
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

    # Wrap environment for RL
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner and load checkpoint
    if isinstance(env_cfg, TrackingEnvCfg):
      runner = MotionTrackingOnPolicyRunner(
        wrapped_env, asdict(agent_cfg), log_dir="", device=device
      )
    else:
      runner = OnPolicyRunner(wrapped_env, asdict(agent_cfg), log_dir="", device=device)

    runner.load(checkpoint_file, map_location=device)
    policy = runner.get_inference_policy(device=device)
    print_info("[INFO]: Pretrained policy loaded successfully")
  else:
    print_info("[INFO]: Using zero-action policy")

  def run_rollout() -> tuple[float, dict[str, float]]:
    if pretrained and wrapped_env is not None:
      # Use wrapped environment for pretrained policy
      current_env = wrapped_env
    else:
      # Use original environment for zero actions
      current_env = env

    obs, _ = current_env.reset()
    if reset_stats and hasattr(env, "reset_timing_stats"):
      env.reset_timing_stats()

    reward_sum = 0.0
    for step in range(num_steps):
      if policy is not None:
        # Use pretrained policy
        with torch.no_grad():
          action = policy(obs)
      else:
        # Use zero actions
        action_dim = env.single_action_space.shape
        if not isinstance(action_dim, tuple) or len(action_dim) != 1:
          raise RuntimeError(
            f"Expected single-action space to be a flat Box; received shape: {action_dim}"
          )
        batched_action_shape = (env.num_envs, action_dim[0])
        action = torch.zeros(
          batched_action_shape,
          device=env.device,
          dtype=torch.float32,
        )

      step_result = current_env.step(action)
      # Handle both 4 and 5 tuple returns
      obs, reward, terminated, truncated = step_result[:4]

      # Handle different reward types
      if hasattr(reward, "mean"):
        reward_sum += reward.mean().item()
      else:
        reward_sum += float(reward)

      if step == 0 and reset_stats and hasattr(env, "reset_timing_stats"):
        env.reset_timing_stats()

    average_timings = (
      env.get_average_step_timings() if hasattr(env, "get_average_step_timings") else {}
    )
    return reward_sum, average_timings

  _print_heading("MJLab Manager Timing Report")
  print_info(f"Task                : {task}")
  print_info(f"Device              : {device}")
  print_info(f"Num envs            : {env.num_envs}")
  print_info(f"Control decimation  : {env.cfg.decimation}")
  print_info(f"Steps to simulate   : {num_steps}")
  print_info(
    f"Policy              : {'Pretrained' if policy is not None else 'Zero-action'}"
  )
  if isinstance(env_cfg, TrackingEnvCfg):
    print_info(f"Motion file         : {env_cfg.commands.motion.motion_file}")

  if profile_output is not None:
    profiler = cProfile.Profile()
    reward_sum, average_timings = profiler.runcall(run_rollout)
    profile_path = Path(profile_output)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(profile_path)
    print_info(
      f"Saved cProfile stats to {profile_path}. View with `uvx tuna {profile_path}`"
    )
  else:
    reward_sum, average_timings = run_rollout()

  total_step_avg = average_timings.get("total_step")
  throughput: dict[str, float] | None = None
  if total_step_avg and total_step_avg > 0.0:
    vec_steps_per_sec = 1.0 / total_step_avg
    env_steps_per_sec = vec_steps_per_sec * env.num_envs
    physics_steps_per_sec = env_steps_per_sec * env.cfg.decimation
    total_duration = total_step_avg * num_steps
    throughput = {
      "total_duration_s": total_duration,
      "vec_steps_per_sec": vec_steps_per_sec,
      "env_steps_per_sec": env_steps_per_sec,
      "physics_steps_per_sec": physics_steps_per_sec,
    }

  # Show manager-level breakdown
  total_step_avg = average_timings.get("total_step")
  if total_step_avg:
    _print_section("Manager Performance (from total step)")

    # Extract manager-level timings
    manager_timings = {}
    for key, value in average_timings.items():
      if key.endswith("_manager.compute") or key in ["sim", "scene"]:
        manager_name = key.replace("_manager.compute", "").replace("_", " ").title()
        manager_timings[manager_name] = value

    if manager_timings:
      rows = _format_timings(manager_timings)
      _print_table(rows, total=total_step_avg)

    # Show sub-term breakdown for each manager
    _print_section("Sub-term Breakdown by Manager")

    # Group by manager
    manager_groups = {}
    for key, value in average_timings.items():
      if "." in key and not key.startswith("total_step"):
        parts = key.split(".")
        if len(parts) >= 2:
          manager = parts[0]
          if manager not in manager_groups:
            manager_groups[manager] = {}
          manager_groups[manager][key] = value

    # Show each manager's sub-terms
    for manager, timings in sorted(
      manager_groups.items(), key=lambda x: sum(x[1].values()), reverse=True
    ):
      manager_total = sum(timings.values())
      if manager_total > 0:
        manager_display = manager.replace("_", " ").title()
        _print_section(f"{manager_display} Sub-terms")
        rows = _format_timings(timings)
        _print_table(rows, total=manager_total)

  _print_section("Summary")
  print_info(
    f"Accumulated reward (mean per step) : {reward_sum / float(num_steps):.4f}"
  )
  print_info(f"Recorded steps                      : {num_steps}")
  if throughput is not None:
    print_info(
      f"Vec throughput                     : {throughput['vec_steps_per_sec']:8.1f} vector steps/s"
    )
    print_info(
      f"Env throughput                     : {throughput['env_steps_per_sec']:8.1f} env steps/s"
    )
    print_info(
      f"Physics throughput                 : {throughput['physics_steps_per_sec']:8.1f} physics steps/s"
    )

  metadata = {
    "task": task,
    "device": device,
    "num_envs": env.num_envs,
    "decimation": env.cfg.decimation,
    "num_steps": num_steps,
    "policy_type": "pretrained" if policy is not None else "zero_action",
    "motion_file": (
      env_cfg.commands.motion.motion_file
      if isinstance(env_cfg, TrackingEnvCfg)
      else None
    ),
    "reward_mean_per_step": reward_sum / float(num_steps),
  }
  if throughput is not None:
    metadata.update(
      {
        "total_duration_s": throughput["total_duration_s"],
        "vec_steps_per_sec": throughput["vec_steps_per_sec"],
        "env_steps_per_sec": throughput["env_steps_per_sec"],
        "physics_steps_per_sec": throughput["physics_steps_per_sec"],
      }
    )

  per_term_timings = {
    label: {
      key.removeprefix(prefix): value
      for key, value in average_timings.items()
      if key.startswith(prefix)
    }
    for prefix, label in GROUP_LABEL_OVERRIDES.items()
  }

  report: dict[str, Any] = {
    "metadata": metadata,
    "average_step_timings": average_timings,
    "average_term_timings": per_term_timings,
  }

  if output_json is not None:
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))
    print_info(f"Saved timing report JSON to {json_path}")

  if output_markdown is not None:
    md_path = Path(output_markdown)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_lines = ["# MJLab Manager Timing Report", ""]
    for key, value in metadata.items():
      md_lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
    md_lines.append("")

    def _md_table(rows: list[tuple[str, float]], total: float) -> list[str]:
      lines = [
        "| Component | Time (ms) | Percent | Bar |",
        "| --- | ---: | ---: | --- |",
      ]
      tree_rows = _build_timing_tree_rows(dict(rows))
      for depth, name, value, is_leaf, is_last in tree_rows:
        percent = (value / total) * 100.0 if total > 0 else 0.0

        # Create tree structure with symbols
        tree_symbols = _format_tree_symbols(depth, is_last, is_leaf)
        display_name = tree_symbols + name

        # Create visual bar
        bar_length = min(20, int(percent * 0.2))
        bar = "█" * bar_length + "░" * (20 - bar_length)

        lines.append(
          f"| `{display_name}` | {_format_seconds(value)} | {percent:0.1f}% | `{bar}` |"
        )
      return lines

    md_lines.append("## Manager Performance (from total step)")
    # Extract manager-level timings
    manager_timings = {}
    for key, value in average_timings.items():
      if key.endswith("_manager.compute") or key in ["sim", "scene"]:
        manager_name = key.replace("_manager.compute", "").replace("_", " ").title()
        manager_timings[manager_name] = value

    if manager_timings:
      rows = _format_timings(manager_timings)
      total_step_avg = average_timings.get("total_step", 0.0)
      md_lines.extend(_md_table(rows, total_step_avg))
    md_lines.append("")

    md_lines.append("## Sub-term Breakdown by Manager")
    # Group by manager
    manager_groups = {}
    for key, value in average_timings.items():
      if "." in key and not key.startswith("total_step"):
        parts = key.split(".")
        if len(parts) >= 2:
          manager = parts[0]
          if manager not in manager_groups:
            manager_groups[manager] = {}
          manager_groups[manager][key] = value

    # Show each manager's sub-terms
    for manager, timings in sorted(
      manager_groups.items(), key=lambda x: sum(x[1].values()), reverse=True
    ):
      manager_total = sum(timings.values())
      if manager_total > 0:
        manager_display = manager.replace("_", " ").title()
        md_lines.append(f"### {manager_display} Sub-terms")
        rows = _format_timings(timings)
        md_lines.extend(_md_table(rows, manager_total))
        md_lines.append("")

    md_lines.append("## Summary")
    md_lines.append(
      f"- **Accumulated reward (mean per step)**: {reward_sum / float(num_steps):.4f}"
    )
    md_lines.append(f"- **Recorded steps**: {num_steps}")
    if throughput is not None:
      md_lines.append(
        f"- **Vec throughput**: {throughput['vec_steps_per_sec']:0.1f} vector steps/s"
      )
      md_lines.append(
        f"- **Env throughput**: {throughput['env_steps_per_sec']:0.1f} env steps/s"
      )
      md_lines.append(
        f"- **Physics throughput**: {throughput['physics_steps_per_sec']:0.1f} physics steps/s"
      )

    md_path.write_text("\n".join(md_lines))
    print_info(f"Saved timing report Markdown to {md_path}")

  env.close()


def main() -> None:
  tyro.cli(run_tracking_timing)


if __name__ == "__main__":
  main()
