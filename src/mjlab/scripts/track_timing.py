"""Run tracking task with zero-action policy and report timing breakdowns."""

from __future__ import annotations

import cProfile
import json
from pathlib import Path
from typing import Any

import torch
import tyro
from prettytable import PrettyTable

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
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

  table = PrettyTable()
  table.field_names = ["Component", "Time (ms)", "Percent"]
  table.align["Component"] = "l"
  table.align["Time (ms)"] = "r"
  table.align["Percent"] = "r"
  table.float_format = "0.3"

  for idx, (depth, name, value) in enumerate(tree_rows):
    percent = (value / total) * 100.0 if total > 0 else 0.0
    display_name = ("  " * depth) + name
    row = [display_name, value * 1e3, f"{percent:0.1f}%"]
    if idx < highlight_top:
      table.add_row([f"*{row[0]}", row[1], row[2]])
    else:
      table.add_row(row)

  print_info(table.get_string())

  if highlight_top > 0 and tree_rows:
    summary = ", ".join(
      f"{name} ({value * 1e3:.2f} ms, {(value / total) * 100.0 if total > 0 else 0.0:0.1f}%)"
      for _, name, value in tree_rows[:highlight_top]
    )
    print_info(f"Top contributors: {summary}")


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


def _build_timing_tree_rows(timings: dict[str, float]) -> list[tuple[int, str, float]]:
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

  rows: list[tuple[int, str, float]] = []

  def traverse(node: dict[str, Any], depth: int) -> None:
    sorted_children = sorted(  # type: ignore[call-arg]
      node["children"].values(),
      key=lambda child: child["total"],
      reverse=True,
    )
    for child in sorted_children:
      value = (
        float(child["value"]) if child["value"] is not None else float(child["total"])
      )
      rows.append((depth, child["label"], value))
      traverse(child, depth + 1)

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
      default_motion = Path("artifacts/lafan_cartwheel:v0/motion.npz").resolve()
      if not default_motion.exists():
        raise FileNotFoundError(
          "Motion file not provided and default motion file missing at "
          f"{default_motion}"
        )
      env_cfg.commands.motion.motion_file = str(default_motion)
  if num_envs is not None:
    env_cfg.scene.num_envs = num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  action_dim = env.single_action_space.shape
  if not isinstance(action_dim, tuple) or len(action_dim) != 1:
    raise RuntimeError(
      f"Expected single-action space to be a flat Box; received shape: {action_dim}"
    )
  batched_action_shape = (env.num_envs, action_dim[0])

  zero_action = torch.zeros(
    batched_action_shape,
    device=env.device,
    dtype=torch.float32,
  )

  def run_rollout() -> tuple[float, dict[str, float]]:
    env.reset()
    if reset_stats:
      env.reset_timing_stats()

    reward_sum = 0.0
    for step in range(num_steps):
      _, reward, _, _, _ = env.step(zero_action)
      reward_sum += reward.mean().item()

      if step == 0 and reset_stats:
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

  prefixes_to_exclude = set(GROUP_LABEL_OVERRIDES.keys())

  filtered_average_timings = _filter_timings(average_timings, prefixes_to_exclude)

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

  _print_section("Average Step Timing")
  avg_rows = _format_timings(filtered_average_timings)
  avg_rows_dict = dict(avg_rows)
  avg_total = average_timings.get("total_step", sum(avg_rows_dict.values()))
  _print_table(avg_rows, total=avg_total)

  grouped_timings = _group_timings_by_root(average_timings)
  for root, timings in sorted(
    grouped_timings.items(), key=lambda item: sum(item[1].values()), reverse=True
  ):
    if root == "total_step":
      continue
    label = GROUP_LABEL_OVERRIDES.get(root, f"Average {root}")
    section_title = label if label.startswith("Average") else f"Average {label}"
    _print_section(section_title)
    rows = _format_timings(timings)
    total = sum(timings.values())
    _print_table(rows, total=total)

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
      lines = ["| Component | Time (ms) | Percent |", "| --- | ---: | ---: |"]
      tree_rows = _build_timing_tree_rows(dict(rows))
      for depth, name, value in tree_rows:
        percent = (value / total) * 100.0 if total > 0 else 0.0
        display_name = "&nbsp;&nbsp;" * depth + name
        lines.append(f"| {display_name} | {_format_seconds(value)} | {percent:0.1f}% |")
      if tree_rows:
        top_summary = ", ".join(
          f"{name} ({value * 1e3:.2f} ms, {(value / total) * 100.0 if total > 0 else 0.0:0.1f}%)"
          for _, name, value in tree_rows[:3]
        )
        lines.append("")
        lines.append(f"Top contributors: {top_summary}")
      return lines

    md_lines.append("## Average Step Timing")
    md_lines.extend(_md_table(avg_rows, avg_total))
    md_lines.append("")

    grouped_timings = _group_timings_by_root(average_timings)
    for root, timings in sorted(
      grouped_timings.items(), key=lambda item: sum(item[1].values()), reverse=True
    ):
      if root == "total_step":
        continue
      label = GROUP_LABEL_OVERRIDES.get(root, root)
      md_lines.append(f"## Average {label}")
      rows = _format_timings(timings)
      total = sum(timings.values())
      md_lines.extend(_md_table(rows, total))
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
