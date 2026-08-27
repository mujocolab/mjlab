"""Summarize the weekend current-only versus TemporalCNN training runs."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_ROOT = Path("logs/rsl_rl/g1_velocity_football")
OUTPUT_DIR = Path("项目管理/实验记录")
OUTPUT_CSV = OUTPUT_DIR / "历史卷积周末对照结果.csv"
OUTPUT_MD = OUTPUT_DIR / "历史卷积周末对照结果.md"
RUN_PREFIX = "E3_hist_ablation_"
FINAL_WINDOW = 100

METRICS = {
  "mean_reward": ("Train/mean_reward", "higher"),
  "ball_velocity_reward": (
    "Episode_Reward/track_ball_lin_vel_xy_exp",
    "higher",
  ),
  "ball_relative_position_reward": (
    "Episode_Reward/track_ball_relative_pos_xy_exp",
    "higher",
  ),
  "ball_out_of_control": (
    "Episode_Termination/ball_out_of_control",
    "lower",
  ),
  "velocity_error": ("Metrics/twist/error_vel_xy", "lower"),
  "action_rate_penalty": ("Episode_Reward/action_rate_l2", "higher"),
}


def _run_identity(run_dir: Path) -> tuple[str, int] | None:
  name = run_dir.name
  marker = f"_{RUN_PREFIX}"
  if marker not in name or "_seed" not in name:
    return None
  suffix = name.split(marker, maxsplit=1)[1]
  label, seed_text = suffix.rsplit("_seed", maxsplit=1)
  if label not in {"current_only", "history10_temporalcnn"}:
    return None
  try:
    return label, int(seed_text)
  except ValueError:
    return None


def _scalar_summary(
  accumulator: EventAccumulator,
  tag: str,
) -> tuple[float, float, int]:
  events = accumulator.Scalars(tag)
  if not events:
    return float("nan"), float("nan"), 0
  steps = np.asarray([event.step for event in events], dtype=np.float64)
  values = np.asarray([event.value for event in events], dtype=np.float64)
  final = float(np.mean(values[-FINAL_WINDOW:]))
  if len(values) == 1 or steps[-1] == steps[0]:
    auc = float(values[-1])
  else:
    auc = float(np.trapezoid(values, steps) / (steps[-1] - steps[0]))
  return final, auc, int(steps[-1])


def collect_runs() -> list[dict[str, float | int | str]]:
  rows: list[dict[str, float | int | str]] = []
  for run_dir in sorted(LOG_ROOT.iterdir()):
    identity = _run_identity(run_dir)
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if identity is None or not event_files:
      continue
    label, seed = identity
    accumulator = EventAccumulator(str(event_files[-1]))
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    row: dict[str, float | int | str] = {
      "run": run_dir.name,
      "label": label,
      "seed": seed,
    }
    max_step = 0
    for metric_name, (tag, _) in METRICS.items():
      if tag not in available:
        final, auc, step = float("nan"), float("nan"), 0
      else:
        final, auc, step = _scalar_summary(accumulator, tag)
      row[f"{metric_name}_final"] = final
      row[f"{metric_name}_auc"] = auc
      max_step = max(max_step, step)
    row["last_iteration"] = max_step
    rows.append(row)
  return rows


def _mean(values: list[float]) -> float:
  finite = [value for value in values if np.isfinite(value)]
  return statistics.fmean(finite) if finite else float("nan")


def write_results(rows: list[dict[str, float | int | str]]) -> None:
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  fieldnames = ["run", "label", "seed", "last_iteration"]
  for metric_name in METRICS:
    fieldnames.extend((f"{metric_name}_final", f"{metric_name}_auc"))
  with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

  grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
  for row in rows:
    grouped[str(row["label"])].append(row)

  lines = [
    "# 历史卷积周末对照结果",
    "",
    "对照组仅使用当前帧 MLP；实验组使用当前帧加 10 帧 TemporalCNN。"
    "两组的视觉矩形、5% 随机丢帧、奖励、环境数量和随机种子一致。",
    "",
    f"- 最终值：最后 {FINAL_WINDOW} 个 iteration 的均值",
    "- AUC：按训练 iteration 积分并除以训练区间长度",
    f"- 已发现运行数：{len(rows)}",
    "",
    "## 分组均值",
    "",
    "| 指标 | 方向 | 当前帧 | 10帧历史 | 历史-当前 |",
    "|---|---:|---:|---:|---:|",
  ]
  current = grouped.get("current_only", [])
  history = grouped.get("history10_temporalcnn", [])
  for metric_name, (_, direction) in METRICS.items():
    current_mean = _mean(
      [float(row[f"{metric_name}_final"]) for row in current]
    )
    history_mean = _mean(
      [float(row[f"{metric_name}_final"]) for row in history]
    )
    lines.append(
      f"| {metric_name} | {direction} | {current_mean:.5f} | "
      f"{history_mean:.5f} | {history_mean - current_mean:+.5f} |"
    )

  paired_reward_wins = 0
  paired_control_wins = 0
  for seed in (42, 43, 44):
    current_row = next(
      (row for row in current if int(row["seed"]) == seed),
      None,
    )
    history_row = next(
      (row for row in history if int(row["seed"]) == seed),
      None,
    )
    if current_row is None or history_row is None:
      continue
    if float(history_row["mean_reward_final"]) > float(
      current_row["mean_reward_final"]
    ):
      paired_reward_wins += 1
    if float(history_row["ball_out_of_control_final"]) < float(
      current_row["ball_out_of_control_final"]
    ):
      paired_control_wins += 1

  lines.extend(
    [
      "",
      "## 判定",
      "",
      f"- 历史组最终平均奖励胜出：{paired_reward_wins}/3 个随机种子。",
      f"- 历史组球失控率更低：{paired_control_wins}/3 个随机种子。",
      "- 建议判定标准：上述两项至少各胜出 2/3，并且平均奖励 AUC 不下降。",
      "",
      f"逐运行原始结果见 `{OUTPUT_CSV}`。",
      "",
    ]
  )
  OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
  rows = collect_runs()
  write_results(rows)
  print(f"Wrote {OUTPUT_CSV}")
  print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
  main()
