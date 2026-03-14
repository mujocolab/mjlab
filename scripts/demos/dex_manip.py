"""Record dexterous-manipulation demo videos across multiple object tasks.

This script replays one trained policy checkpoint on `Mjlab-Dex-Manip` and
records one MP4 per requested object.

Examples:
  # Use latest checkpoint from a run directory and record all matching objects.
  uv run python scripts/demos/dex_manip.py \
    --run-dir logs/rsl_rl/multi_object_left-custom/2026-03-01_12-00-00_run

  # Use an explicit checkpoint and a subset of objects.
  uv run python scripts/demos/dex_manip.py \
    --checkpoint-file logs/rsl_rl/.../model_4000.pt \
    --objects water-bottle,orange,tuna-fish-can
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.dex_manip.env_cfg import apply_dex_manip_overrides
from mjlab.tasks.dex_manip.objects import parse_object_selection
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class DexManipDemoConfig:
  checkpoint_file: str | None = None
  run_dir: str | None = None
  task_id: str = "Mjlab-Dex-Manip"
  objects: str = "all"
  """Comma/semicolon/space separated object names, or 'all'."""
  output_dir: str = "assets/demos/dex_manip"
  steps: int = 400
  device: str | None = None
  num_envs: int = 1
  video_height: int | None = 720
  video_width: int | None = 1280


def _checkpoint_step(path: Path) -> int:
  match = re.search(r"model_(\d+)\.pt$", path.name)
  if match is None:
    return -1
  return int(match.group(1))


def _resolve_checkpoint(cfg: DexManipDemoConfig) -> Path:
  if cfg.checkpoint_file is not None:
    checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint.is_file():
      raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    return checkpoint

  if cfg.run_dir is None:
    raise ValueError("Pass either --checkpoint-file or --run-dir.")

  run_dir = Path(cfg.run_dir).expanduser().resolve()
  if not run_dir.is_dir():
    raise FileNotFoundError(f"Run dir not found: {run_dir}")

  candidates = [path for path in run_dir.glob("model_*.pt") if path.is_file()]
  if not candidates:
    raise FileNotFoundError(f"No model_*.pt checkpoints found under: {run_dir}")

  return max(candidates, key=lambda path: (_checkpoint_step(path), path.stat().st_mtime))


def _resolve_objects(objects: str) -> list[str]:
  return list(parse_object_selection(objects))


def _record_one_task(
  task_id: str,
  object_name: str,
  checkpoint: Path,
  out_dir: Path,
  device: str,
  steps: int,
  num_envs: int,
  video_height: int | None,
  video_width: int | None,
) -> Path:
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  apply_dex_manip_overrides(
    env_cfg,
    objects=object_name,
    envs_per_object=num_envs,
    assignment_mode="cycle",
  )

  if video_height is not None:
    env_cfg.viewer.height = video_height
  if video_width is not None:
    env_cfg.viewer.width = video_width

  name_prefix = f"{object_name}__{checkpoint.stem}"
  expected_video_path = out_dir / f"{name_prefix}-step-0.mp4"

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = VideoRecorder(
    env,
    video_folder=out_dir,
    step_trigger=lambda step: step == 0,
    video_length=steps,
    disable_logger=True,
    name_prefix=name_prefix,
  )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  policy.eval()

  obs, _ = env.reset()
  with torch.no_grad():
    for _ in range(steps):
      action = policy(obs)
      obs, _, _, _ = env.step(action)

  env.close()
  return expected_video_path


def main() -> None:
  cfg = tyro.cli(DexManipDemoConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()

  # Import tasks to populate the registry.
  import mjlab.tasks as _mjlab_tasks  # noqa: F401

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = _resolve_checkpoint(cfg)
  objects = _resolve_objects(cfg.objects)
  out_dir = Path(cfg.output_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  print(f"[INFO] checkpoint={checkpoint}")
  print(f"[INFO] device={device}")
  print(f"[INFO] task_id={cfg.task_id}")
  print(f"[INFO] objects={objects}")
  print(f"[INFO] output_dir={out_dir}")

  for object_name in objects:
    video_path = _record_one_task(
      task_id=cfg.task_id,
      object_name=object_name,
      checkpoint=checkpoint,
      out_dir=out_dir,
      device=device,
      steps=cfg.steps,
      num_envs=cfg.num_envs,
      video_height=cfg.video_height,
      video_width=cfg.video_width,
    )
    print(f"[DONE] {object_name} -> {video_path}")


if __name__ == "__main__":
  main()
