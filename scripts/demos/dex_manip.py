"""Record dexterous-manipulation demo videos for one policy or a teacher bank.

Single-policy mode replays one checkpoint and records one MP4 per requested
object. Teacher-bank mode loads one frozen policy per object, steps all objects
inside one vectorized environment, and records one combined MP4 to demonstrate
batched multi-policy inference.

Examples:
  # Use latest checkpoint from a run directory and record all matching objects.
  uv run python scripts/demos/dex_manip.py \
    --run-dir logs/rsl_rl/multi_object_left-custom/2026-03-01_12-00-00_run

  # Use an explicit checkpoint and a subset of objects.
  uv run python scripts/demos/dex_manip.py \
    --checkpoint-file logs/rsl_rl/.../model_4000.pt \
    --objects water-bottle,orange,tuna-fish-can

  # Use a frozen teacher bank and record one combined multi-object video.
  uv run python scripts/demos/dex_manip.py \
    --policy-bank-file assets/checkpoints/dex_manip_teacher_bank/legacy_20260228/policy_bank.json \
    --objects water-bottle,orange,tuna-fish-can
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.dex_manip.env_cfg import apply_dex_manip_overrides
from mjlab.tasks.dex_manip.inference import FrozenPolicyBank, object_policy_ids_from_env
from mjlab.tasks.dex_manip.objects import parse_object_selection
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import ViewerConfig


@dataclass(frozen=True)
class DexManipDemoConfig:
  checkpoint_file: str | None = None
  run_dir: str | None = None
  policy_bank: str | None = None
  """Semicolon-separated mapping: object=checkpoint_or_run_dir;..."""
  policy_bank_file: str | None = None
  """JSON mapping of object -> checkpoint path or run directory."""
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


def _resolve_checkpoint_from_path(raw_path: str) -> Path:
  path = Path(raw_path).expanduser().resolve()
  if path.is_file():
    return path

  run_dir = path
  if not run_dir.is_dir():
    raise FileNotFoundError(f"Run dir not found: {run_dir}")

  candidates = [path for path in run_dir.glob("model_*.pt") if path.is_file()]
  if not candidates:
    raise FileNotFoundError(f"No model_*.pt checkpoints found under: {run_dir}")

  return max(candidates, key=lambda path: (_checkpoint_step(path), path.stat().st_mtime))


def _resolve_checkpoint(cfg: DexManipDemoConfig) -> Path:
  if cfg.checkpoint_file is not None:
    checkpoint = _resolve_checkpoint_from_path(cfg.checkpoint_file)
    if not checkpoint.is_file():
      raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    return checkpoint

  if cfg.run_dir is None:
    raise ValueError(
      "Pass either --checkpoint-file/--run-dir for single-policy mode, "
      "or --policy-bank/--policy-bank-file for teacher-bank mode."
    )
  return _resolve_checkpoint_from_path(cfg.run_dir)


def _resolve_objects(objects: str) -> list[str]:
  return list(parse_object_selection(objects))


def _parse_policy_bank_inline(raw: str) -> dict[str, str]:
  mapping: dict[str, str] = {}
  for item in [token.strip() for token in raw.split(";") if token.strip()]:
    if "=" not in item:
      raise ValueError(
        f"Invalid policy bank item {item!r}. Expected 'object=checkpoint_or_run_dir'."
      )
    object_name, path = item.split("=", 1)
    normalized = parse_object_selection(object_name)[0]
    mapping[normalized] = path.strip()
  return mapping


def _resolve_policy_bank(cfg: DexManipDemoConfig) -> tuple[str, dict[str, Path]] | None:
  raw_mapping: dict[str, str] | None = None
  if cfg.policy_bank is not None:
    raw_mapping = _parse_policy_bank_inline(cfg.policy_bank)
  elif cfg.policy_bank_file is not None:
    bank_path = Path(cfg.policy_bank_file).expanduser().resolve()
    if not bank_path.is_file():
      raise FileNotFoundError(f"Policy bank file not found: {bank_path}")
    raw_mapping = json.loads(bank_path.read_text())
    if not isinstance(raw_mapping, dict):
      raise ValueError(f"Policy bank file must contain an object mapping: {bank_path}")

  if raw_mapping is None:
    return None

  objects = _resolve_objects(cfg.objects)
  missing = [name for name in objects if name not in raw_mapping]
  if missing:
    raise ValueError(f"Policy bank is missing objects {missing}. Available: {sorted(raw_mapping)}")

  resolved = {name: _resolve_checkpoint_from_path(str(raw_mapping[name])) for name in objects}
  return ",".join(objects), resolved


def _make_runner(
  task_id: str,
  env: RslRlVecEnvWrapper,
  checkpoint: Path,
  device: str,
):
  agent_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  return runner


def _configure_demo_playback_cfg(
  env_cfg,
  *,
  steps: int,
  video_height: int | None,
  video_width: int | None,
) -> None:
  del steps
  env_cfg.episode_length_s = 1e9
  env_cfg.terminations = {"nan": env_cfg.terminations["nan"]}
  if video_height is not None:
    env_cfg.viewer.height = video_height
  if video_width is not None:
    env_cfg.viewer.width = video_width


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
  _configure_demo_playback_cfg(
    env_cfg,
    steps=steps,
    video_height=video_height,
    video_width=video_width,
  )

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

  runner = _make_runner(task_id, env, checkpoint, device)
  policy = runner.get_inference_policy(device=device)
  policy.eval()

  obs, _ = env.reset()
  with torch.no_grad():
    for _ in range(steps):
      action = policy(obs)
      obs, _, _, _ = env.step(action)

  env.close()
  return expected_video_path


def _record_policy_bank(
  task_id: str,
  checkpoints: Mapping[str, Path],
  out_dir: Path,
  device: str,
  steps: int,
  num_envs: int,
  video_height: int | None,
  video_width: int | None,
) -> Path:
  object_names = list(checkpoints)
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  apply_dex_manip_overrides(
    env_cfg,
    objects=";".join(object_names),
    envs_per_object=num_envs,
    assignment_mode="cycle",
  )

  env_origins = []
  env_spacing = float(env_cfg.scene.env_spacing)
  total_envs = len(object_names) * num_envs
  if total_envs > 1:
    center_index = 0.5 * (total_envs - 1)
    env_origins = [((index - center_index) * env_spacing, 0.0, 0.0) for index in range(total_envs)]
  else:
    env_origins = [(0.0, 0.0, 0.0)]
  x_coords = [origin[0] for origin in env_origins]
  center_x = 0.5 * (min(x_coords) + max(x_coords))
  half_span_x = 0.5 * (max(x_coords) - min(x_coords))
  env_cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
  env_cfg.viewer.lookat = (center_x, 0.0, 0.22)
  env_cfg.viewer.distance = max(1.2, 1.2 + 0.9 * half_span_x)
  env_cfg.viewer.elevation = -18.0
  env_cfg.viewer.azimuth = 90.0
  _configure_demo_playback_cfg(
    env_cfg,
    steps=steps,
    video_height=video_height,
    video_width=video_width,
  )

  checkpoint_slug = "__".join(
    f"{object_name}-{checkpoint.stem}" for object_name, checkpoint in checkpoints.items()
  )
  name_prefix = f"teacher_bank__{checkpoint_slug}"
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

  runners = {
    object_name: _make_runner(task_id, env, checkpoint, device)
    for object_name, checkpoint in checkpoints.items()
  }
  policies = [runners[object_name].get_inference_policy(device=device).eval() for object_name in object_names]
  bank = FrozenPolicyBank(object_names, policies)

  obs, _ = env.reset()
  env_policy_ids = object_policy_ids_from_env(env.unwrapped, object_names)
  counts = torch.bincount(env_policy_ids, minlength=len(object_names)).tolist()
  print(f"[INFO] teacher_bank_env_counts={dict(zip(object_names, counts, strict=True))}")

  with torch.no_grad():
    for _ in range(steps):
      action = bank(obs, env_policy_ids)
      obs, _, dones, _ = env.step(action)
      if torch.any(dones):
        env_policy_ids = object_policy_ids_from_env(env.unwrapped, object_names)

  env.close()
  return expected_video_path


def main() -> None:
  cfg = tyro.cli(DexManipDemoConfig, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()

  # Import tasks to populate the registry.
  import mjlab.tasks as _mjlab_tasks  # noqa: F401

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  objects = _resolve_objects(cfg.objects)
  out_dir = Path(cfg.output_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  print(f"[INFO] device={device}")
  print(f"[INFO] task_id={cfg.task_id}")
  print(f"[INFO] objects={objects}")
  print(f"[INFO] output_dir={out_dir}")

  policy_bank = _resolve_policy_bank(cfg)
  if policy_bank is not None:
    _, checkpoints = policy_bank
    print(f"[INFO] teacher_bank={checkpoints}")
    video_path = _record_policy_bank(
      task_id=cfg.task_id,
      checkpoints=checkpoints,
      out_dir=out_dir,
      device=device,
      steps=cfg.steps,
      num_envs=cfg.num_envs,
      video_height=cfg.video_height,
      video_width=cfg.video_width,
    )
    print(f"[DONE] teacher_bank -> {video_path}")
  else:
    checkpoint = _resolve_checkpoint(cfg)
    print(f"[INFO] checkpoint={checkpoint}")
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
