"""Interactive viewer: run a trained checkpoint against ``go2_eval_env_cfg``
with live thermal/battery state monitoring.

Launches the Viser browser-based viewer, which renders the 3D simulation
alongside a live "Metrics" tab that plots every registered ``cfg.metrics``
term each step -- joint temperatures, battery SoC, cumulative energy and
distance, etc. That tab is generic to whatever's registered (see
``ViserTermOverlays.setup_tabs`` in ``src/mjlab/viewer/viser/overlays.py``),
so nothing here needs to know about thermal/battery specifically. The
native/mujoco viewer only plots reward terms, not metrics, so it can't show
this state and isn't offered here.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from heat_bench.envs_mjlab.go2_eval_env_cfg import go2_eval_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer

REFERENCE_TASK_ID = "Mjlab-Velocity-Rough-Unitree-Go1"


@dataclass(frozen=True)
class PlayConfig:
  """Configuration for the interactive heat_bench viewer."""

  checkpoint_file: str | None = None
  """Path to a local checkpoint file. Mutually exclusive with wandb_run_path."""
  wandb_run_path: str | None = None
  """W&B run path ('entity/project/run_id') to resolve a checkpoint from."""
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run (e.g. 'model_4000.pt')."""
  num_envs: int = 4
  """Small by default -- this is for interactive viewing, not batch eval."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""


def run_play(cfg: PlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = go2_eval_env_cfg(play=True)
  agent_cfg = load_rl_cfg(REFERENCE_TASK_ID)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
  elif cfg.wandb_run_path is not None:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    resume_path, _ = get_wandb_checkpoint_path(
      log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
    )
  else:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    resume_path = get_checkpoint_path(log_root_path)
  print(f"[INFO] Loading checkpoint: {resume_path}")

  runner_cls = load_runner_cls(REFERENCE_TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  print(
    "[INFO] Launching Viser viewer -- see the 'Metrics' tab for live "
    "thermal/battery state."
  )
  ViserPlayViewer(env, policy).run()
  env.close()


def main():
  cfg = tyro.cli(PlayConfig)
  if cfg.checkpoint_file is None and cfg.wandb_run_path is None:
    print("Must pass --checkpoint-file or --wandb-run-path.")
    sys.exit(1)
  run_play(cfg)


if __name__ == "__main__":
  main()
