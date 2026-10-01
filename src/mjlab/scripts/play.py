"""Script to play RL agent with RSL-RL."""

import os
import sys
import time as _time
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path, get_wandb_env_yaml_path, load_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""
  env_yaml: str | None = None
  """Path to the env.yaml used during training, for restoring the observation
  configuration (e.g. history lengths). Auto-resolved from the checkpoint directory
  or W&B run when not provided. Use this to override the default resolution."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path: str | Path) -> None:
  """Load env.yaml and apply observation configuration to env_cfg.

  Reads the ``observations`` section of the YAML file saved during training and
  applies per-term and group-level settings (``history_length``,
  ``flatten_history_dim`` etc) to *env_cfg*, so the environment is constructed with
  the same observation layout as during training.

  Raises:
    FileNotFoundError: If *env_yaml_path* does not exist.
    RuntimeError: If a group or term present in the YAML is absent from env_cfg, or
      if the ``func`` reference for a term does not match.
  """

  env_yaml_path = Path(env_yaml_path)
  if not env_yaml_path.exists():
    raise FileNotFoundError(
      f"env.yaml not found at '{env_yaml_path}'. "
      "Pass --env-yaml <path> to specify it explicitly."
    )

  env_yaml = load_yaml(env_yaml_path)

  yaml_observations: dict = env_yaml.get("observations") or {}

  def _qualname(func) -> str:
    return f"{func.__module__}.{func.__qualname__}"

  for group_name, yaml_group in yaml_observations.items():
    if group_name not in env_cfg.observations:
      raise RuntimeError(
        f"Observation group '{group_name}' found in env.yaml is missing from "
        f"env_cfg. Available groups: {list(env_cfg.observations.keys())}"
      )

    group_cfg = env_cfg.observations[group_name]

    # Apply all group-level fields present in the YAML, skipping 'terms' which
    # is handled below.
    _GROUP_SKIP = {"terms"}
    group_fields = {f.name for f in fields(group_cfg)}
    for field_name in group_fields:
      if field_name in _GROUP_SKIP or field_name not in yaml_group:
        continue
      setattr(group_cfg, field_name, yaml_group[field_name])

    yaml_terms: dict = yaml_group.get("terms") or {}

    # Drop terms that are in env_cfg but absent from the YAML — they were not
    # present during training and would corrupt the observation dimension.
    extra_terms = set(group_cfg.terms.keys()) - set(yaml_terms.keys())
    for term_name in extra_terms:
      print(
        f"[WARN]: Dropping obs term '{group_name}.{term_name}' not present in "
        f"env.yaml — it was not used during training."
      )
      del group_cfg.terms[term_name]

    for term_name, yaml_term in yaml_terms.items():
      if term_name not in group_cfg.terms:
        raise RuntimeError(
          f"Observation term '{group_name}.{term_name}' found in env.yaml is "
          f"missing from env_cfg. Available terms: {list(group_cfg.terms.keys())}"
        )

      term_cfg = group_cfg.terms[term_name]
      yaml_func = yaml_term.get("func")

      # Validate that the function reference hasn't changed.
      if yaml_func is not None and term_cfg.func is not None:
        yaml_qn = _qualname(yaml_func)
        cfg_qn = _qualname(term_cfg.func)
        if yaml_qn != cfg_qn:
          raise RuntimeError(
            f"Observation term '{group_name}.{term_name}': func mismatch. "
            f"env.yaml references '{yaml_qn}' but env_cfg has '{cfg_qn}'."
          )

      # Apply all term-level fields present in the YAML, skipping 'func' and
      # 'params' which come from the task config and are only validated, not
      # overwritten.
      _TERM_SKIP = {"func", "params"}
      term_fields = {f.name for f in fields(term_cfg)}
      for field_name in term_fields:
        if field_name in _TERM_SKIP or field_name not in yaml_term:
          continue
        original = getattr(term_cfg, field_name)
        yaml_val = yaml_term[field_name]
        # If the existing field is a dataclass and the YAML value is a dict,
        # update the nested dataclass properties in-place rather than replacing
        # the whole object with a raw dict.
        if is_dataclass(original) and isinstance(yaml_val, dict):
          nested_fields = {f.name for f in fields(original) if f.init}
          for k, v in yaml_val.items():
            if k in nested_fields:
              nested_original = getattr(original, k)
              setattr(original, k, v)
              if nested_original != v:
                print(
                  f"  {group_name}.{term_name}.{field_name}.{k} = {v!r} (was {nested_original!r})"
                )
        else:
          setattr(term_cfg, field_name, yaml_val)
          if original != yaml_val:
            print(
              f"  {group_name}.{term_name}.{field_name} = {yaml_val!r} (was {original!r})"
            )

  print(f"[INFO]: Applied observation config from env.yaml: {env_yaml_path}")


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

    # Resolve env.yaml: explicit CLI path > auto-detect from run dir / W&B.
    if cfg.env_yaml is not None:
      env_yaml_path: Path = Path(cfg.env_yaml)
    elif cfg.wandb_run_path is not None:
      env_yaml_path = get_wandb_env_yaml_path(log_root_path, Path(cfg.wandb_run_path))
    else:
      env_yaml_path = resume_path.parent / "params" / "env.yaml"
    _apply_obs_cfg_from_env_yaml(env_cfg, env_yaml_path)

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _ckpt_runner.get_inference_policy(device=device)

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  maybe_print_top_level_help("play")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
