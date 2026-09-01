import os
from pathlib import Path

import torch
from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from mjlab.rl.multi_teacher import MultiTeacherModel
from mjlab.rl.utils import clean_model_cfg
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


def _migrate_checkpoint(loaded_dict: dict, path: str) -> None:
  """Migrate legacy checkpoints in place to the current rsl-rl 5.x format.

  Handles pre-4.0 checkpoints (``model_state_dict`` with ``actor.*`` /
  ``actor_obs_normalizer.*`` keys) and 4.x actor std keys.
  """
  if "model_state_dict" in loaded_dict:
    print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
    model_state_dict = loaded_dict.pop("model_state_dict")
    actor_state_dict = {}
    critic_state_dict = {}

    for key, value in model_state_dict.items():
      # Migrate actor keys.
      if key.startswith("actor."):
        new_key = key.replace("actor.", "mlp.")
        actor_state_dict[new_key] = value
      elif key.startswith("actor_obs_normalizer."):
        new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
        actor_state_dict[new_key] = value
      elif key in ["std", "log_std"]:
        actor_state_dict[key] = value

      # Migrate critic keys.
      if key.startswith("critic."):
        new_key = key.replace("critic.", "mlp.")
        critic_state_dict[new_key] = value
      elif key.startswith("critic_obs_normalizer."):
        new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
        critic_state_dict[new_key] = value

    loaded_dict["actor_state_dict"] = actor_state_dict
    loaded_dict["critic_state_dict"] = critic_state_dict

  # Migrate rsl-rl 4.x actor keys to 5.x distribution keys.
  actor_sd = loaded_dict.get("actor_state_dict", {})
  if "std" in actor_sd:
    actor_sd["distribution.std_param"] = actor_sd.pop("std")
  if "log_std" in actor_sd:
    actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Strip None-valued optional configs so models don't receive them.
    for key in ("actor", "critic", "student", "teacher"):
      if key in train_cfg:
        train_cfg[key] = clean_model_cfg(train_cfg[key])
    super().__init__(env, train_cfg, log_dir, device)
    init_checkpoint = self.cfg.get("init_checkpoint")
    if init_checkpoint:
      self.load_init_checkpoint(init_checkpoint)

  def load_init_checkpoint(self, path: str) -> None:
    """Initialize the policy weights from a checkpoint, nothing else.

    Unlike ``load``, no optimizer state, iteration count, or env state is
    restored. Accepts PPO checkpoints (``actor_state_dict``) as well as
    distillation checkpoints (``student_state_dict``), so a distilled
    student can be RL fine-tuned, and a previously distilled generalist can
    seed a new round of distillation. Distribution parameters (the action
    std) are dropped so the configured ``init_std`` takes effect.
    """
    print(f"[INFO] Initializing policy weights from checkpoint: {path}")
    loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
    _migrate_checkpoint(loaded_dict, path)
    state_dict = loaded_dict.get("actor_state_dict") or loaded_dict.get(
      "student_state_dict"
    )
    if state_dict is None:
      raise KeyError(f"No actor or student weights found in checkpoint: {path}")
    state_dict = {
      k: v for k, v in state_dict.items() if not k.startswith("distribution.")
    }
    policy = self.alg.get_policy()
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if not k.startswith("distribution.")]
    if bad_missing or unexpected:
      raise RuntimeError(
        f"Checkpoint {path} does not match the policy architecture. "
        f"Missing keys: {bad_missing}. Unexpected keys: {list(unexpected)}."
      )

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    """Resolve ONNX export paths from a checkpoint path."""
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's
    common_step_counter and to respect the ``upload_model`` config flag.
    """
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint.

    Extends the base implementation to:
    1. Restore common_step_counter to preserve curricula state.
    2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
      -> obs_normalizer.*) to the current format (rsl-rl>=4.0).
    """
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    _migrate_checkpoint(loaded_dict, path)
    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos


class MjlabDistillationRunner(MjlabOnPolicyRunner, DistillationRunner):
  """Runner for DAgger-style teacher-student distillation.

  The student policy acts in the environment (sampling from its own action
  distribution, which adds zero-mean exploration noise) while the frozen
  teacher labels every visited state. This is on-policy dataset aggregation
  (DAgger) rather than behavior cloning of teacher rollouts, following
  arXiv:2505.11164. Teacher weights are loaded from the checkpoints listed
  in the ``teacher_checkpoints`` config entry; a multi-teacher config
  distills several experts into a single student.
  """

  alg: Distillation  # pyright: ignore[reportIncompatibleVariableOverride]

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    teacher_checkpoints = tuple(self.cfg.get("teacher_checkpoints") or ())
    if teacher_checkpoints:
      self.load_teacher_checkpoints(teacher_checkpoints)

  def load_teacher_checkpoints(self, paths: tuple[str, ...]) -> None:
    """Load frozen teacher weights from PPO or distillation checkpoints.

    A single-teacher setup takes exactly one path; a multi-teacher setup
    takes one path per expert, in the order the experts are listed in the
    teacher config. When ``inherit_env_state_from_teacher`` is set, the
    env's ``common_step_counter`` is restored from the first checkpoint so
    time-based curricula start in their end-of-training state.
    """
    teacher = self.alg._raw_teacher  # noqa: SLF001
    if isinstance(teacher, MultiTeacherModel):
      models = list(teacher.teachers)
    else:
      models = [teacher]
    if len(paths) != len(models):
      raise ValueError(
        f"Got {len(paths)} teacher checkpoint(s) for {len(models)} teacher "
        "model(s); provide exactly one checkpoint per teacher."
      )

    first_infos: dict | None = None
    for model, path in zip(models, paths, strict=True):
      print(f"[INFO] Loading teacher checkpoint: {path}")
      loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
      _migrate_checkpoint(loaded_dict, path)
      state_dict = (
        loaded_dict.get("actor_state_dict")
        or loaded_dict.get("student_state_dict")
        or loaded_dict.get("teacher_state_dict")
      )
      if state_dict is None:
        raise KeyError(f"No actor, student, or teacher weights found in: {path}")
      model.load_state_dict(state_dict)
      if first_infos is None:
        first_infos = loaded_dict.get("infos") or {}
    self.alg.teacher_loaded = True

    if self.cfg.get("inherit_env_state_from_teacher") and first_infos:
      if "env_state" in first_infos:
        counter = first_infos["env_state"]["common_step_counter"]
        self.env.unwrapped.common_step_counter = counter
        print(f"[INFO] Inherited env common_step_counter from teacher: {counter}")
