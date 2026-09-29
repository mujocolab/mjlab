import wandb
from rsl_rl.utils import WandbLogWriter

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabDistillationRunner, MjlabOnPolicyRunner


class _OnnxExportOnSaveMixin:
  """Exports the current policy to ONNX alongside every checkpoint save."""

  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)  # type: ignore[misc]
    policy_dir, filename, onnx_path = self._get_export_paths(path)  # type: ignore[attr-defined]
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)  # type: ignore[attr-defined]
      is_wandb = isinstance(self.logger.writer, WandbLogWriter)  # type: ignore[attr-defined]
      run_name: str = wandb.run.name if is_wandb and wandb.run else "local"  # type: ignore[assignment]
      # The exported policy consumes the "student" obs set when distilling
      # or fine-tuning a distilled policy, and "actor" otherwise.
      obs_sets: dict = self.cfg.get("obs_groups") or {}  # type: ignore[attr-defined]
      policy_groups = obs_sets.get("student") or obs_sets.get("actor") or ("actor",)
      metadata = get_base_metadata(self.env.unwrapped, run_name, policy_groups[0])
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if is_wandb and self.cfg["upload_model"]:  # type: ignore[attr-defined]
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")


class VelocityOnPolicyRunner(_OnnxExportOnSaveMixin, MjlabOnPolicyRunner):
  pass


class VelocityDistillationRunner(_OnnxExportOnSaveMixin, MjlabDistillationRunner):
  """Distillation runner that exports the student to ONNX on every save."""
