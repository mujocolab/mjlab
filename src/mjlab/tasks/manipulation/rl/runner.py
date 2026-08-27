import wandb
from rsl_rl.utils import WandbLogWriter

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class ManipulationOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      is_wandb = isinstance(self.logger.writer, WandbLogWriter)
      run_name: str = wandb.run.name if is_wandb and wandb.run else "local"  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if is_wandb and self.cfg["upload_model"]:
        wandb.save(
          str(onnx_path),
          base_path=str(policy_dir),
        )
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
