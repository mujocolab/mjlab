from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      metadata = get_base_metadata(self.env.unwrapped, self._get_run_name())
      attach_metadata_to_onnx(str(onnx_path), metadata)
      self._upload_model(str(onnx_path))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
