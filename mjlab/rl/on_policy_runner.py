from rsl_rl.runners import OnPolicyRunner

from pathlib import Path
import wandb
from mjlab.rl.exporter import export_policy_as_onnx, attach_onnx_metadata


class MjlabOnPolicyRunner(OnPolicyRunner):
  def save(self, path: str, infos=None):
    """Override the save method to also save the policy as an ONNX file."""
    super().save(path, infos)
    if self.logger_type in ["wandb"]:
      policy_path = Path(path).parent
      export_policy_as_onnx(
        self.alg.policy, normalizer=self.obs_normalizer, path=policy_path
      )
      attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path)
      wandb.save(policy_path / "policy.onnx", base_path=policy_path)
