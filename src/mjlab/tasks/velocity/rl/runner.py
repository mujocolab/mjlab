import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    if self.alg.policy.actor_obs_normalization:
      normalizer = self.alg.policy.actor_obs_normalizer
    else:
      normalizer = None
    export_velocity_policy_as_onnx(
      self.alg.policy,
      normalizer=normalizer,
      path=policy_path,
      filename=filename,
    )
    # Attach metadata (use "local" for run_path if not using wandb)
    run_name = wandb.run.name if self.logger_type == "wandb" and wandb.run else "local"
    attach_onnx_metadata(
      self.env.unwrapped,
      run_name,  # type: ignore
      path=policy_path,
      filename=filename,
    )
    if self.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
