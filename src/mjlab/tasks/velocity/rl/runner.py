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
    actor = self.alg.actor
    if actor.obs_normalization:
      normalizer = actor.obs_normalizer
    else:
      normalizer = None
    export_velocity_policy_as_onnx(
      actor,
      normalizer=normalizer,
      path=policy_path,
      filename=filename,
    )
    # Attach metadata (use "local" for run_path if not using wandb)
    logger_type = self.cfg.get("logger", "tensorboard")
    run_name = wandb.run.name if logger_type == "wandb" and wandb.run else "local"
    attach_onnx_metadata(
      self.env.unwrapped,
      run_name,  # type: ignore
      path=policy_path,
      filename=filename,
    )
    if logger_type == "wandb":
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
