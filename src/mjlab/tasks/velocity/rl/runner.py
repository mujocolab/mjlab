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

    # Note: In new rsl_rl, the Actor (MLPModel) usually contains the normalizer internally.
    # If we pass the normalizer to the exporter, it will wrap the actor and run:
    # normalizer(x) -> actor(normalizer(x)) -> actor_internal_normalizer(normalizer(x))
    # This causes double normalization.

    # We pass None so the exporter uses Identity, relying on the actor's internal normalizer.
    export_velocity_policy_as_onnx(
      self.alg.actor,
      normalizer=None,
      path=policy_path,
      filename=filename,
    )
    # Attach metadata (use "local" for run_path if not using wandb)
    run_name = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )
    attach_onnx_metadata(
      self.env.unwrapped,
      run_name,  # type: ignore
      path=policy_path,
      filename=filename,
    )
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
