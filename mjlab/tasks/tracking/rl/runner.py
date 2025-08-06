from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env.vec_env import VecEnv
import wandb
import os

from mjlab.tasks.tracking.rl.exporter import (
  export_motion_policy_as_onnx,
  attach_onnx_metadata,
)


class MotionTrackingOnPolicyRunner(OnPolicyRunner):
  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device="cpu",
    registry_name: str = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    if self.logger_type in ["wandb"]:
      policy_path = path.split("model")[0]
      filename = policy_path.split("/")[-2] + ".onnx"
      if self.alg.policy.actor_obs_normalization:
        normalizer = self.alg.policy.actor_obs_normalizer
      else:
        normalizer = None
      export_motion_policy_as_onnx(
        self.env.unwrapped,
        self.alg.policy,
        normalizer=normalizer,
        path=policy_path,
        filename=filename,
      )
      attach_onnx_metadata(
        self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename
      )
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

      # link the artifact registry to this run
      if self.registry_name is not None:
        wandb.run.use_artifact(self.registry_name)
        self.registry_name = None
