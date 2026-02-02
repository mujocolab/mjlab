from typing import cast

from etils.epath import Path
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.rl.exporter import _OnnxPolicyExporter
import torch


def export_motion_policy_as_onnx(
  env: ManagerBasedRlEnv,
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  path = Path(path)
  path.mkdir(parents=True, exist_ok=True)
  policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
  policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
  def __init__(
    self, env: ManagerBasedRlEnv, actor_critic, normalizer=None, verbose=False
  ):
    super().__init__(actor_critic, normalizer, verbose)
    cmd = cast(MotionCommand, env.command_manager.get_term("motion"))

    self.joint_pos = cmd.motion.joint_pos.to("cpu")
    self.joint_vel = cmd.motion.joint_vel.to("cpu")
    self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
    self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
    self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
    self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
    self.time_step_total: int = self.joint_pos.shape[0]

  def forward(self, x, time_step):  # type: ignore[invalid-method-override]
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.actor(self.normalizer(x)),
      self.joint_pos[time_step_clamped],
      self.joint_vel[time_step_clamped],
      self.body_pos_w[time_step_clamped],
      self.body_quat_w[time_step_clamped],
      self.body_lin_vel_w[time_step_clamped],
      self.body_ang_vel_w[time_step_clamped],
    )

  def export(self, path, filename):
    self.to("cpu")
    obs = torch.zeros(1, self.actor[0].in_features)
    time_step = torch.zeros(1, 1)
    with (Path(path) / filename).open("wb") as f:
      torch.onnx.export(
          self,
          (obs, time_step),
          f,
          export_params=True,
          opset_version=11,
          verbose=self.verbose,
          input_names=["obs", "time_step"],
          output_names=[
              "actions",
              "joint_pos",
              "joint_vel",
              "body_pos_w",
              "body_quat_w",
              "body_lin_vel_w",
              "body_ang_vel_w",
          ],
          dynamic_axes={},
          dynamo=False,
      )


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  """Attach tracking-specific metadata to ONNX model.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.
    path: Directory containing the ONNX file.
    filename: Name of the ONNX file.
  """
  onnx_path = Path(path) / filename

  # Get base metadata common to all tasks.
  metadata = get_base_metadata(env, run_path)

  # Add tracking-specific metadata.
  motion_term = env.command_manager.get_term("motion")
  assert isinstance(motion_term, MotionCommand)
  motion_term_cfg = motion_term.cfg
  metadata.update(
    {
      "anchor_body_name": motion_term_cfg.anchor_body_name,
      "body_names": list(motion_term_cfg.body_names),
    }
  )

  attach_metadata_to_onnx(onnx_path, metadata)
