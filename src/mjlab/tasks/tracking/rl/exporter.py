import os
from typing import cast

import onnx
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.joint_actions import JointAction
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.third_party.isaaclab.isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter


def export_motion_policy_as_onnx(
  env: ManagerBasedRlEnv,
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
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

  def forward(self, x, time_step):  # pyright: ignore [reportIncompatibleMethodOverride]
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
    torch.onnx.export(
      self,
      (obs, time_step),
      os.path.join(path, filename),
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


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
  fmt = f"{{:.{decimals}f}}"
  return delimiter.join(
    fmt.format(x)
    if isinstance(x, (int, float))
    else str(x)  # numbers → format, strings → as-is
    for x in arr
  )


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  robot: Entity = env.scene["robot"]
  onnx_path = os.path.join(path, filename)
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointAction)
  ctrl_ids = robot.indexing.ctrl_ids.cpu().numpy()
  joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
  joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]
  motion_term = env.command_manager.get_term("motion")
  assert isinstance(motion_term, MotionCommand)
  motion_term_cfg = motion_term.cfg
  metadata = {
    "run_path": run_path,
    "joint_names": robot.joint_names,
    "joint_stiffness": joint_stiffness.tolist(),
    "joint_damping": joint_damping.tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": env.command_manager.active_terms,
    "observation_names": env.observation_manager.active_terms["policy"],
    "action_scale": joint_action._scale[0].cpu().tolist()
    if isinstance(joint_action._scale, torch.Tensor)
    else joint_action._scale,
    "anchor_body_name": motion_term_cfg.anchor_body_name,
    "body_names": motion_term_cfg.body_names,
  }

  model = onnx.load(onnx_path)

  for k, v in metadata.items():
    entry = onnx.StringStringEntryProto()
    entry.key = k
    entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
    model.metadata_props.append(entry)

  onnx.save(model, onnx_path)
