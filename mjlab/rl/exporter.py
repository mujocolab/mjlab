import os
from pathlib import Path
from typing import Optional
import onnx
import torch
import copy
from mjlab.core import MjxEnv, MjxTask


def export_policy_as_onnx(
  policy: object,
  normalizer: Optional[object],
  path: Path,
  filename: str = "policy.onnx",
  verbose: bool = False,
):
  """Export policy to an ONNX file."""
  policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
  if not path.exists():
    path.mkdir(parents=True, exist_ok=True)
  policy_exporter.export(path, filename)


class _OnnxPolicyExporter(torch.nn.Module):
  def __init__(
    self, policy: object, normalizer: Optional[object] = None, verbose: bool = False
  ):
    super().__init__()
    self.verbose = verbose
    self.is_recurrent = policy.is_recurrent
    # copy policy parameters
    if hasattr(policy, "actor"):
      self.actor = copy.deepcopy(policy.actor)
      if self.is_recurrent:
        self.rnn = copy.deepcopy(policy.memory_a.rnn)
    elif hasattr(policy, "student"):
      self.actor = copy.deepcopy(policy.student)
      if self.is_recurrent:
        self.rnn = copy.deepcopy(policy.memory_s.rnn)
    else:
      raise ValueError("Policy does not have an actor/student module.")
    # set up recurrent network
    if self.is_recurrent:
      self.rnn.cpu()
      self.forward = self.forward_lstm
    # copy normalizer if exists
    if normalizer:
      self.normalizer = copy.deepcopy(normalizer)
    else:
      self.normalizer = torch.nn.Identity()

  def forward_lstm(self, x_in, h_in, c_in):
    x_in = self.normalizer(x_in)
    x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
    x = x.squeeze(0)
    return self.actor(x), h, c

  def forward(self, x):
    return self.actor(self.normalizer(x))

  def export(self, path: Path, filename: str) -> None:
    self.to("cpu")
    save_path = path / filename
    if self.is_recurrent:
      obs = torch.zeros(1, self.rnn.input_size)
      h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
      c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
      actions, h_out, c_out = self(obs, h_in, c_in)
      torch.onnx.export(
        self,
        (obs, h_in, c_in),
        save_path,
        export_params=True,
        opset_version=11,
        verbose=self.verbose,
        input_names=["obs", "h_in", "c_in"],
        output_names=["actions", "h_out", "c_out"],
        dynamic_axes={},
      )
    else:
      obs = torch.zeros(1, self.actor[0].in_features)
      torch.onnx.export(
        self,
        obs,
        save_path,
        export_params=True,
        opset_version=11,
        verbose=self.verbose,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
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
  env: MjxEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  onnx_path = os.path.join(path, filename)

  task: MjxTask = env.task
  robot = task.robot

  command_names = []
  if hasattr(task, "command_names"):
    command_names = task.command_names

  metadata = {
    "run_path": run_path,
    "joint_names": robot.joint_names,
    "joint_stiffness": robot.joint_stiffness,
    "joint_damping": robot.joint_damping,
    "default_joint_pos": robot.default_joint_pos_nominal,
    "command_names": command_names,
    "observation_names": task.observation_names,
    "action_scale": task.cfg.action_scale,
  }

  model = onnx.load(onnx_path)

  for k, v in metadata.items():
    entry = onnx.StringStringEntryProto()
    entry.key = k
    entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
    model.metadata_props.append(entry)

  onnx.save(model, onnx_path)
