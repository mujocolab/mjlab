"""Shared utilities for ONNX policy export across RL tasks."""

import onnx
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction


def list_to_csv_str(
  arr, *, decimals: int = 3, delimiter: str = ",", sub_delimiter: str = ";"
) -> str:
  """Convert list to CSV string with specified decimal precision.

  Elements that are themselves sequences (e.g. a per-dimension scale or a
  [min, max] clip range) are joined with `sub_delimiter` instead of being
  `str()`-formatted, which would otherwise embed a second, ambiguous set of
  commas inside the top-level comma-delimited string.
  """
  fmt = f"{{:.{decimals}f}}"

  def format_scalar(x) -> str:
    return fmt.format(x) if isinstance(x, (int, float)) else str(x)

  def format_entry(x) -> str:
    if isinstance(x, (list, tuple)):
      return sub_delimiter.join(format_scalar(v) for v in x)
    return format_scalar(x)

  return delimiter.join(format_entry(x) for x in arr)


def get_base_metadata(
  env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float]:
  """Get base metadata common to all RL policy exports.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.

  Returns:
    Dictionary of metadata fields that are common across all tasks.
  """
  robot: Entity = env.scene["robot"]
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)
  # Build mapping from joint name to actuator ID for natural joint order.
  # Each spec actuator controls exactly one joint (via its target field).
  joint_name_to_ctrl_id = {}
  for actuator in robot.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    joint_name_to_ctrl_id[joint_name] = actuator.id
  # Get actuator IDs in natural joint order (same order as robot.joint_names).
  ctrl_ids_natural = [
    joint_name_to_ctrl_id[jname]
    for jname in robot.joint_names  # global joint order
    if jname in joint_name_to_ctrl_id  # skip non-actuated joints
  ]
  joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
  joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]
  observation_term_scale: list = []
  observation_term_flatten_history_dim: list = []
  observation_term_history_length: list = []
  observation_term_clip: list = []
  observation_names = env.observation_manager.active_terms["actor"]

  for active_term in observation_names:
    cfg = env.observation_manager.get_term_cfg("actor", active_term)

    if cfg.scale is None:
      observation_term_scale.append(1.0)
    else:
      raw_scale = cfg.scale
      scale = (
        raw_scale.cpu().tolist() if isinstance(raw_scale, torch.Tensor) else raw_scale
      )
      observation_term_scale.append(scale)

    raw_clip = cfg.clip
    if raw_clip is None:
      observation_term_clip.append([float("-inf"), float("inf")])
    else:
      observation_term_clip.append(list(raw_clip))

    observation_term_flatten_history_dim.append(cfg.flatten_history_dim)
    observation_term_history_length.append(cfg.history_length)

  return {
    "run_path": run_path,
    "joint_names": list(robot.joint_names),
    "joint_stiffness": joint_stiffness.tolist(),
    "joint_damping": joint_damping.tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": list(env.command_manager.active_terms),
    "observation_names": observation_names,
    "observation_terms_scale": observation_term_scale,
    "observation_terms_flatten_history_dim": observation_term_flatten_history_dim,
    "observation_terms_history_length": observation_term_history_length,
    "observation_terms_clip": observation_term_clip,
    "action_scale": joint_action._scale[0].cpu().tolist()
    if isinstance(joint_action._scale, torch.Tensor)
    else joint_action._scale,
  }


def attach_metadata_to_onnx(
  onnx_path: str, metadata: dict[str, list | str | float]
) -> None:
  """Attach metadata to an ONNX model file.

  Args:
    onnx_path: Path to the ONNX model file.
    metadata: Dictionary of metadata key-value pairs to attach.
  """
  model = onnx.load(onnx_path)

  for k, v in metadata.items():
    entry = onnx.StringStringEntryProto()
    entry.key = k
    entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
    model.metadata_props.append(entry)

  onnx.save(model, onnx_path)
