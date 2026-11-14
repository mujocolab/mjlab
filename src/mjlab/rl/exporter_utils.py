"""Shared utilities for ONNX policy export across RL tasks."""

from collections.abc import Sequence

import mujoco
import onnx
import torch

from mjlab.actuator import (
  Actuator,
  BuiltinDcMotorActuator,
  BuiltinPdActuator,
  DcMotorInputMode,
  IdealPdActuator,
)
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.managers.action_manager import ActionTerm


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


def _get_single_actuator_gains(
  act: Actuator, mj_model: mujoco.MjModel
) -> tuple[list[float], list[float]]:
  """Extract per-target (stiffness, damping) for an actuator instance."""
  if isinstance(act, IdealPdActuator):
    assert act.default_stiffness is not None
    assert act.default_damping is not None
    return (
      act.default_stiffness[0].cpu().tolist(),
      act.default_damping[0].cpu().tolist(),
    )
  if isinstance(act, BuiltinPdActuator):
    n = act.num_targets
    ctrl_ids = act.global_ctrl_ids.cpu().tolist()
    pos_ids, vel_ids = ctrl_ids[:n], ctrl_ids[n:]
    return (
      mj_model.actuator_gainprm[pos_ids, 0].tolist(),
      mj_model.actuator_gainprm[vel_ids, 0].tolist(),
    )
  if isinstance(act, BuiltinDcMotorActuator):
    ctrl_ids = act.global_ctrl_ids.cpu().tolist()
    if act.cfg.mode == DcMotorInputMode.VOLTAGE:
      zeros = [0.0] * len(ctrl_ids)
      return zeros, zeros.copy()
    return (
      mj_model.actuator_gainprm[ctrl_ids, 4].tolist(),
      mj_model.actuator_gainprm[ctrl_ids, 6].tolist(),
    )
  ctrl_ids = act.global_ctrl_ids.cpu().tolist()
  return (
    mj_model.actuator_gainprm[ctrl_ids, 0].tolist(),
    (-mj_model.actuator_biasprm[ctrl_ids, 2]).tolist(),
  )


def _extract_transmission_gains(
  robot: Entity,
  mj_model: mujoco.MjModel,
  transmission_type: TransmissionType,
  natural_names: Sequence[str],
) -> tuple[list[float], list[float]]:
  """Extract stiffness and damping in natural component order."""
  target_to_gains: dict[str, tuple[float, float]] = {}
  if robot.actuators:
    for act in robot.actuators:
      if act.transmission_type != transmission_type:
        continue
      kp_list, kd_list = _get_single_actuator_gains(act, mj_model)
      for tname, kp, kd in zip(act.target_names, kp_list, kd_list, strict=True):
        target_to_gains[tname] = (float(kp), float(kd))
  elif robot.spec.actuators:
    expected_trn = (
      mujoco.mjtTrn.mjTRN_TENDON
      if transmission_type == TransmissionType.TENDON
      else mujoco.mjtTrn.mjTRN_JOINT
    )
    for spec_act in robot.spec.actuators:
      if spec_act.trntype != expected_trn:
        continue
      tname = spec_act.target.split("/")[-1]
      kp = float(mj_model.actuator_gainprm[spec_act.id, 0])
      kd = float(-mj_model.actuator_biasprm[spec_act.id, 2])
      target_to_gains[tname] = (kp, kd)

  stiffness = [target_to_gains[n][0] for n in natural_names if n in target_to_gains]
  damping = [target_to_gains[n][1] for n in natural_names if n in target_to_gains]
  return stiffness, damping


def _term_scale_values(term: ActionTerm) -> float | list[float]:
  """Extract scalar or per-target action scale for a single action term."""
  raw = getattr(term, "scale", getattr(term.cfg, "scale", 1.0))
  if isinstance(raw, torch.Tensor):
    vals = raw[0].cpu().tolist() if raw.ndim > 1 else raw.cpu().tolist()
    return [float(v) for v in vals] if isinstance(vals, list) else float(vals)
  if isinstance(raw, (int, float)):
    return float(raw)
  return [float(v) for v in raw]


def _combine_term_scales(terms: Sequence[ActionTerm]) -> float | list[float]:
  """Combine scales across one or more action terms."""
  if not terms:
    return []
  if len(terms) == 1:
    return _term_scale_values(terms[0])
  expanded: list[float] = []
  for term in terms:
    scale = _term_scale_values(term)
    if isinstance(scale, list):
      expanded.extend(scale)
    else:
      expanded.extend([scale] * term.action_dim)
  return expanded


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
  mj_model = env.sim.mj_model

  joint_stiffness, joint_damping = _extract_transmission_gains(
    robot, mj_model, TransmissionType.JOINT, robot.joint_names
  )

  active_action_terms = [
    env.action_manager.get_term(name) for name in env.action_manager.active_terms
  ]
  action_names: list[str] = []
  action_transmission_types: list[str] = []
  tendon_action_terms: list[BaseAction] = []
  for term_name, term in zip(
    env.action_manager.active_terms, active_action_terms, strict=True
  ):
    if isinstance(term, BaseAction):
      action_names.extend(term.target_names)
      action_transmission_types.extend(
        [term.cfg.transmission_type.value] * term.action_dim
      )
      if term.cfg.transmission_type == TransmissionType.TENDON:
        tendon_action_terms.append(term)
    else:
      action_names.extend([f"{term_name}_{i}" for i in range(term.action_dim)])
      action_transmission_types.extend(["custom"] * term.action_dim)

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

  default_joint_pos = (
    robot.data.default_joint_pos[0].cpu().tolist()
    if robot.data.default_joint_pos is not None
    else []
  )

  metadata: dict[str, list | str | float] = {
    "run_path": run_path,
    "joint_names": list(robot.joint_names),
    "joint_stiffness": joint_stiffness,
    "joint_damping": joint_damping,
    "default_joint_pos": default_joint_pos,
    "action_names": action_names,
    "action_transmission_types": action_transmission_types,
    "command_names": list(env.command_manager.active_terms),
    "observation_names": observation_names,
    "observation_terms_scale": observation_term_scale,
    "observation_terms_flatten_history_dim": observation_term_flatten_history_dim,
    "observation_terms_history_length": observation_term_history_length,
    "observation_terms_clip": observation_term_clip,
    "action_scale": _combine_term_scales(active_action_terms),
  }

  if len(robot.tendon_names) > 0:
    tendon_stiffness, tendon_damping = _extract_transmission_gains(
      robot, mj_model, TransmissionType.TENDON, robot.tendon_names
    )
    tendon_ids_np = robot.indexing.tendon_ids.cpu().numpy()
    metadata["tendon_names"] = list(robot.tendon_names)
    metadata["tendon_stiffness"] = tendon_stiffness
    metadata["tendon_damping"] = tendon_damping
    metadata["default_tendon_len"] = mj_model.tendon_length0[tendon_ids_np].tolist()
    if tendon_action_terms:
      metadata["tendon_action_scale"] = _combine_term_scales(tendon_action_terms)

  return metadata


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
