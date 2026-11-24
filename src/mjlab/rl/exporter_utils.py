"""Shared utilities for ONNX policy export across RL tasks."""

import warnings

import onnx
import torch

from mjlab.actuator import XmlActuator
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import BaseAction, TendonLengthAction


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
  # Gains come from the actuator config. Actuators without a stiffness or damping
  # field (e.g. motors) report NaN. XML actuators have no gains in their config,
  # so theirs are read from the MuJoCo gain and bias parameters instead. Names may
  # collide across transmission types, so the key includes the type.
  gains: dict[tuple[TransmissionType, str], tuple[float, float]] = {}
  mj_model = env.sim.mj_model
  for actuator in robot.actuators:
    if isinstance(actuator, XmlActuator):
      ctrl_ids = actuator.global_ctrl_ids.cpu().tolist()
      target_gains = zip(
        mj_model.actuator_gainprm[ctrl_ids, 0].tolist(),
        (-mj_model.actuator_biasprm[ctrl_ids, 2]).tolist(),
        strict=True,
      )
    else:
      stiffness = getattr(actuator.cfg, "stiffness", float("nan"))
      damping = getattr(actuator.cfg, "damping", float("nan"))
      target_gains = [(stiffness, damping)] * len(actuator.target_names)
    for name, kp_kd in zip(actuator.target_names, target_gains, strict=True):
      gains[(actuator.transmission_type, name)] = kp_kd

  action_names: list[str] = []
  action_target_names: list[str] = []
  action_scale: list[float] = []
  action_offset: list[float] = []
  driven_targets: dict[TransmissionType, set[str]] = {}
  default_tendon_pos: dict[str, float] = {}
  for term_name in env.action_manager.active_terms:
    action = env.action_manager.get_term(term_name)
    action_names.append(term_name)
    if not isinstance(action, BaseAction):
      warnings.warn(
        f"Action term '{term_name}' ({type(action).__name__}) is not a "
        "BaseAction. Its scale and offset are exported as NaN.",
        stacklevel=2,
      )
      action_target_names.extend(f"{term_name}_{i}" for i in range(action.action_dim))
      action_scale.extend([float("nan")] * action.action_dim)
      action_offset.extend([float("nan")] * action.action_dim)
      continue
    scale, offset = (
      v[0].cpu().tolist() if isinstance(v, torch.Tensor) else [v] * action.action_dim
      for v in (action.scale, action.offset)
    )
    action_target_names.extend(action.target_names)
    action_scale.extend(scale)
    action_offset.extend(offset)
    driven_targets.setdefault(action.cfg.transmission_type, set()).update(
      action.target_names
    )
    # Assume a tendon length action's offset is the tendon's default length.
    if isinstance(action, TendonLengthAction):
      default_tendon_pos.update(zip(action.target_names, offset, strict=True))

  transmission_metadata: dict[str, list] = {}
  for transmission, names in (
    (TransmissionType.JOINT, robot.joint_names),
    (TransmissionType.TENDON, robot.tendon_names),
    (TransmissionType.SITE, robot.site_names),
  ):
    if transmission not in driven_targets:
      continue
    # All joints are exported. Tendons and sites only when an action drives them.
    if transmission != TransmissionType.JOINT:
      names = [n for n in names if n in driven_targets[transmission]]
    actuated = [gains[(transmission, n)] for n in names if (transmission, n) in gains]
    prefix = transmission.value
    transmission_metadata[f"{prefix}_names"] = list(names)
    transmission_metadata[f"{prefix}_stiffness"] = [kp for kp, _ in actuated]
    transmission_metadata[f"{prefix}_damping"] = [kd for _, kd in actuated]
    if transmission == TransmissionType.JOINT:
      transmission_metadata["default_joint_pos"] = (
        robot.data.default_joint_pos[0].cpu().tolist()
      )
    elif transmission == TransmissionType.TENDON:
      transmission_metadata["default_tendon_pos"] = [
        default_tendon_pos.get(n, float("nan")) for n in names
      ]

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
    **transmission_metadata,
    "command_names": list(env.command_manager.active_terms),
    "observation_names": observation_names,
    "observation_terms_scale": observation_term_scale,
    "observation_terms_flatten_history_dim": observation_term_flatten_history_dim,
    "observation_terms_history_length": observation_term_history_length,
    "observation_terms_clip": observation_term_clip,
    "action_names": action_names,
    "action_target_names": action_target_names,
    "action_scale": action_scale,
    "action_offset": action_offset,
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
