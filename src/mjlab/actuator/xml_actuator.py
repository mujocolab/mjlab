"""Use actuators already defined in your robot XML.

When your MJCF already has ``<position>``, ``<velocity>``, ``<motor>``, or ``<muscle>``
elements, ``XmlActuatorCfg`` wraps them without re-specifying gains or limits. The
actuator type is auto-detected from the spec element. Delay and other ``ActuatorCfg``
fields work as usual.

If your XML has actuators on both joints and tendons, use two configs with different
``transmission_type`` values; a single config can only target one namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import mujoco
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd, CommandField
from mjlab.utils.mujoco import detect_command_field

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class XmlActuatorCfg(ActuatorCfg):
  """Wrap existing XML-defined actuators.

  Detects the actuator type (position, velocity, motor, muscle) from the MuJoCo spec
  element during ``edit_spec``, or validates against an explicit ``command_field`` if
  provided.
  """

  command_field: CommandField | None = None
  """If provided, require XML actuators to match this command field.
  If None, auto-detect strictly from the spec element."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> XmlActuator:
    return XmlActuator(self, entity, target_ids, target_names)


class XmlActuator(Actuator[XmlActuatorCfg]):
  """Wrapper for XML-defined actuators."""

  def __init__(
    self,
    cfg: XmlActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self._command_field: CommandField | None = None

  @property
  def command_field(self) -> CommandField:
    """The resolved command field (detected or explicit), not the raw config value.

    Set during ``edit_spec``. Use ``self.cfg.command_field`` to access the user's
    original input (which may be None for auto-detection).
    """
    assert self._command_field is not None
    return self._command_field

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    # Filter to only targets that have corresponding XML actuators.
    filtered_target_ids = []
    filtered_target_names = []
    for i, target_name in enumerate(target_names):
      actuator = self._find_actuator_for_target(spec, target_name)
      if actuator is not None:
        self._mjs_actuators.append(actuator)
        filtered_target_ids.append(self._target_ids_list[i])
        # Store the user-facing (stripped) name, not the spec name.
        filtered_target_names.append(self._target_names[i])

    if len(filtered_target_names) == 0:
      raise ValueError(
        f"No XML actuators found for any targets matching the patterns. "
        f"Searched targets: {target_names}. "
        f"XML actuator config expects actuators to already exist in the XML."
      )

    # Update target IDs and names to only include those with actuators.
    self._target_ids_list = filtered_target_ids
    self._target_names = filtered_target_names

    if self.cfg.command_field is not None:
      # Validate that every matched actuator is compatible with the
      # requested command field.
      for act, name in zip(self._mjs_actuators, self._target_names, strict=True):
        try:
          detected = detect_command_field(act)
        except ValueError:
          continue  # Unrecognized type; trust the explicit override.
        if detected != self.cfg.command_field:
          raise ValueError(
            f"XML actuator for '{name}' is type '{detected}', but "
            f"command_field='{self.cfg.command_field}' was requested."
          )
      self._command_field = self.cfg.command_field
    else:
      # Auto-detect from spec elements. Raises for unrecognized types.
      fields = [detect_command_field(a) for a in self._mjs_actuators]
      unique = set(fields)
      if len(unique) > 1:
        breakdown = dict(zip(self._target_names, fields, strict=True))
        raise ValueError(
          f"Mixed XML actuator types for targets: {breakdown}. "
          f"Split into separate XmlActuatorCfg groups."
        )
      self._command_field = cast(CommandField, fields[0])

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    if self._command_field == "position":
      return cmd.position_target
    if self._command_field == "velocity":
      return cmd.velocity_target
    return cmd.effort_target

  def _find_actuator_for_target(
    self, spec: mujoco.MjSpec, target_name: str
  ) -> mujoco.MjsActuator | None:
    """Find an actuator that targets the given target."""
    for actuator in spec.actuators:
      if actuator.target == target_name:
        return actuator
    return None
