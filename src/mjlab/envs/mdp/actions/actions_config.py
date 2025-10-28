from dataclasses import dataclass

from mjlab.envs.mdp.actions import joint_actions
from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import ActionTermCfg


@dataclass(kw_only=True)
class JointActionCfg(ActionTermCfg):
  actuator_names: list[str]
  """List of actuator names or regex expressions that the action will be mapped to."""
  scale: float | dict[str, float] = 1.0
  """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
  offset: float | dict[str, float] = 0.0
  """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
  preserve_order: bool = False
  """Whether to preserve the order of the joint names in the action output. Defaults to False."""


@dataclass(kw_only=True)
class JointPositionActionCfg(JointActionCfg):
  class_type: type[ActionTerm] = joint_actions.JointPositionAction
  use_default_offset: bool = True


@dataclass(kw_only=True)
class BinaryJointActionCfg(ActionTermCfg):
  actuator_names: list[str]
  """List of actuator names or regex expressions that the action will be mapped to."""
  open_command_expr: dict[str, float]
  """Dictionary of open command expressions for the binary joint action."""
  close_command_expr: dict[str, float]
  """Dictionary of close command expressions for the binary joint action."""


@dataclass(kw_only=True)
class BinaryJointPositionActionCfg(BinaryJointActionCfg):
  class_type: type[ActionTerm] = joint_actions.BinaryJointPositionAction
  use_default_offset: bool = True