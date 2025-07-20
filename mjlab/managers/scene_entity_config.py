from dataclasses import dataclass, field, MISSING
import mujoco


@dataclass
class SceneEntityCfg:
  """Configuration for a scene entity that is used by the manager's term."""

  name: str = MISSING
  joint_names: str | list[str] | None = None
  body_names: str | list[str] | None = None
  joint_ids: list[str] | slice = field(default_factory=lambda: slice(None))
  body_ids: list[int] | slice = field(default_factory=lambda: slice(None))
  preserve_order: bool = False

  def resolve(self, model: mujoco.MjModel) -> None:
    self._resolve_joint_names(model)
    self._resolve_body_names(model)

  def _resolve_joint_names(self, model: mujoco.MjModel) -> None:
    pass

  def _resolve_body_names(self, model: mujoco.MjModel) -> None:
    pass
