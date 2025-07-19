from dataclasses import dataclass, field
import mujoco

slice_field = field(default_factory=lambda: slice(None))


@dataclass
class SceneEntityCfg:
  name: str
  joint_names: str | list[str] | None = None
  body_names: str | list[str] | None = None
  preserve_order: bool = False
  joint_ids: list[str] | slice = slice_field
  joint_q_adr: list[int] | slice = slice_field
  joint_v_adr: list[int] | slice = slice_field
  body_ids: list[int] | slice = slice_field
  actuator_ids: list[int] | slice = slice_field

  def resolve(self, model: mujoco.MjModel) -> None:
    self._resolve_joint_names(model)
    self._resolve_body_names(model)

  def _resolve_joint_names(self, model: mujoco.MjModel) -> None:
    pass

  def _resolve_body_names(self, model: mujoco.MjModel) -> None:
    pass