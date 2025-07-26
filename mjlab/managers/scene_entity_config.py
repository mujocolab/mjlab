from dataclasses import dataclass, MISSING, field

from mjlab.entities import Robot
from mjlab.scene import Scene


@dataclass
class SceneEntityCfg:
  """Configuration for a scene entity that is used by the manager's term."""

  name: str = MISSING

  joint_names: str | list[str] | None = None
  joint_ids: list[int] | slice = field(default_factory=lambda: slice(None))

  body_names: str | list[str] | None = None
  body_ids: list[int] | slice = field(default_factory=lambda: slice(None))

  preserve_order: bool = False

  def resolve(self, scene: Scene) -> None:
    self._resolve_joint_names(scene)
    self._resolve_body_names(scene)

  def _resolve_joint_names(self, scene: Scene) -> None:
    if self.joint_names is not None or self.joint_ids != slice(None):
      entity: Robot = scene[self.name]
      if self.joint_names is not None and self.joint_ids != slice(None):
        if isinstance(self.joint_names, str):
          self.joint_names = [self.joint_names]
        if isinstance(self.joint_ids, int):
          self.joint_ids = [self.joint_ids]
        joint_ids, _ = entity.find_joints(
          self.joint_names, preserve_order=self.preserve_order
        )
        joint_names = [entity.joint_names[i] for i in self.joint_ids]
        if joint_ids != self.joint_ids or joint_names != self.joint_names:
          raise ValueError()
      elif self.joint_names is not None:
        if isinstance(self.joint_names, str):
          self.joint_names = [self.joint_names]
        self.joint_ids, _ = entity.find_joints(
          self.joint_names, preserve_order=self.preserve_order
        )
        if (
          len(self.joint_ids) == entity.num_joints
          and self.joint_names == entity.joint_names
        ):
          self.joint_ids = slice(None)
      elif self.joint_ids != slice(None):
        if isinstance(self.joint_ids, int):
          self.joint_ids = [self.joint_ids]
        self.joint_names = [entity.joint_names[i] for i in self.joint_ids]

  def _resolve_body_names(self, scene: Scene) -> None:
    if self.body_names is not None or self.body_ids != slice(None):
      entity: Robot = scene[self.name]
      if self.body_names is not None and self.body_ids != slice(None):
        if isinstance(self.body_names, str):
          self.body_names = [self.body_names]
        if isinstance(self.body_ids, int):
          self.body_ids = [self.body_ids]
        body_ids, _ = entity.find_bodies(
          self.body_names, preserve_order=self.preserve_order
        )
        body_names = [entity.body_names[i] for i in self.body_ids]
        if body_ids != self.body_ids or body_names != self.body_names:
          raise ValueError()
      elif self.body_names is not None:
        if isinstance(self.body_names, str):
          self.body_names = [self.body_names]
        self.body_ids, _ = entity.find_bodies(
          self.body_names, preserve_order=self.preserve_order
        )
        if (
          len(self.body_ids) == entity.num_bodies
          and self.body_names == entity.body_names
        ):
          self.body_ids = slice(None)
      elif self.body_ids != slice(None):
        if isinstance(self.body_ids, int):
          self.body_ids = [self.body_ids]
        self.body_names = [entity.body_names[i] for i in self.body_ids]
