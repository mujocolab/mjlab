from dataclasses import dataclass, field

from mjlab.entity import Entity
from mjlab.scene import Scene


# TODO: The four _resolve_*_names methods are nearly identical. Refactor.
@dataclass
class SceneEntityCfg:
  """Configuration for a scene entity that is used by the manager's term."""

  name: str

  joint_names: str | list[str] | None = None
  joint_ids: list[int] | slice = field(default_factory=lambda: slice(None))

  body_names: str | list[str] | None = None
  body_ids: list[int] | slice = field(default_factory=lambda: slice(None))

  geom_names: str | list[str] | None = None
  geom_ids: list[int] | slice = field(default_factory=lambda: slice(None))

  site_names: str | list[str] | None = None
  site_ids: list[int] | slice = field(default_factory=lambda: slice(None))

  sensor_names: str | list[str] | None = None
  sensor_ids: list[int] | slice = field(default_factory=lambda: slice(None))

  preserve_order: bool = False

  def resolve(self, scene: Scene) -> None:
    self._resolve_joint_names(scene)
    self._resolve_body_names(scene)
    self._resolve_geom_names(scene)
    self._resolve_site_names(scene)
    self._resolve_sensor_names(scene)

  def _resolve_joint_names(self, scene: Scene) -> None:
    if self.joint_names is not None or isinstance(self.joint_ids, list):
      entity: Entity = scene[self.name]

      # Joint name regex --> joint indices.
      if self.joint_names is not None and isinstance(self.joint_ids, list):
        if isinstance(self.joint_names, str):
          self.joint_names = [self.joint_names]
        if isinstance(self.joint_ids, int):
          self.joint_ids = [self.joint_ids]

        joint_ids, _ = entity.find_joints(
          self.joint_names, preserve_order=self.preserve_order
        )
        joint_names = [entity.joint_names[i] for i in self.joint_ids]
        if joint_ids != self.joint_ids or joint_names != self.joint_names:
          raise ValueError("Inconsistent joint names and indices.")

      # Joint indices --> joint names.
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

      # Joint indices --> joint names.
      elif isinstance(self.joint_ids, list):
        if isinstance(self.joint_ids, int):
          self.joint_ids = [self.joint_ids]
        self.joint_names = [entity.joint_names[i] for i in self.joint_ids]

  def _resolve_body_names(self, scene: Scene) -> None:
    if self.body_names is not None or isinstance(self.body_ids, list):
      entity: Entity = scene[self.name]

      # Body name regex --> body indices.
      if self.body_names is not None and isinstance(self.body_ids, list):
        if isinstance(self.body_names, str):
          self.body_names = [self.body_names]
        if isinstance(self.body_ids, int):
          self.body_ids = [self.body_ids]
        body_ids, _ = entity.find_bodies(
          self.body_names, preserve_order=self.preserve_order
        )
        body_names = [entity.body_names[i] for i in self.body_ids]
        if body_ids != self.body_ids or body_names != self.body_names:
          raise ValueError("Inconsistent body names and indices.")

      # Body indices --> body names.
      elif self.body_names is not None:
        if isinstance(self.body_names, str):
          self.body_names = [self.body_names]
        self.body_ids, self.body_names = entity.find_bodies(
          self.body_names, preserve_order=self.preserve_order
        )
        if (
          len(self.body_ids) == entity.num_bodies
          and self.body_names == entity.body_names
        ):
          self.body_ids = slice(None)

      # Body indices --> body names.
      elif isinstance(self.body_ids, list):
        if isinstance(self.body_ids, int):
          self.body_ids = [self.body_ids]
        self.body_names = [entity.body_names[i] for i in self.body_ids]

  def _resolve_geom_names(self, scene: Scene) -> None:
    if self.geom_names is not None or isinstance(self.geom_ids, list):
      entity: Entity = scene[self.name]

      # Geom name regex --> geom indices.
      if self.geom_names is not None and isinstance(self.geom_ids, list):
        if isinstance(self.geom_names, str):
          self.geom_names = [self.geom_names]
        if isinstance(self.geom_ids, int):
          self.geom_ids = [self.geom_ids]
        geom_ids, _ = entity.find_geoms(
          self.geom_names, preserve_order=self.preserve_order
        )
        geom_names = [entity.geom_names[i] for i in self.geom_ids]
        if geom_ids != self.geom_ids or geom_names != self.geom_names:
          raise ValueError("Inconsistent geom names and indices.")

      # Geom indices --> geom names.
      elif self.geom_names is not None:
        if isinstance(self.geom_names, str):
          self.geom_names = [self.geom_names]
        self.geom_ids, _ = entity.find_geoms(
          self.geom_names, preserve_order=self.preserve_order
        )
        if (
          len(self.geom_ids) == entity.num_geoms
          and self.geom_names == entity.geom_names
        ):
          self.geom_ids = slice(None)

      # Geom indices --> geom names.
      elif isinstance(self.geom_ids, list):
        if isinstance(self.geom_ids, int):
          self.geom_ids = [self.geom_ids]
        self.geom_names = [entity.geom_names[i] for i in self.geom_ids]

  def _resolve_site_names(self, scene: Scene) -> None:
    if self.site_names is not None or isinstance(self.site_ids, list):
      entity: Entity = scene[self.name]

      # Site name regex --> site indices.
      if self.site_names is not None and isinstance(self.site_ids, list):
        if isinstance(self.site_names, str):
          self.site_names = [self.site_names]
        if isinstance(self.site_ids, int):
          self.site_ids = [self.site_ids]
        site_ids, _ = entity.find_sites(
          self.site_names, preserve_order=self.preserve_order
        )
        site_names = [entity.site_names[i] for i in self.site_ids]
        if site_ids != self.site_ids or site_names != self.site_names:
          raise ValueError("Inconsistent site names and indices.")

      # Site indices --> site names.
      elif self.site_names is not None:
        if isinstance(self.site_names, str):
          self.site_names = [self.site_names]
        self.site_ids, _ = entity.find_sites(
          self.site_names, preserve_order=self.preserve_order
        )
        if (
          len(self.site_ids) == entity.num_sites
          and self.site_names == entity.site_names
        ):
          self.site_ids = slice(None)

      # Site indices --> site names.
      elif isinstance(self.site_ids, list):
        if isinstance(self.site_ids, int):
          self.site_ids = [self.site_ids]
        self.site_names = [entity.site_names[i] for i in self.site_ids]

  def _resolve_sensor_names(self, scene: Scene) -> None:
    if self.sensor_names is not None or isinstance(self.sensor_ids, list):
      entity: Entity = scene[self.name]

      # Sensor name regex --> sensor indices.
      if self.sensor_names is not None and isinstance(self.sensor_ids, list):
        if isinstance(self.sensor_names, str):
          self.sensor_names = [self.sensor_names]
        if isinstance(self.sensor_ids, int):
          self.sensor_ids = [self.sensor_ids]
        sensor_ids, _ = entity.find_sensors(
          self.sensor_names, preserve_order=self.preserve_order
        )
        sensor_names = [entity.sensor_names[i] for i in self.sensor_ids]
        if sensor_ids != self.sensor_ids or sensor_names != self.sensor_names:
          raise ValueError("Inconsistent sensor names and indices.")

      # Sensor indices --> sensor names.
      elif self.sensor_names is not None:
        if isinstance(self.sensor_names, str):
          self.sensor_names = [self.sensor_names]
        self.sensor_ids, _ = entity.find_sensors(
          self.sensor_names, preserve_order=self.preserve_order
        )
        if (
          len(self.sensor_ids) == entity.num_sensors
          and self.sensor_names == entity.sensor_names
        ):
          self.sensor_ids = slice(None)

      # Sensor indices --> sensor names.
      elif isinstance(self.sensor_ids, list):
        if isinstance(self.sensor_ids, int):
          self.sensor_ids = [self.sensor_ids]
        self.sensor_names = [entity.sensor_names[i] for i in self.sensor_ids]
