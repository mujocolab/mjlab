from dataclasses import dataclass, field

from mjlab.entity import Entity
from mjlab.scene import Scene


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

  preserve_order: bool = False

  def resolve(self, scene: Scene) -> None:
    self._resolve_joint_names(scene)
    self._resolve_body_names(scene)
    self._resolve_geom_names(scene)
    self._resolve_site_names(scene)

  def _resolve_joint_names(self, scene: Scene) -> None:
    self._resolve_names(
      scene,
      names_field="joint_names",
      ids_field="joint_ids",
      find_method="find_joints",
      entity_names_attr="joint_names",
      entity_num_attr="num_joints",
    )

  def _resolve_body_names(self, scene: Scene) -> None:
    self._resolve_names(
      scene,
      names_field="body_names",
      ids_field="body_ids",
      find_method="find_bodies",
      entity_names_attr="body_names",
      entity_num_attr="num_bodies",
    )

  def _resolve_geom_names(self, scene: Scene) -> None:
    self._resolve_names(
      scene,
      names_field="geom_names",
      ids_field="geom_ids",
      find_method="find_geoms",
      entity_names_attr="geom_names",
      entity_num_attr="num_geoms",
    )

  def _resolve_site_names(self, scene: Scene) -> None:
    self._resolve_names(
      scene,
      names_field="site_names",
      ids_field="site_ids",
      find_method="find_sites",
      entity_names_attr="site_names",
      entity_num_attr="num_sites",
    )

  def _resolve_names(
    self,
    scene: Scene,
    *,
    names_field: str,
    ids_field: str,
    find_method: str,
    entity_names_attr: str,
    entity_num_attr: str,
  ) -> None:
    """Resolve names/ids for an entity component.

    Handles three cases:
    1. Names and IDs provided: validate they are consistent.
    2. Names only: resolve to IDs and canonical names; collapse to `slice(None)` if full selection.
    3. IDs only: resolve to names.
    """
    entity: Entity = scene[self.name]

    names = getattr(self, names_field)
    ids = getattr(self, ids_field)

    if names is None and not isinstance(ids, list):
      return

    # Names + IDs --> validate both ways.
    if names is not None and not isinstance(ids, slice):
      if isinstance(names, str):
        names = [names]
      if isinstance(ids, int):
        ids = [ids]

      resolver = getattr(entity, find_method)
      resolved_ids, resolved_names = resolver(names, preserve_order=self.preserve_order)
      entity_names = getattr(entity, entity_names_attr)
      try:
        names_from_ids = [entity_names[i] for i in ids]
      except IndexError as e:
        component_type = names_field.replace("_names", "")
        raise ValueError(
          f"Invalid {component_type} ID in {ids}. "
          f"Entity '{self.name}' has {len(entity_names)} {component_type}s (valid IDs: 0-{len(entity_names) - 1})."
        ) from e
      if resolved_ids != ids or names_from_ids != names:
        component_type = names_field.replace("_names", "")
        raise ValueError(
          f"Inconsistent {component_type} names and indices. "
          f"Names {names!r} resolve to IDs {resolved_ids}, but IDs {ids} resolve to names {names_from_ids}."
        )

      # Persist normalized forms.
      setattr(self, names_field, names)
      setattr(self, ids_field, ids)
      return

    # Names only --> ids + canonical names.
    if names is not None:
      if isinstance(names, str):
        names = [names]
      resolver = getattr(entity, find_method)
      resolved_ids, resolved_names = resolver(names, preserve_order=self.preserve_order)
      setattr(self, ids_field, resolved_ids)
      setattr(self, names_field, resolved_names)

      # Collapse to slice(None) when selecting all in canonical order.
      if len(resolved_ids) == getattr(
        entity, entity_num_attr
      ) and resolved_names == getattr(entity, entity_names_attr):
        setattr(self, ids_field, slice(None))
      return

    # IDs only --> names.
    if isinstance(ids, (list, int)):
      if isinstance(ids, int):
        ids = [ids]
      entity_names = getattr(entity, entity_names_attr)
      try:
        resolved_names = [entity_names[i] for i in ids]
      except IndexError as e:
        component_type = names_field.replace("_names", "")
        raise ValueError(
          f"Invalid {component_type} ID in {ids}. "
          f"Entity '{self.name}' has {len(entity_names)} {component_type}s (valid IDs: 0-{len(entity_names) - 1})."
        ) from e
      setattr(self, ids_field, ids)
      setattr(self, names_field, resolved_names)
