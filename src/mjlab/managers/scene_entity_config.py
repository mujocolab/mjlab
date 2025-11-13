"""Configuration for scene entities used by manager terms."""

from dataclasses import dataclass, field
from typing import NamedTuple

from mjlab.entity import Entity
from mjlab.scene import Scene


class _FieldConfig(NamedTuple):
  """Configuration for a resolvable entity field."""

  names_attr: str
  ids_attr: str
  find_method: str
  num_attr: str
  kind_label: str


_FIELD_CONFIGS = [
  _FieldConfig("joint_names", "joint_ids", "find_joints", "num_joints", "joint"),
  _FieldConfig("body_names", "body_ids", "find_bodies", "num_bodies", "body"),
  _FieldConfig("geom_names", "geom_ids", "find_geoms", "num_geoms", "geom"),
  _FieldConfig("site_names", "site_ids", "find_sites", "num_sites", "site"),
  _FieldConfig(
    "actuator_names", "actuator_ids", "find_actuators", "num_actuators", "actuator"
  ),
]


@dataclass
class SceneEntityCfg:
  """Configuration for a scene entity that is used by the manager's term.

  This configuration allows flexible specification of entity components either by name
  or by ID. During resolution, it ensures consistency between names and IDs, and can
  optimize to slice(None) when all components are selected.

  Attributes:
    name: The name of the entity in the scene.
    joint_names: Names of joints to include. Can be a single string or tuple.
    joint_ids: IDs of joints to include. Can be a list or slice.
    body_names: Names of bodies to include. Can be a single string or tuple.
    body_ids: IDs of bodies to include. Can be a list or slice.
    geom_names: Names of geometries to include. Can be a single string or tuple.
    geom_ids: IDs of geometries to include. Can be a list or slice.
    site_names: Names of sites to include. Can be a single string or tuple.
    site_ids: IDs of sites to include. Can be a list or slice.
    actuator_names: Names of actuators to include. Can be a single string or list.
    actuator_ids: IDs of actuators to include. Can be a list or slice.
    preserve_order: If True, maintains the order of components as specified. If False,
      allows reordering for optimization.
  """

  name: str

  joint_names: str | tuple[str, ...] | None = None
  joint_ids: list[int] | slice = field(default_factory=lambda: slice(None))

  body_names: str | tuple[str, ...] | None = None
  body_ids: list[int] | slice = field(default_factory=lambda: slice(None))

  geom_names: str | tuple[str, ...] | None = None
  geom_ids: list[int] | slice = field(default_factory=lambda: slice(None))

  site_names: str | tuple[str, ...] | None = None
  site_ids: list[int] | slice = field(default_factory=lambda: slice(None))

  actuator_names: str | list[str] | None = None
  actuator_ids: list[int] | slice = field(default_factory=lambda: slice(None))

  preserve_order: bool = False

  def resolve(self, scene: Scene) -> None:
    """Resolve names and IDs for all configured fields.

    This method ensures consistency between names and IDs for each field type.
    It handles three cases:
    1. Both names and IDs provided: Validates they match
    2. Only names provided: Computes IDs (optimizes to slice(None) if all selected)
    3. Only IDs provided: Computes names

    Args:
      scene: The scene containing the entity to resolve against.

    Raises:
      ValueError: If provided names and IDs are inconsistent.
      KeyError: If the entity name is not found in the scene.
    """
    entity = scene[self.name]

    for config in _FIELD_CONFIGS:
      self._resolve_field(entity, config)

  def _resolve_field(self, entity: Entity, config: _FieldConfig) -> None:
    """Resolve a single field's names and IDs.

    Args:
      entity: The entity to resolve against.
      config: Field configuration specifying attribute names and methods.
    """
    names = getattr(self, config.names_attr)
    ids = getattr(self, config.ids_attr)

    # Early return if nothing to resolve.
    if names is None and not isinstance(ids, list):
      return

    # Get entity metadata.
    entity_all_names = getattr(entity, config.names_attr)
    entity_count = getattr(entity, config.num_attr)
    find_method = getattr(entity, config.find_method)

    # Normalize single values to lists for uniform processing.
    names = self._normalize_to_list(names)
    if names is not None:
      setattr(self, config.names_attr, names)

    if isinstance(ids, (int, list)):
      ids = self._normalize_to_list(ids)
      setattr(self, config.ids_attr, ids)

    # Handle three resolution cases.
    if names is not None and isinstance(ids, list):
      self._validate_consistency(
        names, ids, entity_all_names, find_method, config.kind_label
      )
    elif names is not None:
      self._resolve_names_to_ids(
        names, entity_all_names, entity_count, find_method, config.ids_attr
      )
    elif isinstance(ids, list):
      self._resolve_ids_to_names(ids, entity_all_names, config.names_attr)

  def _normalize_to_list(self, value: str | int | list | None) -> list | None:
    """Convert single values to lists for uniform processing."""
    if value is None:
      return None
    if isinstance(value, (str, int)):
      return [value]
    return value

  def _validate_consistency(
    self,
    names: list[str],
    ids: list[int],
    entity_all_names: list[str],
    find_method,
    kind_label: str,
  ) -> None:
    """Validate that provided names and IDs are consistent.

    Raises:
      ValueError: If names and IDs don't match.
    """
    found_ids, _ = find_method(names, preserve_order=self.preserve_order)
    computed_names = [entity_all_names[i] for i in ids]

    if found_ids != ids or computed_names != names:
      raise ValueError(
        f"Inconsistent {kind_label} names and indices. "
        f"Names {names} resolved to indices {found_ids}, "
        f"but indices {ids} (mapping to names {computed_names}) were provided."
      )

  def _resolve_names_to_ids(
    self,
    names: list[str],
    entity_all_names: list[str],
    entity_count: int,
    find_method,
    ids_attr: str,
  ) -> None:
    """Resolve names to IDs, optimizing to slice(None) when all are selected."""
    found_ids, _ = find_method(names, preserve_order=self.preserve_order)

    # Optimize to slice(None) if all components are selected in order.
    if len(found_ids) == entity_count and names == entity_all_names:
      setattr(self, ids_attr, slice(None))
    else:
      setattr(self, ids_attr, found_ids)

  def _resolve_ids_to_names(
    self, ids: list[int], entity_all_names: list[str], names_attr: str
  ) -> None:
    """Resolve IDs to their corresponding names."""
    resolved_names = [entity_all_names[i] for i in ids]
    setattr(self, names_attr, resolved_names)
