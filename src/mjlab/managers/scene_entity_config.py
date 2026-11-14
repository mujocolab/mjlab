"""Configuration for scene entities used by manager terms."""

from dataclasses import dataclass, field
from typing import NamedTuple

import torch

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
  _FieldConfig("tendon_names", "tendon_ids", "find_tendons", "num_tendons", "tendon"),
  _FieldConfig(
    "camera_names",
    "camera_ids",
    "find_cameras",
    "num_cameras",
    "camera",
  ),
  _FieldConfig(
    "light_names",
    "light_ids",
    "find_lights",
    "num_lights",
    "light",
  ),
  _FieldConfig(
    "material_names",
    "material_ids",
    "find_materials",
    "num_materials",
    "material",
  ),
  _FieldConfig(
    "texture_names",
    "texture_ids",
    "find_textures",
    "num_textures",
    "texture",
  ),
  _FieldConfig("pair_names", "pair_ids", "find_pairs", "num_pairs", "pair"),
]


@dataclass
class SceneEntityCfg:
  """Configuration for a scene entity that is used by the manager's term.

  This configuration allows flexible specification of entity components either by name
  or by ID. During resolution, it ensures consistency between names and IDs, and can
  optimize to slice(None) when all components are selected.
  """

  name: str
  """The name of the entity in the scene."""

  joint_names: str | tuple[str, ...] | None = None
  """Names of joints to include. Can be a single string or tuple."""

  joint_ids: torch.Tensor | list[int] | slice = field(
    default_factory=lambda: slice(None)
  )
  """IDs of joints to include. Resolved to an on-device tensor or slice."""

  body_names: str | tuple[str, ...] | None = None
  """Names of bodies to include. Can be a single string or tuple."""

  body_ids: torch.Tensor | list[int] | slice = field(
    default_factory=lambda: slice(None)
  )
  """IDs of bodies to include. Resolved to an on-device tensor or slice."""

  geom_names: str | tuple[str, ...] | None = None
  """Names of geometries to include. Can be a single string or tuple."""

  geom_ids: torch.Tensor | list[int] | slice = field(
    default_factory=lambda: slice(None)
  )
  """IDs of geometries to include. Resolved to an on-device tensor or slice."""

  site_names: str | tuple[str, ...] | None = None
  """Names of sites to include. Can be a single string or tuple."""

  site_ids: torch.Tensor | list[int] | slice = field(
    default_factory=lambda: slice(None)
  )
  """IDs of sites to include. Resolved to an on-device tensor or slice."""

  actuator_names: str | list[str] | None = None
  """Names of actuators to include. Can be a single string or list."""

  actuator_ids: torch.Tensor | list[int] | slice = field(
    default_factory=lambda: slice(None)
  )
  """IDs of actuators to include. Resolved to an on-device tensor or slice."""

  tendon_names: str | tuple[str, ...] | None = None
  """Names of tendons to include. Can be a single string or tuple."""

  tendon_ids: torch.Tensor | list[int] | slice = field(
    default_factory=lambda: slice(None)
  )
  """IDs of tendons to include. Resolved to an on-device tensor or slice."""

  camera_names: str | tuple[str, ...] | None = None
  """Names of cameras to include. Can be a single string or tuple."""

  camera_ids: torch.Tensor | list[int] | slice = field(
    default_factory=lambda: slice(None)
  )
  """IDs of cameras to include. Resolved to an on-device tensor or slice."""

  light_names: str | tuple[str, ...] | None = None
  """Names of lights to include. Can be a single string or tuple."""

  light_ids: torch.Tensor | list[int] | slice = field(
    default_factory=lambda: slice(None)
  )
  """IDs of lights to include. Resolved to an on-device tensor or slice."""

  material_names: str | tuple[str, ...] | None = None
  """Names of materials to include. Can be a single string or tuple."""

  material_ids: torch.Tensor | list[int] | slice = field(
    default_factory=lambda: slice(None)
  )
  """IDs of materials to include. Resolved to an on-device tensor or slice."""

  texture_names: str | tuple[str, ...] | None = None
  """Names of textures to include. Can be a single string or tuple."""

  texture_ids: torch.Tensor | list[int] | slice = field(
    default_factory=lambda: slice(None)
  )
  """IDs of textures to include. Resolved to an on-device tensor or slice."""

  pair_names: str | tuple[str, ...] | None = None
  """Names of contact pairs to include. Can be a single string or tuple."""

  pair_ids: torch.Tensor | list[int] | slice = field(
    default_factory=lambda: slice(None)
  )
  """IDs of contact pairs to include. Resolved to an on-device tensor or slice."""

  preserve_order: bool = False
  """If True, maintains the order of components as specified."""

  def resolve(self, scene: Scene) -> None:
    """Resolve names and IDs for all configured fields.

    This method ensures consistency between names and IDs for each field type.
    It handles three cases:
    1. Both names and IDs provided: Validates they match
    2. Only names provided: Computes IDs (optimizes to slice(None) if all selected)
    3. Only IDs provided: Computes names

    Resolved non-slice ID collections are stored as ``torch.long`` tensors on the
    scene/entity device so indexing simulation tensors does not trigger implicit
    host-to-device copies or CUDA stream synchronizations.

    Args:
      scene: The scene containing the entity to resolve against.

    Raises:
      ValueError: If provided names and IDs are inconsistent.
      KeyError: If the entity name is not found in the scene.
    """
    entity = scene[self.name]
    device = self._infer_device(scene, entity)

    for config in _FIELD_CONFIGS:
      self._resolve_field(entity, config, device)

  @staticmethod
  def _infer_device(scene: Scene, entity: Entity) -> torch.device | str:
    scene_device = getattr(scene, "device", None)
    if isinstance(scene_device, (str, torch.device)):
      return scene_device
    data = getattr(entity, "_data", None)
    data_device = getattr(data, "device", None)
    if isinstance(data_device, (str, torch.device)):
      return data_device
    indexing = getattr(entity, "indexing", None)
    body_ids = getattr(indexing, "body_ids", None)
    if isinstance(body_ids, torch.Tensor):
      return body_ids.device
    return "cpu"

  def _resolve_field(
    self,
    entity: Entity,
    config: _FieldConfig,
    device: torch.device | str,
  ) -> None:
    """Resolve a single field's names and IDs.

    Args:
      entity: The entity to resolve against.
      config: Field configuration specifying attribute names and methods.
      device: Device on which resolved index tensors should live.
    """
    names = getattr(self, config.names_attr)
    ids = getattr(self, config.ids_attr)

    # Early return if nothing to resolve.
    if names is None and isinstance(ids, slice):
      return

    # Get entity metadata.
    entity_all_names = getattr(entity, config.names_attr)
    entity_count = getattr(entity, config.num_attr)
    find_method = getattr(entity, config.find_method)

    # Normalize single values and tensors to lists for uniform validation.
    names = self._normalize_to_list(names)
    ids_list: list[int] | None = None
    if not isinstance(ids, slice):
      ids_list = self._normalize_ids_to_list(ids)

    # Handle three resolution cases.
    if names is not None and ids_list is not None:
      setattr(self, config.names_attr, names)
      self._validate_consistency(
        names, ids_list, entity_all_names, find_method, config.kind_label
      )
      setattr(
        self,
        config.ids_attr,
        torch.tensor(ids_list, dtype=torch.long, device=device),
      )
    elif names is not None:
      self._resolve_names_to_ids(
        names,
        entity_all_names,
        entity_count,
        find_method,
        config.names_attr,
        config.ids_attr,
        device,
      )
    elif ids_list is not None:
      self._resolve_ids_to_names(ids_list, entity_all_names, config.names_attr)
      setattr(
        self,
        config.ids_attr,
        torch.tensor(ids_list, dtype=torch.long, device=device),
      )

  def _normalize_to_list(self, value: str | int | tuple | list | None) -> list | None:
    """Convert single values to lists for uniform processing."""
    if value is None:
      return None
    if isinstance(value, (str, int)):
      return [value]
    if isinstance(value, list):
      return value
    return list(value)

  @staticmethod
  def _normalize_ids_to_list(
    value: torch.Tensor | list[int] | tuple[int, ...] | int,
  ) -> list[int]:
    """Convert an ID specification into a plain Python list of ints."""
    if isinstance(value, torch.Tensor):
      if value.ndim == 0:
        return [int(value.item())]
      return [int(i) for i in value.tolist()]
    if isinstance(value, int):
      return [value]
    return [int(i) for i in value]

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
    names_attr: str,
    ids_attr: str,
    device: torch.device | str,
  ) -> None:
    """Resolve names to IDs, optimizing to slice(None) when all are selected."""
    found_ids, found_names = find_method(names, preserve_order=self.preserve_order)

    # Keep names and IDs in the same order.
    setattr(self, names_attr, found_names)

    # Optimize to slice(None) if all components are selected in order.
    if len(found_ids) == entity_count and found_names == list(entity_all_names):
      setattr(self, ids_attr, slice(None))
    else:
      setattr(
        self,
        ids_attr,
        torch.tensor(found_ids, dtype=torch.long, device=device),
      )

  def _resolve_ids_to_names(
    self, ids: list[int], entity_all_names: list[str], names_attr: str
  ) -> None:
    """Resolve IDs to their corresponding names."""
    resolved_names = [entity_all_names[i] for i in ids]
    setattr(self, names_attr, resolved_names)
