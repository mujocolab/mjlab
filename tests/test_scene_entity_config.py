"""Unit tests for SceneEntityCfg resolution logic."""

from dataclasses import dataclass

import pytest
import torch
from conftest import get_test_device

from mjlab.managers.scene_entity_config import SceneEntityCfg


@dataclass
class _FakeEntity:
  name: str

  joint_names: tuple[str, ...]
  body_names: tuple[str, ...]
  geom_names: tuple[str, ...]
  site_names: tuple[str, ...]

  @property
  def num_joints(self) -> int:
    return len(self.joint_names)

  @property
  def num_bodies(self) -> int:
    return len(self.body_names)

  @property
  def num_geoms(self) -> int:
    return len(self.geom_names)

  @property
  def num_sites(self) -> int:
    return len(self.site_names)

  # find_* helpers return (ids, names) similar to Entity API.
  def _find(
    self,
    query_names: tuple[str, ...],
    pool: tuple[str, ...],
    preserve_order: bool,
  ) -> tuple[list[int], list[str]]:
    # Treat query as exact names (no regex) to keep tests minimal.
    indices = [list(pool).index(n) for n in query_names]
    if not preserve_order:
      indices.sort()
    names = [list(pool)[i] for i in indices]
    return indices, names

  def find_joints(self, query_names: tuple[str, ...], preserve_order: bool = False):
    return self._find(query_names, self.joint_names, preserve_order)

  def find_bodies(self, query_names: tuple[str, ...], preserve_order: bool = False):
    return self._find(query_names, self.body_names, preserve_order)

  def find_geoms(self, query_names: tuple[str, ...], preserve_order: bool = False):
    return self._find(query_names, self.geom_names, preserve_order)

  def find_sites(self, query_names: tuple[str, ...], preserve_order: bool = False):
    return self._find(query_names, self.site_names, preserve_order)


@pytest.fixture
def fake_entity() -> _FakeEntity:
  names = ("a", "b", "c")
  return _FakeEntity(
    name="robot",
    joint_names=names,
    body_names=names,
    geom_names=names,
    site_names=names,
  )


@pytest.fixture
def fake_scene(fake_entity: _FakeEntity):
  # Minimal scene-like mapping.
  return {fake_entity.name: fake_entity}


@pytest.mark.parametrize(
  "field_names",
  [
    ("joint_names", "joint_ids"),
    ("body_names", "body_ids"),
    ("geom_names", "geom_ids"),
    ("site_names", "site_ids"),
  ],
)
def test_names_to_ids_sets_slice_when_all(fake_scene, fake_entity, field_names):
  """Providing full set of names resolves ids and collapses to slice(None)."""
  names_attr, ids_attr = field_names

  cfg = SceneEntityCfg(name=fake_entity.name)
  setattr(cfg, names_attr, getattr(fake_entity, names_attr))

  cfg.resolve(fake_scene)

  ids_value = getattr(cfg, ids_attr)
  assert isinstance(ids_value, slice), f"{ids_attr} should collapse to slice(None)"
  assert ids_value == slice(None)


@pytest.mark.parametrize(
  "field_names,ids",
  [
    (("joint_names", "joint_ids"), [0, 2]),
    (("body_names", "body_ids"), [1]),
    (("geom_names", "geom_ids"), [2, 0]),
    (("site_names", "site_ids"), [1, 2]),
  ],
)
def test_ids_to_names_resolves_names(fake_scene, fake_entity, field_names, ids):
  """Providing explicit ids populates the corresponding names list and tensor ids."""
  names_attr, ids_attr = field_names

  cfg = SceneEntityCfg(name=fake_entity.name)
  setattr(cfg, ids_attr, ids.copy())

  cfg.resolve(fake_scene)

  names_value = getattr(cfg, names_attr)
  ids_value = getattr(cfg, ids_attr)
  expected = [getattr(fake_entity, names_attr)[i] for i in ids]
  assert names_value == expected
  assert isinstance(ids_value, torch.Tensor)
  assert ids_value.dtype == torch.long
  assert ids_value.tolist() == ids


@pytest.mark.parametrize(
  "field_names,provided_names,provided_ids",
  [
    (("joint_names", "joint_ids"), ["a"], [1]),
    (("body_names", "body_ids"), ["b"], [2]),
    (("geom_names", "geom_ids"), ["c"], [0]),
    (("site_names", "site_ids"), ["a"], [2]),
  ],
)
def test_inconsistent_names_and_ids_raise(
  fake_scene, field_names, provided_names, provided_ids
):
  """When both names and ids are provided but disagree, a ValueError is raised."""
  names_attr, ids_attr = field_names

  cfg = SceneEntityCfg(name="robot")
  setattr(cfg, names_attr, provided_names.copy())
  setattr(cfg, ids_attr, provided_ids.copy())  # Must be list to trigger check.

  with pytest.raises(ValueError):
    cfg.resolve(fake_scene)


def test_tuple_names_with_consistent_ids(fake_scene):
  """Tuple names with matching IDs should validate and convert IDs to a tensor."""
  cfg = SceneEntityCfg(name="robot")
  cfg.joint_names = ("a", "b")
  cfg.joint_ids = [0, 1]

  cfg.resolve(fake_scene)

  assert cfg.joint_names == ["a", "b"]
  assert isinstance(cfg.joint_ids, torch.Tensor)
  assert cfg.joint_ids.dtype == torch.long
  assert cfg.joint_ids.tolist() == [0, 1]


@pytest.mark.parametrize(
  "field_names",
  [
    ("joint_names", "joint_ids"),
    ("body_names", "body_ids"),
    ("geom_names", "geom_ids"),
    ("site_names", "site_ids"),
  ],
)
def test_names_reordered_to_match_ids_when_not_preserving_order(
  fake_scene, field_names
):
  """With preserve_order=False, names are reordered to internal order."""
  names_attr, ids_attr = field_names

  # Give a subset in reverse of internal order.
  cfg = SceneEntityCfg(name="robot")
  setattr(cfg, names_attr, ["c", "a"])

  cfg.resolve(fake_scene)

  # IDs and names must both be in internal order: a=0, c=2.
  ids_value = getattr(cfg, ids_attr)
  assert isinstance(ids_value, torch.Tensor)
  assert ids_value.tolist() == [0, 2]
  assert getattr(cfg, names_attr) == ["a", "c"]


@pytest.mark.parametrize(
  "field_names",
  [
    ("joint_names", "joint_ids"),
    ("body_names", "body_ids"),
    ("geom_names", "geom_ids"),
    ("site_names", "site_ids"),
  ],
)
def test_names_preserve_user_order_when_preserving_order(fake_scene, field_names):
  """With preserve_order=True, names stay in the user-specified order."""
  names_attr, ids_attr = field_names

  cfg = SceneEntityCfg(name="robot", preserve_order=True)
  setattr(cfg, names_attr, ["c", "a"])

  cfg.resolve(fake_scene)

  # User order preserved: c=2, a=0.
  ids_value = getattr(cfg, ids_attr)
  assert isinstance(ids_value, torch.Tensor)
  assert ids_value.tolist() == [2, 0]
  assert getattr(cfg, names_attr) == ["c", "a"]


def test_resolve_places_ids_on_scene_device_and_is_idempotent(fake_entity):
  """Resolved ID tensors live on scene.device and survive repeated resolve() calls."""
  device = get_test_device()

  class _DeviceScene:
    def __init__(self, entity: _FakeEntity, dev: str) -> None:
      self._entities = {entity.name: entity}
      self.device = dev

    def __getitem__(self, key: str) -> _FakeEntity:
      return self._entities[key]

  scene = _DeviceScene(fake_entity, device)
  cfg = SceneEntityCfg(name="robot", joint_names=("c", "a"), preserve_order=True)
  cfg.resolve(scene)  # ty: ignore[invalid-argument-type]

  assert isinstance(cfg.joint_ids, torch.Tensor)
  assert cfg.joint_ids.device == torch.device(device)
  assert cfg.joint_ids.dtype == torch.long
  assert cfg.joint_ids.tolist() == [2, 0]

  # Indexing an on-device tensor directly works without host conversion.
  joint_pos = torch.tensor([[10.0, 20.0, 30.0]], device=device)
  selected = joint_pos[:, cfg.joint_ids]
  assert selected.tolist() == [[30.0, 10.0]]

  # Calling resolve() again is idempotent.
  cfg.resolve(scene)  # ty: ignore[invalid-argument-type]
  assert isinstance(cfg.joint_ids, torch.Tensor)
  assert cfg.joint_ids.device == torch.device(device)
  assert cfg.joint_ids.tolist() == [2, 0]
