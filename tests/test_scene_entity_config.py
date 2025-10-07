"""Unit tests for SceneEntityCfg resolution logic."""

from dataclasses import dataclass
from typing import List, Tuple

import pytest

from mjlab.managers.scene_entity_config import SceneEntityCfg


@dataclass
class _FakeEntity:
  name: str

  joint_names: List[str]
  body_names: List[str]
  geom_names: List[str]
  site_names: List[str]

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
    self, query_names: List[str], pool: List[str]
  ) -> Tuple[List[int], List[str]]:
    # Treat query as exact names (no regex) to keep tests minimal.
    indices = [pool.index(n) for n in query_names]
    names = [pool[i] for i in indices]
    return indices, names

  def find_joints(self, query_names: List[str], preserve_order: bool = False):
    return self._find(query_names, self.joint_names)

  def find_bodies(self, query_names: List[str], preserve_order: bool = False):
    return self._find(query_names, self.body_names)

  def find_geoms(self, query_names: List[str], preserve_order: bool = False):
    return self._find(query_names, self.geom_names)

  def find_sites(self, query_names: List[str], preserve_order: bool = False):
    return self._find(query_names, self.site_names)


@pytest.fixture
def fake_entity() -> _FakeEntity:
  names = ["a", "b", "c"]
  return _FakeEntity(
    name="robot",
    joint_names=names.copy(),
    body_names=names.copy(),
    geom_names=names.copy(),
    site_names=names.copy(),
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
  setattr(cfg, names_attr, getattr(fake_entity, names_attr).copy())

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
  """Providing explicit ids populates the corresponding names list."""
  names_attr, ids_attr = field_names

  cfg = SceneEntityCfg(name=fake_entity.name)
  setattr(cfg, ids_attr, ids.copy())

  cfg.resolve(fake_scene)

  names_value = getattr(cfg, names_attr)
  expected = [getattr(fake_entity, names_attr)[i] for i in ids]
  assert names_value == expected


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
