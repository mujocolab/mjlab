"""Unit tests for SceneEntityCfg resolution logic."""

from dataclasses import dataclass

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import Entity, EntityCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg


@dataclass
class _FakeIndexing:
  """Fake indexing for testing. Assumes all joints are 1-DOF (hinge-like)."""

  num_joints: int

  def get_q_dof_ids(
    self, joint_ids: torch.Tensor | list[int] | slice
  ) -> torch.Tensor | slice:
    if isinstance(joint_ids, slice):
      return torch.arange(self.num_joints)
    if isinstance(joint_ids, list):
      return torch.tensor(joint_ids)
    return joint_ids

  def get_v_dof_ids(
    self, joint_ids: torch.Tensor | list[int] | slice
  ) -> torch.Tensor | slice:
    if isinstance(joint_ids, slice):
      return torch.arange(self.num_joints)
    if isinstance(joint_ids, list):
      return torch.tensor(joint_ids)
    return joint_ids


@dataclass
class _FakeData:
  """Fake data for testing."""

  indexing: _FakeIndexing


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

  @property
  def data(self) -> _FakeData:
    return _FakeData(indexing=_FakeIndexing(num_joints=self.num_joints))

  # find_* helpers return (ids, names) similar to Entity API.
  def _find(
    self, query_names: tuple[str, ...], pool: tuple[str, ...]
  ) -> tuple[list[int], list[str]]:
    # Treat query as exact names (no regex) to keep tests minimal.
    indices = [list(pool).index(n) for n in query_names]
    names = [list(pool)[i] for i in indices]
    return indices, names

  def find_joints(self, query_names: tuple[str, ...], preserve_order: bool = False):
    return self._find(query_names, self.joint_names)

  def find_bodies(self, query_names: tuple[str, ...], preserve_order: bool = False):
    return self._find(query_names, self.body_names)

  def find_geoms(self, query_names: tuple[str, ...], preserve_order: bool = False):
    return self._find(query_names, self.geom_names)

  def find_sites(self, query_names: tuple[str, ...], preserve_order: bool = False):
    return self._find(query_names, self.site_names)


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


# ============================================================================
# Ball Joint DOF Resolution Tests (using real Entity)
# ============================================================================


MIXED_JOINTS_XML = """
<mujoco>
  <worldbody>
    <body name="base">
      <joint name="hinge0" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.1"/>
      <body name="link1" pos="0.2 0 0">
        <joint name="ball0" type="ball"/>
        <geom type="sphere" size="0.1"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="hinge1" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def mixed_joint_scene():
  """Create a real scene with mixed joint types."""
  device = get_test_device()
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(MIXED_JOINTS_XML))
  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return {"robot": entity}


def test_joint_q_ids_expands_for_ball_joint(mixed_joint_scene):
  """Test joint_q_ids correctly expands to 4 indices for a ball joint."""
  cfg = SceneEntityCfg(name="robot", joint_names=("ball0",))
  cfg.resolve(mixed_joint_scene)

  # Ball joint should expand to 4 qpos DOFs.
  assert cfg.joint_q_ids == [1, 2, 3, 4]


def test_joint_v_ids_expands_for_ball_joint(mixed_joint_scene):
  """Test joint_v_ids correctly expands to 3 indices for a ball joint."""
  cfg = SceneEntityCfg(name="robot", joint_names=("ball0",))
  cfg.resolve(mixed_joint_scene)

  # Ball joint should expand to 3 qvel DOFs.
  assert cfg.joint_v_ids == [1, 2, 3]


def test_joint_q_ids_mixed_selection(mixed_joint_scene):
  """Test joint_q_ids for selecting hinge + ball (1 + 4 = 5 DOFs)."""
  cfg = SceneEntityCfg(name="robot", joint_names=("hinge0", "ball0"))
  cfg.resolve(mixed_joint_scene)

  # hinge0: 1 DOF at index 0, ball0: 4 DOFs at indices 1-4.
  assert cfg.joint_q_ids == [0, 1, 2, 3, 4]


def test_joint_v_ids_mixed_selection(mixed_joint_scene):
  """Test joint_v_ids for selecting hinge + ball (1 + 3 = 4 DOFs)."""
  cfg = SceneEntityCfg(name="robot", joint_names=("hinge0", "ball0"))
  cfg.resolve(mixed_joint_scene)

  # hinge0: 1 DOF at index 0, ball0: 3 DOFs at indices 1-3.
  assert cfg.joint_v_ids == [0, 1, 2, 3]


def test_joint_q_ids_hinges_only(mixed_joint_scene):
  """Test joint_q_ids for selecting only hinges (skip ball)."""
  cfg = SceneEntityCfg(name="robot", joint_names=("hinge0", "hinge1"))
  cfg.resolve(mixed_joint_scene)

  # hinge0: index 0, hinge1: index 5.
  assert cfg.joint_q_ids == [0, 5]


def test_joint_v_ids_hinges_only(mixed_joint_scene):
  """Test joint_v_ids for selecting only hinges (skip ball)."""
  cfg = SceneEntityCfg(name="robot", joint_names=("hinge0", "hinge1"))
  cfg.resolve(mixed_joint_scene)

  # hinge0: index 0, hinge1: index 4.
  assert cfg.joint_v_ids == [0, 4]


def test_joint_dof_ids_slice_when_all_joints(mixed_joint_scene):
  """Test joint_q_ids/joint_v_ids remain slice(None) when all joints selected."""
  cfg = SceneEntityCfg(name="robot", joint_names=("hinge0", "ball0", "hinge1"))
  cfg.resolve(mixed_joint_scene)

  # When all joints are selected and joint_ids becomes slice(None),
  # joint_q_ids and joint_v_ids should also be slice(None).
  assert cfg.joint_ids == slice(None)
  assert cfg.joint_q_ids == slice(None)
  assert cfg.joint_v_ids == slice(None)
