"""Tests for domain randomization on the MuJoCo simulation backend."""

import mujoco
import numpy as np
import pytest
import torch

from mjlab.entity import EntityCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.mujoco_sim import MujocoModelBridge, MujocoModelField, MujocoSimulation
from mjlab.sim.sim import SimulationCfg

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" axis="1 0 0" range="-2 2" limited="true"/>
      <geom type="box" size="0.2 0.15 0.1" mass="1"/>
      <body name="pole">
        <joint name="hinge_1" type="hinge" axis="0 1 0" limited="false"/>
        <geom type="capsule" fromto="0 0 0 0 0 1" size="0.045" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide" joint="slider" gear="10"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def base_model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_string(_CARTPOLE_XML)


def _make_models(n: int) -> list[mujoco.MjModel]:
  return [mujoco.MjModel.from_xml_string(_CARTPOLE_XML) for _ in range(n)]


# ---------------------------------------------------------------------------
# MujocoModelField unit tests
# ---------------------------------------------------------------------------


def test_write_through_copies_only_updated_rows() -> None:
  """Writing env_ids=[0, 2] leaves model[1] unchanged."""
  models = _make_models(3)
  original_mass_1 = models[1].body_mass.copy()

  data = torch.tensor(
    np.array([m.body_mass.copy() for m in models]), dtype=torch.float32
  )
  field = MujocoModelField(data, models, "body_mass")

  env_ids = torch.tensor([0, 2])
  entity_ids = torch.arange(models[0].nbody)
  env_grid, entity_grid = torch.meshgrid(env_ids, entity_ids, indexing="ij")
  field[env_grid, entity_grid] = 42.0

  np.testing.assert_array_equal(models[1].body_mass, original_mass_1)
  np.testing.assert_allclose(models[0].body_mass, 42.0)
  np.testing.assert_allclose(models[2].body_mass, 42.0)


def test_model_bridge_field_access(base_model: mujoco.MjModel) -> None:
  """Bridge returns correctly-shaped fields, enforces read-only types, and caches."""
  bridge = MujocoModelBridge(2, base_model, [base_model], "cpu")

  # Lazy fields return MujocoModelField with shape (N, nentity[, ...]).
  assert isinstance(bridge.body_mass, MujocoModelField)
  assert bridge.body_mass.shape == (2, base_model.nbody)
  assert isinstance(bridge.jnt_range, MujocoModelField)
  assert bridge.jnt_range.shape == (2, base_model.njnt, 2)

  # geom_type is read-only (not per-env), so it is a plain tensor.
  assert isinstance(bridge.geom_type, torch.Tensor)
  assert not isinstance(bridge.geom_type, MujocoModelField)
  assert bridge.geom_type.shape == (base_model.ngeom,)

  # Field objects are cached across accesses.
  assert bridge.body_mass is bridge.body_mass


def test_write_through_preserves_int_dtype() -> None:
  """Integer-typed fields round-trip through the bridge without dtype loss."""
  models = _make_models(2)
  bridge = MujocoModelBridge(2, models[0], models, "cpu")

  # geom_contype is int32 in MjModel; writing through must not lossy-cast.
  field = bridge.geom_contype
  assert field.dtype == torch.int32
  env_ids = torch.tensor([0])
  geom_ids = torch.arange(models[0].ngeom)
  env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
  field[env_grid, geom_grid] = 5
  np.testing.assert_array_equal(models[0].geom_contype, 5)
  assert models[0].geom_contype.dtype == np.int32


# ---------------------------------------------------------------------------
# MujocoSimulation DR methods
# ---------------------------------------------------------------------------


@pytest.fixture
def sim(base_model: mujoco.MjModel) -> MujocoSimulation:
  return MujocoSimulation(
    num_envs=3, cfg=SimulationCfg(backend="mujoco"), model=base_model
  )


def test_expand_model_fields_creates_independent_copies(
  sim: MujocoSimulation,
) -> None:
  """expand_model_fields creates N independent copies and tracks expanded field names."""
  assert sim.expanded_fields == set()
  sim.expand_model_fields(("body_mass", "geom_size"))
  assert sim._mj_models_expanded
  assert len(sim._env_models) == sim.num_envs
  assert "body_mass" in sim.expanded_fields
  assert "geom_size" in sim.expanded_fields

  original_mass_1 = sim._env_models[1].body_mass.copy()
  sim._env_models[0].body_mass[:] = 999.0
  np.testing.assert_array_equal(sim._env_models[1].body_mass, original_mass_1)


def test_recompute_constants(sim: MujocoSimulation) -> None:
  """recompute_constants(none) is a noop; set_const updates derived fields."""
  sim.expand_model_fields(("body_mass",))
  original = sim._env_models[0].body_subtreemass.copy()

  sim.recompute_constants(RecomputeLevel.none)
  np.testing.assert_array_equal(sim._env_models[0].body_subtreemass, original)

  sim._env_models[0].body_mass[1] = 999.0  # cart body
  sim.recompute_constants(RecomputeLevel.set_const)
  assert sim._env_models[0].body_subtreemass[1] != pytest.approx(original[1], abs=1.0)


def test_get_default_field(
  sim: MujocoSimulation,
  base_model: mujoco.MjModel,
) -> None:
  """get_default_field returns the compile-time value and is cached."""
  defaults = sim.get_default_field("body_mass")
  np.testing.assert_allclose(defaults.numpy(), base_model.body_mass, rtol=1e-5)
  assert sim.get_default_field("body_mass") is defaults


# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------


class _MujocoEnv:
  """Minimal env-like object satisfying the interface expected by dr.* functions."""

  def __init__(self, scene: Scene, sim: MujocoSimulation, device: str) -> None:
    self.scene = scene
    self.sim = sim
    self.num_envs = sim.num_envs
    self.device = device


def _make_mujoco_env(
  num_envs: int,
  device: str = "cpu",
  expand_fields: tuple[str, ...] = (),
) -> _MujocoEnv:
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(_CARTPOLE_XML))
  scene_cfg = SceneCfg(
    num_envs=num_envs,
    env_spacing=5.0,
    entities={"robot": entity_cfg},
  )
  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim = MujocoSimulation(
    num_envs=num_envs, cfg=SimulationCfg(backend="mujoco"), model=model
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)  # type: ignore[arg-type]
  if expand_fields:
    sim.expand_model_fields(expand_fields)
  return _MujocoEnv(scene, sim, device)


@pytest.fixture
def cartpole_sim() -> MujocoSimulation:
  return MujocoSimulation(
    num_envs=2,
    cfg=SimulationCfg(backend="mujoco"),
    model=mujoco.MjModel.from_xml_string(_CARTPOLE_XML),
  )


def _step_and_assert_diverge(sim: MujocoSimulation, n_steps: int = 5) -> None:
  sim.data.ctrl[:] = 1.0
  for _ in range(n_steps):
    sim.step()
  assert not torch.allclose(sim.data.qpos[0], sim.data.qpos[1])


def test_reference_stability_after_expand(cartpole_sim: MujocoSimulation) -> None:
  """A MujocoModelField created before expand_model_fields sees the expanded models."""
  sim = cartpole_sim

  # Access the field before expansion — _env_models has 1 entry.
  field_before = sim.model.body_mass
  assert len(field_before._env_models) == 1

  sim.expand_model_fields(("body_mass",))

  # The same field object now reflects the expanded list.
  assert len(field_before._env_models) == 2

  # Writes through the pre-expansion reference still sync to the expanded models.
  pole_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pole")
  body_ids = torch.tensor([pole_id])
  e0, b = torch.meshgrid(torch.tensor([0]), body_ids, indexing="ij")
  field_before[e0, b] = 99.0
  np.testing.assert_allclose(sim._env_models[0].body_mass[pole_id], 99.0, rtol=1e-5)


def test_geom_size_dr_updates_bounds() -> None:
  """dr.geom_size scales geom_size and recomputes geom_rbound in the per-env models."""
  num_envs = 2
  env = _make_mujoco_env(
    num_envs=num_envs, expand_fields=("geom_size", "geom_rbound", "geom_aabb")
  )

  # Use entity indexing — the scene prefixes geom names with the entity name,
  # so entity indexing gives global model IDs directly.
  geom_ids = env.scene["robot"].indexing.geom_ids  # dtype=torch.int
  env_ids = torch.arange(num_envs, dtype=torch.int)
  env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")

  original_size = env.sim.model.geom_size[env_grid, geom_grid].clone()
  original_rbound = env.sim.model.geom_rbound[env_grid, geom_grid].clone()

  dr.geom_size(env, env_ids, ranges=(2.0, 2.0), operation="scale")

  for i in range(num_envs):
    m = env.sim._env_models[i]
    for j, geom_id in enumerate(geom_ids.tolist()):
      np.testing.assert_allclose(
        m.geom_size[geom_id, 0], 2.0 * original_size[i, j, 0].item(), rtol=1e-5
      )
      np.testing.assert_allclose(
        m.geom_rbound[geom_id], 2.0 * original_rbound[i, j].item(), rtol=1e-5
      )


def test_joint_stiffness_dr(cartpole_sim: MujocoSimulation) -> None:
  """Envs with different DOF stiffness produce diverging qpos from a displaced start."""
  sim = cartpole_sim
  mj_model = sim.mj_model
  sim.expand_model_fields(("jnt_stiffness",))

  hinge_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_1")
  hinge_qpos_id = int(mj_model.jnt_qposadr[hinge_id])
  jnt_ids = torch.tensor([hinge_id])

  e0, j = torch.meshgrid(torch.tensor([0]), jnt_ids, indexing="ij")
  e1, _ = torch.meshgrid(torch.tensor([1]), jnt_ids, indexing="ij")
  sim.model.jnt_stiffness[e0, j] = 1.0
  sim.model.jnt_stiffness[e1, j] = 100.0

  # Displace the hinge joint to trigger stiffness forces.
  sim.data.qpos[:, hinge_qpos_id] = 0.3
  for _ in range(20):
    sim.step()

  assert not torch.allclose(sim.data.qpos[0], sim.data.qpos[1])


def test_startup_then_reset_dr(cartpole_sim: MujocoSimulation) -> None:
  """DR persists through reset; re-randomizing after reset produces new divergence."""
  sim = cartpole_sim
  sim.expand_model_fields(("body_mass",))

  pole_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pole")
  body_ids = torch.tensor([pole_id])
  e0, b = torch.meshgrid(torch.tensor([0]), body_ids, indexing="ij")
  e1, _ = torch.meshgrid(torch.tensor([1]), body_ids, indexing="ij")

  # Phase 1: initial DR + rollout.
  sim.model.body_mass[e0, b] = 0.1
  sim.model.body_mass[e1, b] = 10.0
  sim.recompute_constants(RecomputeLevel.set_const)
  _step_and_assert_diverge(sim)

  # Reset env 0 and re-randomize; env 1 retains its mass from phase 1.
  sim.reset(torch.tensor([0]))
  sim.model.body_mass[e0, b] = 5.0
  sim.recompute_constants(RecomputeLevel.set_const)
  _step_and_assert_diverge(sim)
