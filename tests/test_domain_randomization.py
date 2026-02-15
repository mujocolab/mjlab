"""Tests for domain randomization functionality."""

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.envs.mdp.dr import (
  _randomize_model_field,
  body_com_offset,
  body_inertia,
  body_mass,
  geom_friction,
  geom_pos,
  geom_rgba,
  joint_armature,
  joint_damping,
  joint_friction,
  joint_limits,
  joint_stiffness,
  site_pos,
)
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg

# Suppress the expected warning from sim_data.py about index_put_ on expanded tensors.
pytestmark = pytest.mark.filterwarnings(
  "ignore:Use of index_put_ on expanded tensors is deprecated:UserWarning"
)

ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="1.0"
        friction="0.5 0.01 0.005"/>
      <site name="base_site" pos="0 0 0.1"/>
      <body name="foot1" pos="0.2 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
        <geom name="foot1_geom" type="box" size="0.05 0.05 0.05" mass="0.1"
          friction="0.5 0.01 0.005"/>
        <site name="foot1_site" pos="0 0 0"/>
      </body>
      <body name="foot2" pos="-0.2 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="0 1.57"/>
        <geom name="foot2_geom" type="box" size="0.05 0.05 0.05" mass="0.1"
          friction="0.5 0.01 0.005"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

FRICTION_RANGE = (0.3, 1.2)
DAMPING_RANGE = (0.1, 0.5)
NUM_ENVS = 4


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


def create_test_env(device, num_envs=NUM_ENVS, expand_fields=None):
  """Create a test environment with a robot for domain randomization testing."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML))
  scene_cfg = SceneCfg(num_envs=num_envs, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()

  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  scene.initialize(model, sim.model, sim.data)
  if expand_fields is None:
    expand_fields = ("geom_friction", "dof_damping")
  sim.expand_model_fields(expand_fields)

  class Env:
    def __init__(self, scene, sim):
      self.scene = scene
      self.sim = sim
      self.num_envs = scene.num_envs
      self.device = device

  return Env(scene, sim)


def assert_values_in_range(values, min_val, max_val):
  """Assert all values are within the specified range."""
  assert torch.all((values >= min_val) & (values <= max_val))


def assert_values_changed(old_values, new_values):
  """Assert that values changed after randomization."""
  assert not torch.all(new_values == old_values)


def assert_has_diversity(values, min_unique=2):
  """Assert that values have sufficient diversity across environments."""
  unique_values = torch.unique(values)
  assert len(unique_values) >= min_unique


def test_geom_friction(device):
  """Test that friction randomization changes values and respects ranges."""
  torch.manual_seed(123)
  env = create_test_env(device)
  robot = env.scene["robot"]

  indices = robot.indexing.geom_ids
  initial_values = env.sim.model.geom_friction[:, indices[0], 0].clone()

  geom_friction(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=FRICTION_RANGE,
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
  )

  new_values = env.sim.model.geom_friction[:, indices[0], 0]
  assert_values_changed(initial_values, new_values)
  assert_values_in_range(new_values, FRICTION_RANGE[0], FRICTION_RANGE[1])
  assert_has_diversity(new_values)


def test_joint_damping(device):
  """Test that joint damping randomization works."""
  torch.manual_seed(789)
  env = create_test_env(device)
  robot = env.scene["robot"]

  indices = robot.indexing.joint_v_adr
  env.sim.model.dof_damping[:, indices] = 0.0
  initial_values = env.sim.model.dof_damping[:, indices[0]].clone()

  joint_damping(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=DAMPING_RANGE,
    asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
  )

  new_values = env.sim.model.dof_damping[:, indices[0]]
  assert_values_changed(initial_values, new_values)
  assert_values_in_range(new_values, DAMPING_RANGE[0], DAMPING_RANGE[1])
  assert_has_diversity(new_values)


@pytest.mark.skipif(
  not torch.cuda.is_available(), reason="CUDA required for graph capture"
)
def test_expand_model_fields_recreates_cuda_graph(device):
  """Verify that CUDA graph is recreated after expand_model_fields."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML))
  scene_cfg = SceneCfg(num_envs=NUM_ENVS, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()

  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=NUM_ENVS, cfg=sim_cfg, model=model, device=device)
  scene.initialize(model, sim.model, sim.data)

  if not sim.use_cuda_graph:
    pytest.skip("CUDA graph capture not enabled on this device")

  original_step_graph = sim.step_graph

  sim.expand_model_fields(("geom_friction",))

  assert sim.step_graph is not original_step_graph, (
    "CUDA graph was not recreated after expand_model_fields"
  )


def test_friction_scale_uses_defaults(device):
  """Verify scale operations use defaults to prevent accumulation."""
  env = create_test_env(device, num_envs=2)
  robot = env.scene["robot"]

  geom_idx = robot.indexing.geom_ids[0]
  default_friction = env.sim.get_default_field("geom_friction")[geom_idx, 0].item()

  # Randomize 3 times with scale — should NOT accumulate.
  for _ in range(3):
    geom_friction(
      env,  # pyright: ignore[reportArgumentType]
      env_ids=None,
      ranges=(2.0, 2.0),
      operation="scale",
      asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
      axes=[0],
    )

  final_friction = env.sim.model.geom_friction[0, geom_idx, 0].item()
  assert abs(final_friction - default_friction * 2.0) < 1e-5


def test_friction_scale_partial_axes(device):
  """Verify scale on partial axes doesn't affect non-randomized axes."""
  env = create_test_env(device, num_envs=2)
  robot = env.scene["robot"]
  env.sim.expand_model_fields(("geom_friction",))

  geom_idx = robot.indexing.geom_ids[0]
  default_friction = env.sim.get_default_field("geom_friction")[geom_idx].clone()

  geom_friction(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(2.0, 2.0),
    operation="scale",
    asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
    axes=[0],
  )

  final_friction = env.sim.model.geom_friction[0, geom_idx]

  assert abs(final_friction[0] - default_friction[0] * 2.0) < 1e-5, (
    f"Expected axis 0 to be {default_friction[0] * 2.0}, got {final_friction[0]}"
  )
  assert abs(final_friction[1] - default_friction[1]) < 1e-5, (
    f"Expected axis 1 to remain {default_friction[1]}, got {final_friction[1]}"
  )
  assert abs(final_friction[2] - default_friction[2]) < 1e-5, (
    f"Expected axis 2 to remain {default_friction[2]}, got {final_friction[2]}"
  )


def test_single_env_without_expand(device):
  """Verify randomization works with num_envs=1 without expand_model_fields."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML))
  scene_cfg = SceneCfg(num_envs=1, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()

  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(model, sim.model, sim.data)

  class Env:
    def __init__(self, scene, sim):
      self.scene = scene
      self.sim = sim
      self.num_envs = scene.num_envs
      self.device = device

  env = Env(scene, sim)
  robot = env.scene["robot"]

  geom_idx = robot.indexing.geom_ids[0]
  original_friction = sim.model.geom_friction[0, geom_idx, 0].item()

  geom_friction(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(2.0, 2.0),
    operation="scale",
    asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
    axes=[0],
  )

  final_friction = sim.model.geom_friction[0, geom_idx, 0].item()
  assert abs(final_friction - original_friction * 2.0) < 1e-5

  # Randomize again — no accumulation.
  geom_friction(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(2.0, 2.0),
    operation="scale",
    asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
    axes=[0],
  )

  final_friction_2 = sim.model.geom_friction[0, geom_idx, 0].item()
  assert abs(final_friction_2 - original_friction * 2.0) < 1e-5


def test_shared_random(device):
  """Verify shared_random broadcasts a single value to all entities per env."""
  torch.manual_seed(42)
  env = create_test_env(device, num_envs=4)
  robot = env.scene["robot"]

  geom_ids = robot.indexing.geom_ids

  geom_friction(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=FRICTION_RANGE,
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
    shared_random=True,
  )

  friction = env.sim.model.geom_friction[:, geom_ids, 0]

  for env_idx in range(env.num_envs):
    env_friction = friction[env_idx]
    assert torch.allclose(env_friction, env_friction[0].expand_as(env_friction)), (
      f"Env {env_idx} has different friction values: {env_friction}"
    )

  env_frictions = friction[:, 0]
  assert len(torch.unique(env_frictions)) > 1, "All envs have the same friction"
  assert_values_in_range(friction, FRICTION_RANGE[0], FRICTION_RANGE[1])


def test_body_mass_updates_subtreemass(device):
  """Verify body_mass calls recompute_constants (set_const)."""
  env = create_test_env(
    device, num_envs=2, expand_fields=("body_mass", "body_subtreemass")
  )
  robot = env.scene["robot"]

  body_ids = robot.indexing.body_ids
  original_subtreemass = env.sim.model.body_subtreemass[:, body_ids].clone()

  body_mass(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(2.0, 2.0),
    operation="scale",
    asset_cfg=SceneEntityCfg("robot", body_names=(".*",)),
  )

  # Subtreemass is a derived quantity — verify it actually updated.
  # (The decorator declares recompute="set_const" which is called by the
  # EventManager at the end of apply(). But in this unit test we call
  # the function directly, so we manually recompute.)
  env.sim.recompute_constants("set_const")

  new_subtreemass = env.sim.model.body_subtreemass[:, body_ids]
  assert not torch.allclose(original_subtreemass, new_subtreemass), (
    "body_subtreemass was not updated after body_mass randomization"
  )


def test_per_component_ranges(device):
  """Verify string-keyed dict ranges resolve per-component patterns."""
  env = create_test_env(device, num_envs=2)
  robot = env.scene["robot"]

  v_adr = robot.indexing.joint_v_adr

  joint_damping(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges={"joint1": (10.0, 10.0), "joint2": (20.0, 20.0)},
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
  )

  joint1_damping = env.sim.model.dof_damping[:, v_adr[0]]
  joint2_damping = env.sim.model.dof_damping[:, v_adr[1]]

  assert torch.allclose(joint1_damping, torch.tensor(10.0, device=device))
  assert torch.allclose(joint2_damping, torch.tensor(20.0, device=device))


def test_derived_fields_auto_expanded(device):
  """Verify decorator auto-includes derived fields for recompute levels."""
  fields = getattr(body_mass, "model_fields")  # noqa: B009
  assert "body_mass" in fields
  assert "body_subtreemass" in fields
  assert getattr(body_mass, "recompute") == "set_const"  # noqa: B009

  # geom_friction has no recompute — no derived fields.
  assert getattr(geom_friction, "model_fields") == ("geom_friction",)  # noqa: B009
  assert getattr(geom_friction, "recompute") == "none"  # noqa: B009


def test_requires_model_fields_decorator():
  """Test that the decorator correctly computes all_fields with derived."""

  @requires_model_fields("body_mass", recompute="set_const")
  def dummy(env, env_ids):
    pass

  fields = getattr(dummy, "model_fields")  # noqa: B009
  assert "body_mass" in fields
  assert "body_subtreemass" in fields
  assert "dof_invweight0" in fields
  assert getattr(dummy, "recompute") == "set_const"  # noqa: B009


def test_com_offset_randomization(device):
  """Verify body_com_offset works and recomputes derived fields."""
  env = create_test_env(
    device,
    num_envs=2,
    expand_fields=("body_ipos", "body_subtreemass"),
  )
  robot = env.scene["robot"]

  body_ids = robot.indexing.body_ids
  original_ipos = env.sim.model.body_ipos[:, body_ids].clone()

  body_com_offset(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges={0: (-0.1, 0.1), 1: (-0.1, 0.1), 2: (-0.1, 0.1)},
    operation="add",
    asset_cfg=SceneEntityCfg("robot", body_names=(".*",)),
  )

  new_ipos = env.sim.model.body_ipos[:, body_ids]
  assert not torch.allclose(original_ipos, new_ipos), (
    "body_ipos was not changed after body_com_offset"
  )

  # Verify recompute metadata is set correctly.
  assert getattr(body_com_offset, "recompute") == "set_const"  # noqa: B009

  # Manually recompute (EventManager does this automatically).
  env.sim.recompute_constants("set_const")

  new_subtreemass = env.sim.model.body_subtreemass[:, body_ids]
  # body_subtreemass should still be valid after recompute (sanity check).
  assert new_subtreemass.sum() > 0


def test_body_inertia(device):
  """Verify body_inertia modifies body_inertia."""
  env = create_test_env(device, num_envs=2, expand_fields=("body_inertia",))
  robot = env.scene["robot"]
  body_ids = robot.indexing.body_ids
  original = env.sim.model.body_inertia[:, body_ids].clone()

  body_inertia(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(2.0, 2.0),
    operation="scale",
    asset_cfg=SceneEntityCfg("robot", body_names=(".*",)),
  )

  env.sim.recompute_constants("set_const_0")
  new = env.sim.model.body_inertia[:, body_ids]
  assert not torch.allclose(original, new), "body_inertia was not changed"


def test_joint_armature(device):
  """Verify joint_armature modifies dof_armature."""
  env = create_test_env(device, num_envs=2, expand_fields=("dof_armature",))
  robot = env.scene["robot"]
  v_adr = robot.indexing.joint_v_adr

  joint_armature(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(5.0, 5.0),
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
  )

  new = env.sim.model.dof_armature[:, v_adr]
  assert torch.allclose(new, torch.tensor(5.0, device=device))


def test_joint_friction(device):
  """Verify joint_friction modifies dof_frictionloss."""
  env = create_test_env(device, num_envs=2, expand_fields=("dof_frictionloss",))
  robot = env.scene["robot"]
  v_adr = robot.indexing.joint_v_adr

  joint_friction(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(1.0, 1.0),
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
  )

  new = env.sim.model.dof_frictionloss[:, v_adr]
  assert torch.allclose(new, torch.tensor(1.0, device=device))


def test_joint_stiffness(device):
  """Verify joint_stiffness modifies jnt_stiffness."""
  env = create_test_env(device, num_envs=2, expand_fields=("jnt_stiffness",))
  robot = env.scene["robot"]
  joint_ids = robot.indexing.joint_ids

  joint_stiffness(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(3.0, 3.0),
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
  )

  new = env.sim.model.jnt_stiffness[:, joint_ids]
  assert torch.allclose(new, torch.tensor(3.0, device=device))


def test_joint_limits(device):
  """Verify joint_limits modifies jnt_range."""
  env = create_test_env(device, num_envs=2, expand_fields=("jnt_range",))
  robot = env.scene["robot"]
  joint_ids = robot.indexing.joint_ids
  original = env.sim.model.jnt_range[:, joint_ids].clone()

  joint_limits(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(-0.1, 0.1),
    operation="add",
    asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
  )

  new = env.sim.model.jnt_range[:, joint_ids]
  assert not torch.allclose(original, new), "jnt_range was not changed"


def test_log_uniform_distribution(device):
  """Verify log_uniform distribution produces values in expected range."""
  torch.manual_seed(42)
  env = create_test_env(device, num_envs=4)

  geom_friction(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(0.1, 10.0),
    operation="abs",
    distribution="log_uniform",
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
  )

  geom_ids = env.scene["robot"].indexing.geom_ids
  friction = env.sim.model.geom_friction[:, geom_ids, 0]
  assert_values_in_range(friction, 0.1, 10.0)
  assert_has_diversity(friction)


def test_gaussian_distribution(device):
  """Verify gaussian distribution produces values centered around mean."""
  torch.manual_seed(42)
  env = create_test_env(device, num_envs=4)

  # Gaussian: ranges=(mean, std)
  geom_friction(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(0.5, 0.01),
    operation="abs",
    distribution="gaussian",
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
  )

  geom_ids = env.scene["robot"].indexing.geom_ids
  friction = env.sim.model.geom_friction[:, geom_ids, 0]
  # With std=0.01, values should be very close to mean=0.5.
  assert_values_in_range(friction, 0.3, 0.7)


def test_event_manager_recompute_integration(device):
  """Verify EventManager auto-recomputes when DR terms fire."""
  from unittest.mock import Mock, patch

  from mjlab.managers.event_manager import EventManager, EventTermCfg

  # Create a no-op function with the same decorator attributes as
  # body_mass — we only care about EventManager's recompute
  # tracking, not the actual randomization logic.
  def fake_dr(env, env_ids, **kwargs):
    pass

  fake_dr.model_fields = getattr(body_mass, "model_fields")  # noqa: B009  # type: ignore[attr-defined]
  fake_dr.recompute = getattr(body_mass, "recompute")  # noqa: B009  # type: ignore[attr-defined]

  env = Mock()
  env.num_envs = 2
  env.device = device
  env.scene = {"robot": Mock()}
  env.sim = Mock()
  env.sim.recompute_constants = Mock()

  cfg = {
    "mass": EventTermCfg(
      mode="reset",
      func=fake_dr,
      params={},
    ),
  }

  with patch.object(SceneEntityCfg, "resolve"):
    manager = EventManager(cfg, env)

  # apply() should call recompute_constants with the fired term's level.
  env.sim.recompute_constants.reset_mock()
  manager.apply("reset", env_ids=torch.tensor([0]), global_env_step_count=1)
  env.sim.recompute_constants.assert_called_once_with("set_const")


def test_event_manager_no_recompute_when_no_dr(device):
  """Verify EventManager does NOT recompute when no DR terms fire."""
  from unittest.mock import Mock

  from mjlab.managers.event_manager import EventManager, EventTermCfg

  # A no-op function with no recompute metadata (simulates a non-DR term).
  def fake_non_dr(env, env_ids, **kwargs):
    pass

  env = Mock()
  env.num_envs = 2
  env.device = device
  env.scene = {"robot": Mock()}
  env.sim = Mock()
  env.sim.recompute_constants = Mock()

  cfg = {
    "reset": EventTermCfg(
      mode="startup",
      func=fake_non_dr,
      params={},
    ),
  }

  manager = EventManager(cfg, env)
  env.sim.recompute_constants.reset_mock()
  manager.apply("startup")
  env.sim.recompute_constants.assert_not_called()


def test_recompute_constants_validates_level(device):
  """Verify recompute_constants rejects invalid levels."""
  env = create_test_env(device, num_envs=1, expand_fields=())
  with pytest.raises(ValueError, match="Unknown recompute level"):
    env.sim.recompute_constants("invalid_level")


@pytest.mark.slow
def test_g1_foot_friction_shared_across_geoms(device):
  """Verify G1 velocity env has uniform foot friction across all collision geoms.

  The G1 robot has 7 collision geometries per foot (14 total). This test ensures
  that the shared_random parameter in foot_friction event config correctly
  broadcasts the same friction value to all foot geoms within each environment.

  Regression test for issue #481.
  """
  import io
  import warnings
  from contextlib import redirect_stderr, redirect_stdout

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

  cfg = unitree_g1_flat_env_cfg()

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg, device=device)

  try:
    robot = env.scene["robot"]

    # Get the foot collision geom indices (7 per foot, 14 total).
    foot_geom_names = [
      f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
    ]
    foot_geom_ids, _ = robot.find_geoms(foot_geom_names)
    foot_geom_indices = robot.indexing.geom_ids[foot_geom_ids]

    # Get friction values for all foot geoms.
    friction = env.sim.model.geom_friction[:, foot_geom_indices, 0]

    # All foot geoms within each env should have the same friction.
    for env_idx in range(env.num_envs):
      env_friction = friction[env_idx]
      assert torch.allclose(env_friction, env_friction[0].expand_as(env_friction)), (
        f"Env {env_idx} has different friction values across foot geoms: {env_friction}"
      )

    # Different envs should have different friction values (with high probability).
    if env.num_envs > 1:
      env_frictions = friction[:, 0]
      assert len(torch.unique(env_frictions)) > 1, (
        "All envs have the same friction - shared_random may not be working"
      )
  finally:
    env.close()


def test_empty_dict_ranges_is_noop(device):
  """Verify empty dict ranges doesn't crash and is a no-op."""
  env = create_test_env(device, num_envs=2)
  robot = env.scene["robot"]

  geom_ids = robot.indexing.geom_ids
  original = env.sim.model.geom_friction[:, geom_ids].clone()

  # Should return early without error.
  _randomize_model_field(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    field="geom_friction",
    entity_type="geom",
    ranges={},
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
  )

  new = env.sim.model.geom_friction[:, geom_ids]
  assert torch.allclose(original, new), "Empty ranges should be a no-op"


def test_requires_model_fields_rejects_invalid_level():
  """Verify decorator rejects unknown recompute levels."""
  with pytest.raises(ValueError, match="Unknown recompute level"):

    @requires_model_fields("body_mass", recompute="invalid")
    def dummy(env, env_ids):
      pass


def test_geom_pos_randomization(device):
  """Verify geom_pos modifies geom_pos."""
  env = create_test_env(device, num_envs=2, expand_fields=("geom_pos",))
  robot = env.scene["robot"]

  geom_ids = robot.indexing.geom_ids
  original = env.sim.model.geom_pos[:, geom_ids].clone()

  geom_pos(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(-0.1, 0.1),
    operation="add",
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
  )

  new = env.sim.model.geom_pos[:, geom_ids]
  assert not torch.allclose(original, new), "geom_pos was not changed"


def test_site_pos_randomization(device):
  """Verify site_pos modifies site_pos."""
  env = create_test_env(device, num_envs=2, expand_fields=("site_pos",))
  robot = env.scene["robot"]

  site_ids = robot.indexing.site_ids
  original = env.sim.model.site_pos[:, site_ids].clone()

  site_pos(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(-0.1, 0.1),
    operation="add",
    asset_cfg=SceneEntityCfg("robot", site_names=(".*",)),
  )

  new = env.sim.model.site_pos[:, site_ids]
  assert not torch.allclose(original, new), "site_pos was not changed"


def test_geom_rgba_randomization(device):
  """Verify geom_rgba modifies geom_rgba."""
  env = create_test_env(device, num_envs=2, expand_fields=("geom_rgba",))
  robot = env.scene["robot"]

  geom_ids = robot.indexing.geom_ids
  original = env.sim.model.geom_rgba[:, geom_ids].clone()

  geom_rgba(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    ranges=(0.0, 1.0),
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
  )

  new = env.sim.model.geom_rgba[:, geom_ids]
  assert not torch.allclose(original, new), "geom_rgba was not changed"
