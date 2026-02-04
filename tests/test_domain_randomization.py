"""Tests for domain randomization functionality."""

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.envs.mdp.events import randomize_field
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
      <body name="foot1" pos="0.2 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
        <geom name="foot1_geom" type="box" size="0.05 0.05 0.05" mass="0.1"
          friction="0.5 0.01 0.005"/>
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
MASS_SCALE_RANGE = (0.8, 1.2)
DAMPING_RANGE = (0.1, 0.5)
NUM_ENVS = 4


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


def create_test_env(device, num_envs=NUM_ENVS):
  """Create a test environment with a robot for domain randomization testing."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML))
  scene_cfg = SceneCfg(num_envs=num_envs, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()

  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  scene.initialize(model, sim.model, sim.data)
  sim.expand_model_fields(("geom_friction", "body_mass", "dof_damping"))

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


@pytest.mark.parametrize(
  "field,ranges,operation,entity_names,axes,seed",
  [
    ("geom_friction", FRICTION_RANGE, "abs", {"geom_names": [".*"]}, [0], 123),
    ("body_mass", MASS_SCALE_RANGE, "scale", {"body_names": [".*"]}, None, 456),
    ("dof_damping", DAMPING_RANGE, "abs", {"joint_names": [".*"]}, None, 789),
  ],
)
def test_randomize_field(device, field, ranges, operation, entity_names, axes, seed):
  """Test that randomization changes values, respects ranges, and creates diversity."""
  torch.manual_seed(seed)
  env = create_test_env(device)
  robot = env.scene["robot"]

  # Get the appropriate indices and initial values based on field type.
  model_field: torch.Tensor
  initial_values: torch.Tensor
  if field == "geom_friction":
    indices = robot.indexing.geom_ids
    model_field = env.sim.model.geom_friction[:, indices[0], 0]
    initial_values = model_field.clone()
  elif field == "body_mass":
    indices = robot.indexing.body_ids
    model_field = env.sim.model.body_mass[:, indices[0]]
    initial_values = model_field.clone()
  else:
    assert field == "dof_damping"
    indices = robot.indexing.joint_v_adr
    env.sim.model.dof_damping[:, indices] = 0.0
    model_field = env.sim.model.dof_damping[:, indices[0]]
    initial_values = model_field.clone()

  randomize_field(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    field=field,
    ranges=ranges,
    operation=operation,
    asset_cfg=SceneEntityCfg("robot", **entity_names),
    axes=axes,
  )

  new_values = model_field

  assert_values_changed(initial_values, new_values)

  if operation == "abs":
    assert_values_in_range(new_values, ranges[0], ranges[1])
  elif operation == "scale":
    assert_values_in_range(
      new_values, ranges[0] * initial_values, ranges[1] * initial_values
    )

  assert_has_diversity(new_values)


@pytest.mark.skipif(
  not torch.cuda.is_available(), reason="CUDA required for graph capture"
)
def test_expand_model_fields_recreates_cuda_graph(device):
  """Verify that CUDA graph is recreated after expand_model_fields.

  Regression test for a bug where expand_model_fields replaced arrays with new
  allocations, but the CUDA graph still held pointers to the old addresses.
  The simulation ran fine but silently ignored domain randomization. The graph
  kept reading from the old arrays containing the original non-randomized values.
  """
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

  # Graph should have been recreated (different object identity).
  # If the graph isn't recreated, domain randomization silently doesn't apply because
  # the graph reads from the old arrays.
  assert sim.step_graph is not original_step_graph, (
    "CUDA graph was not recreated after expand_model_fields"
  )


def test_randomize_field_scale_uses_defaults(device):
  """Verify scale/add operations use defaults to prevent accumulation."""
  env = create_test_env(device, num_envs=2)
  robot = env.scene["robot"]
  env.sim.expand_model_fields(("body_mass",))

  body_idx = robot.indexing.body_ids[0]
  default_mass = env.sim.get_default_field("body_mass")[body_idx].item()

  # Randomize 3 times with scale operation.
  for _ in range(3):
    randomize_field(
      env,  # pyright: ignore[reportArgumentType]
      env_ids=None,
      field="body_mass",
      ranges=(2.0, 2.0),
      operation="scale",
      asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
    )

  # Values should NOT accumulate.
  final_mass = env.sim.model.body_mass[0, body_idx].item()
  assert abs(final_mass - default_mass * 2.0) < 1e-5


def test_randomize_field_scale_partial_axes(device):
  """Verify scale operation on partial axes doesn't affect non-randomized axes."""
  env = create_test_env(device, num_envs=2)
  robot = env.scene["robot"]
  env.sim.expand_model_fields(("geom_friction",))

  geom_idx = robot.indexing.geom_ids[0]
  default_friction = env.sim.get_default_field("geom_friction")[geom_idx].clone()

  # Randomize only axis 0 (sliding friction) with scale operation.
  randomize_field(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    field="geom_friction",
    ranges=(2.0, 2.0),  # Scale by 2.0
    operation="scale",
    asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
    axes=[0],  # Only randomize axis 0
  )

  final_friction = env.sim.model.geom_friction[0, geom_idx]

  # Axis 0 should be scaled by 2.0
  assert abs(final_friction[0] - default_friction[0] * 2.0) < 1e-5, (
    f"Expected axis 0 to be {default_friction[0] * 2.0}, got {final_friction[0]}"
  )

  # Axes 1 and 2 should remain unchanged at default values
  assert abs(final_friction[1] - default_friction[1]) < 1e-5, (
    f"Expected axis 1 to remain {default_friction[1]}, got {final_friction[1]}"
  )
  assert abs(final_friction[2] - default_friction[2]) < 1e-5, (
    f"Expected axis 2 to remain {default_friction[2]}, got {final_friction[2]}"
  )


def test_randomize_field_single_env_without_expand(device):
  """Verify randomization works with num_envs=1 without calling expand_model_fields.

  For single-env simulations, expand_model_fields is not required because
  all worlds share the same memory. The default values should be lazily
  stored when first needed for scale/add operations.
  """
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML))
  scene_cfg = SceneCfg(num_envs=1, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()

  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(model, sim.model, sim.data)
  # Intentionally NOT calling sim.expand_model_fields().

  class Env:
    def __init__(self, scene, sim):
      self.scene = scene
      self.sim = sim
      self.num_envs = scene.num_envs
      self.device = device

  env = Env(scene, sim)
  robot = env.scene["robot"]

  body_idx = robot.indexing.body_ids[0]
  original_mass = sim.model.body_mass[0, body_idx].item()

  # Randomize with scale operation should work without expand_model_fields.
  randomize_field(
    env,  # type: ignore[arg-type]
    env_ids=None,
    field="body_mass",
    ranges=(2.0, 2.0),
    operation="scale",
    asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
  )

  final_mass = sim.model.body_mass[0, body_idx].item()
  assert abs(final_mass - original_mass * 2.0) < 1e-5, (
    f"Expected mass {original_mass * 2.0}, got {final_mass}"
  )

  # Randomize again should still be original * 2.0 (no accumulation).
  randomize_field(
    env,  # type: ignore[arg-type]
    env_ids=None,
    field="body_mass",
    ranges=(2.0, 2.0),
    operation="scale",
    asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
  )

  final_mass_2 = sim.model.body_mass[0, body_idx].item()
  assert abs(final_mass_2 - original_mass * 2.0) < 1e-5, (
    f"Expected mass {original_mass * 2.0} (no accumulation), got {final_mass_2}"
  )


def test_randomize_field_shared_random(device):
  """Verify shared_random broadcasts a single value to all entities per environment."""
  torch.manual_seed(42)
  env = create_test_env(device, num_envs=4)
  robot = env.scene["robot"]

  # Get all geom indices (base_geom, foot1_geom, foot2_geom).
  geom_ids = robot.indexing.geom_ids

  randomize_field(
    env,  # pyright: ignore[reportArgumentType]
    env_ids=None,
    field="geom_friction",
    ranges=FRICTION_RANGE,
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
    axes=[0],
    shared_random=True,
  )

  friction = env.sim.model.geom_friction[:, geom_ids, 0]

  # All geoms within the same env should have the same friction.
  for env_idx in range(env.num_envs):
    env_friction = friction[env_idx]
    assert torch.allclose(env_friction, env_friction[0].expand_as(env_friction)), (
      f"Env {env_idx} has different friction values: {env_friction}"
    )

  # Different envs should have different friction values.
  env_frictions = friction[:, 0]
  assert len(torch.unique(env_frictions)) > 1, "All envs have the same friction"

  # All values should be in range.
  assert_values_in_range(friction, FRICTION_RANGE[0], FRICTION_RANGE[1])


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
