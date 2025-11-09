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
  "field,ranges,operation,asset_names,axes,seed",
  [
    ("geom_friction", FRICTION_RANGE, "abs", {"geom_names": [".*"]}, [0], 123),
    ("body_mass", MASS_SCALE_RANGE, "scale", {"body_names": [".*"]}, None, 456),
    ("dof_damping", DAMPING_RANGE, "abs", {"joint_names": [".*"]}, None, 789),
  ],
)
def test_randomize_field(device, field, ranges, operation, asset_names, axes, seed):
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
    env,  # type: ignore[arg-type]
    env_ids=None,
    field=field,
    ranges=ranges,
    operation=operation,
    asset_cfg=SceneEntityCfg("robot", **asset_names),
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
