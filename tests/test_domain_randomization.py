"""Tests for domain randomization functionality."""

import mujoco
import pytest
import torch

from mjlab.entity import EntityCfg
from mjlab.envs.mdp.events import randomize_field
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg


def get_test_device() -> str:
  """Get device for testing, preferring CUDA if available."""
  if torch.cuda.is_available():
    return "cuda:0"
  return "cpu"


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def robot_xml():
  """Simple robot with geoms and joints."""
  return """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="1.0" friction="0.5 0.01 0.005"/>
          <body name="foot1" pos="0.2 0 0">
            <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="foot1_geom" type="box" size="0.05 0.05 0.05" mass="0.1" friction="0.5 0.01 0.005"/>
          </body>
          <body name="foot2" pos="-0.2 0 0">
            <joint name="joint2" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="foot2_geom" type="box" size="0.05 0.05 0.05" mass="0.1" friction="0.5 0.01 0.005"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """


@pytest.fixture
def initialized_env(robot_xml, device):
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml))
  scene_cfg = SceneCfg(num_envs=4, entities={"robot": entity_cfg})

  scene = Scene(scene_cfg, device)
  model = scene.compile()

  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=4, cfg=sim_cfg, model=model, device=device)

  scene.initialize(model, sim.model, sim.data)

  class MockEnv:
    def __init__(self, scene, sim):
      self.scene = scene
      self.sim = sim
      self.num_envs = scene.num_envs
      self.device = device

  return MockEnv(scene, sim)


def test_randomize_geom_friction_changes_values(initialized_env):
  """Test that randomizing geom friction changes values."""
  env = initialized_env
  robot = env.scene["robot"]
  geom_indices = robot.indexing.geom_ids
  initial_friction = env.sim.model.geom_friction[0, geom_indices[0], 0].clone()

  randomize_field(
    env,
    env_ids=None,
    field="geom_friction",
    ranges=(0.3, 1.2),
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", geom_names=[".*"]),
    axes=[0],
  )

  new_friction = env.sim.model.geom_friction[0, geom_indices[0], 0]
  assert new_friction != initial_friction
  assert 0.3 <= new_friction <= 1.2


def test_randomize_multiple_geoms(initialized_env):
  """Test that randomization works across multiple geoms."""
  env = initialized_env
  robot = env.scene["robot"]
  geom_indices = robot.indexing.geom_ids

  initial_frictions = []
  for geom_idx in geom_indices:
    initial_frictions.append(env.sim.model.geom_friction[0, geom_idx, 0].item())

  randomize_field(
    env,
    env_ids=None,
    field="geom_friction",
    ranges=(0.3, 1.2),
    operation="abs",
    asset_cfg=SceneEntityCfg("robot"),
    axes=[0],
  )

  for i, geom_idx in enumerate(geom_indices):
    new_friction = env.sim.model.geom_friction[0, geom_idx, 0].item()
    if new_friction != initial_frictions[i]:
      return

  pytest.fail("No geom friction values changed after randomization")


def test_randomize_respects_range(initialized_env):
  """Test that randomization produces values within specified range."""
  env = initialized_env
  robot = env.scene["robot"]

  randomize_field(
    env,
    env_ids=None,
    field="geom_friction",
    ranges=(0.8, 1.5),
    operation="abs",
    asset_cfg=SceneEntityCfg("robot"),
  )

  geom_indices = robot.indexing.geom_ids
  for geom_idx in geom_indices:
    friction_values = env.sim.model.geom_friction[:, geom_idx, 0]
    assert torch.all((friction_values >= 0.8) & (friction_values <= 1.5)), (
      f"Friction out of range: {friction_values}"
    )


def test_randomize_body_mass_scale(initialized_env):
  """Test that scaling body mass works correctly."""
  env = initialized_env
  robot = env.scene["robot"]
  body_indices = robot.indexing.body_ids
  initial_mass = env.sim.model.body_mass[0, body_indices[0]].clone()

  randomize_field(
    env,
    env_ids=None,
    field="body_mass",
    ranges=(0.8, 1.2),
    operation="scale",
    asset_cfg=SceneEntityCfg("robot", body_names=[".*"]),
  )

  new_mass = env.sim.model.body_mass[0, body_indices[0]]
  assert new_mass != initial_mass
  assert 0.8 * initial_mass <= new_mass <= 1.2 * initial_mass


def test_randomize_dof_damping(initialized_env):
  """Test that DOF damping randomization works."""
  env = initialized_env
  robot = env.scene["robot"]
  joint_ids = robot.indexing.joint_ids

  if len(joint_ids) == 0:
    pytest.skip("No joints in robot")

  dof_indices = robot.indexing.joint_v_adr
  env.sim.model.dof_damping[:, dof_indices] = 0.0

  randomize_field(
    env,
    env_ids=None,
    field="dof_damping",
    ranges=(0.1, 0.5),
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", joint_names=[".*"]),
  )

  damping_values = env.sim.model.dof_damping[:, dof_indices]
  assert torch.all((damping_values >= 0.1) & (damping_values <= 0.5))
