"""Shared test fixtures and utilities."""

import os
from pathlib import Path

import mujoco
import pytest
import torch
import warp as wp

from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
  """Configure test environment settings automatically for all tests."""
  wp.config.quiet = True


def get_test_device() -> str:
  """Get device for testing, preferring CUDA if available.

  Can be overridden with FORCE_CPU=1 environment variable to test
  CPU-only behavior on GPU machines.
  """
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def fixtures_dir() -> Path:
  """Path to test fixtures directory."""
  return Path(__file__).parent / "fixtures"


def load_fixture_xml(fixture_name: str) -> str:
  """Load XML content from fixture file.

  Args:
    fixture_name: Name of the fixture file (without .xml extension) or full path.

  Returns:
    XML content as string.
  """
  fixtures_path = Path(__file__).parent / "fixtures"
  if not fixture_name.endswith(".xml"):
    fixture_name = f"{fixture_name}.xml"
  fixture_file = fixtures_path / fixture_name
  return fixture_file.read_text()


def create_entity_with_actuator(xml_string: str, actuator_cfg):
  """Create entity with actuator from XML string.

  Args:
    xml_string: MuJoCo XML model string.
    actuator_cfg: Actuator configuration.

  Returns:
    Entity instance.
  """
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(xml_string),
    articulation=EntityArticulationInfoCfg(actuators=(actuator_cfg,)),
  )
  return Entity(cfg)


def create_entity_from_fixture(fixture_name: str, actuator_cfg=None):
  """Create entity from fixture file.

  Args:
    fixture_name: Name of the fixture file (without .xml extension).
    actuator_cfg: Optional actuator configuration.

  Returns:
    Entity instance.
  """
  xml_string = load_fixture_xml(fixture_name)
  if actuator_cfg is not None:
    return create_entity_with_actuator(xml_string, actuator_cfg)
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml_string))
  return Entity(cfg)


def initialize_entity(entity: Entity, device: str, num_envs: int = 1):
  """Initialize entity with simulation.

  Args:
    entity: Entity to initialize.
    device: Device to use ("cpu" or "cuda").
    num_envs: Number of environments.

  Returns:
    Tuple of (entity, simulation).
  """
  model = entity.compile()
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


# =============================================================================
# XML Fixture Loaders
# =============================================================================


@pytest.fixture
def fixed_base_box_xml() -> str:
  """Load fixed base box XML fixture."""
  return load_fixture_xml("fixed_base_box")


@pytest.fixture
def floating_base_box_xml() -> str:
  """Load floating base box XML fixture."""
  return load_fixture_xml("floating_base_box")


@pytest.fixture
def fixed_base_articulated_xml() -> str:
  """Load fixed base articulated robot XML fixture."""
  return load_fixture_xml("fixed_base_articulated")


@pytest.fixture
def floating_base_articulated_xml() -> str:
  """Load floating base articulated robot XML fixture."""
  return load_fixture_xml("floating_base_articulated")


@pytest.fixture
def biped_xml() -> str:
  """Load biped robot XML fixture with ground plane."""
  return load_fixture_xml("biped")
