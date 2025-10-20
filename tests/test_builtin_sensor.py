"""Tests for BuiltinSensor functionality."""

from __future__ import annotations

import mujoco
import pytest
import torch

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor.builtin_sensor import BuiltinSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg


def get_test_device() -> str:
  """Get device for testing, preferring CUDA if available."""
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def articulated_robot_xml():
  """XML for a simple articulated robot with joints."""
  return """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="5.0"/>
          <site name="base_site" pos="0 0 0"/>
          <body name="link1" pos="0.3 0 0">
            <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
            <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
            <site name="link1_site" pos="0 0 0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
  """


def test_jointpos_sensor(articulated_robot_xml, device):
  """Verify joint pos sensor returns correctly shaped tensor for scalar joint values."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(articulated_robot_xml)
  )

  jointpos_sensor_cfg = BuiltinSensorCfg(
    name="joint1_pos",
    sensor_type="jointpos",
    objtype="joint",
    objname="joint1",
    obj_entity="robot",
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"robot": entity_cfg},
    sensors=(jointpos_sensor_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["joint1_pos"]
  sim.step()
  data = sensor.data

  assert isinstance(data, torch.Tensor)
  assert data.shape == (2, 1)


def test_accelerometer_sensor(articulated_robot_xml, device):
  """Verify accelerometer reads non-zero acceleration when robot is on floor."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(articulated_robot_xml)
  )

  accel_sensor_cfg = BuiltinSensorCfg(
    name="base_accel",
    sensor_type="accelerometer",
    objtype="site",
    objname="base_site",
    obj_entity="robot",
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"robot": entity_cfg},
    sensors=(accel_sensor_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  sensor = scene["base_accel"]

  # Step simulation multiple times to let robot fall and land on floor.
  for _ in range(100):
    sim.step()

  data = sensor.data

  assert isinstance(data, torch.Tensor)
  assert data.shape == (2, 3)
  assert torch.any(torch.abs(data) > 0)


def test_multiple_sensors(articulated_robot_xml, device):
  """Verify multiple sensors can be registered and return correctly shaped data."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(articulated_robot_xml)
  )

  jointpos_cfg = BuiltinSensorCfg(
    name="joint1_pos",
    sensor_type="jointpos",
    objtype="joint",
    objname="joint1",
    obj_entity="robot",
  )
  jointvel_cfg = BuiltinSensorCfg(
    name="joint1_vel",
    sensor_type="jointvel",
    objtype="joint",
    objname="joint1",
    obj_entity="robot",
  )
  gyro_cfg = BuiltinSensorCfg(
    name="base_gyro",
    sensor_type="gyro",
    objtype="site",
    objname="base_site",
    obj_entity="robot",
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=3.0,
    entities={"robot": entity_cfg},
    sensors=(jointpos_cfg, jointvel_cfg, gyro_cfg),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  jointpos_sensor = scene["joint1_pos"]
  jointvel_sensor = scene["joint1_vel"]
  gyro_sensor = scene["base_gyro"]

  sim.step()

  pos_data = jointpos_sensor.data
  vel_data = jointvel_sensor.data
  gyro_data = gyro_sensor.data

  assert pos_data.shape == (1, 1)
  assert vel_data.shape == (1, 1)
  assert gyro_data.shape == (1, 3)


def test_error_on_mismatched_ref_params(articulated_robot_xml, device):
  """Verify ValueError is raised when reftype is provided without refname."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(articulated_robot_xml)
  )

  invalid_sensor_cfg = BuiltinSensorCfg(
    name="invalid_sensor",
    sensor_type="jointpos",
    objtype="joint",
    objname="joint1",
    obj_entity="robot",
    reftype="body",
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=3.0,
    entities={"robot": entity_cfg},
    sensors=(invalid_sensor_cfg,),
  )

  with pytest.raises(ValueError, match="Provide both reftype and refname"):
    Scene(scene_cfg, device)
