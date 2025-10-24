"""Tests for builtin_sensor.py."""

from __future__ import annotations

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor.builtin_sensor import BuiltinSensorCfg, ObjRef
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def robot_with_xml_sensors():
  """XML for robot with sensors already defined in the XML."""
  return """
    <mujoco>
      <worldbody>
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
      <sensor>
        <jointpos name="xml_joint_sensor" joint="joint1"/>
        <accelerometer name="xml_accel_sensor" site="base_site"/>
        <gyro name="xml_gyro_sensor" site="link1_site"/>
      </sensor>
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
    obj=ObjRef(type="joint", name="joint1", entity="robot"),
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

  sensor = scene["robot/joint1_pos"]
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
    obj=ObjRef(type="site", name="base_site", entity="robot"),
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

  sensor = scene["robot/base_accel"]

  # Step to make robot fall.
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
    obj=ObjRef(type="joint", name="joint1", entity="robot"),
  )
  jointvel_cfg = BuiltinSensorCfg(
    name="joint1_vel",
    sensor_type="jointvel",
    obj=ObjRef(type="joint", name="joint1", entity="robot"),
  )
  gyro_cfg = BuiltinSensorCfg(
    name="base_gyro",
    sensor_type="gyro",
    obj=ObjRef(type="site", name="base_site", entity="robot"),
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

  jointpos_sensor = scene["robot/joint1_pos"]
  jointvel_sensor = scene["robot/joint1_vel"]
  gyro_sensor = scene["robot/base_gyro"]

  sim.step()

  pos_data = jointpos_sensor.data
  vel_data = jointvel_sensor.data
  gyro_data = gyro_sensor.data

  assert pos_data.shape == (1, 1)
  assert vel_data.shape == (1, 1)
  assert gyro_data.shape == (1, 3)


def test_error_on_invalid_ref():
  """Verify ValueError is raised when ref provided for unsupported sensor type."""
  with pytest.raises(ValueError, match="does not support ref specification"):
    BuiltinSensorCfg(
      name="invalid_sensor",
      sensor_type="jointpos",
      obj=ObjRef(type="joint", name="joint1", entity="robot"),
      ref=ObjRef(type="body", name="base"),
    )


def test_error_on_missing_obj():
  """Verify ValueError is raised when obj is not provided for required sensor type."""
  with pytest.raises(ValueError, match="requires obj with type='joint'"):
    BuiltinSensorCfg(
      name="invalid_sensor",
      sensor_type="jointpos",
    )


def test_error_on_wrong_obj_type_for_site_sensor():
  """Verify ValueError is raised when wrong obj type is used for site sensor."""
  with pytest.raises(ValueError, match="requires obj.type='site'"):
    BuiltinSensorCfg(
      name="invalid_sensor",
      sensor_type="accelerometer",
      obj=ObjRef(type="body", name="base"),
    )


def test_error_on_wrong_obj_type_for_body_sensor():
  """Verify ValueError is raised when wrong obj type is used for body sensor."""
  with pytest.raises(ValueError, match="requires obj.type='body'"):
    BuiltinSensorCfg(
      name="invalid_sensor",
      sensor_type="subtreecom",
      obj=ObjRef(type="site", name="base"),
    )


def test_error_on_wrong_obj_type_for_joint_sensor():
  """Verify ValueError is raised when wrong obj type is used for joint sensor."""
  with pytest.raises(ValueError, match="requires obj.type='joint'"):
    BuiltinSensorCfg(
      name="invalid_sensor",
      sensor_type="jointvel",
      obj=ObjRef(type="body", name="base"),
    )


def test_spatial_frame_sensor_accepts_multiple_types():
  """Verify spatial frame sensors accept body, xbody, geom, site, camera."""
  BuiltinSensorCfg(
    name="frame_sensor_body",
    sensor_type="framepos",
    obj=ObjRef(type="body", name="test"),
  )
  BuiltinSensorCfg(
    name="frame_sensor_xbody",
    sensor_type="framepos",
    obj=ObjRef(type="xbody", name="test"),
  )
  BuiltinSensorCfg(
    name="frame_sensor_geom",
    sensor_type="framepos",
    obj=ObjRef(type="geom", name="test"),
  )
  BuiltinSensorCfg(
    name="frame_sensor_site",
    sensor_type="framepos",
    obj=ObjRef(type="site", name="test"),
  )
  BuiltinSensorCfg(
    name="frame_sensor_camera",
    sensor_type="framepos",
    obj=ObjRef(type="camera", name="test"),
  )


def test_xml_sensors_auto_discovered(robot_with_xml_sensors, device):
  """Verify sensors defined in entity XML are automatically discovered and exposed."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_xml_sensors)
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"robot": entity_cfg},
    sensors=(),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=2, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  joint_sensor = scene["robot/xml_joint_sensor"]
  accel_sensor = scene["robot/xml_accel_sensor"]
  gyro_sensor = scene["robot/xml_gyro_sensor"]

  sim.step()

  joint_data = joint_sensor.data
  accel_data = accel_sensor.data
  gyro_data = gyro_sensor.data

  assert isinstance(joint_data, torch.Tensor)
  assert joint_data.shape == (2, 1)
  assert isinstance(accel_data, torch.Tensor)
  assert accel_data.shape == (2, 3)
  assert isinstance(gyro_data, torch.Tensor)
  assert gyro_data.shape == (2, 3)


def test_builtin_sensor_errors_on_duplicate_name(robot_with_xml_sensors, device):
  """Verify BuiltinSensorCfg throws error when name conflicts with XML sensor."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_with_xml_sensors)
  )

  duplicate_sensor_cfg = BuiltinSensorCfg(
    name="xml_joint_sensor",
    sensor_type="jointpos",
    obj=ObjRef(type="joint", name="joint1", entity="robot"),
  )

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=3.0,
    entities={"robot": entity_cfg},
    sensors=(duplicate_sensor_cfg,),
  )

  with pytest.raises(ValueError, match="defined in both entity XML and scene config"):
    Scene(scene_cfg, device)


def test_cutoff_parameter(articulated_robot_xml, device):
  """Verify cutoff parameter propagates to MuJoCo model."""
  entity_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(articulated_robot_xml)
  )

  sensor_cfg = BuiltinSensorCfg(
    name="joint1_pos",
    sensor_type="jointpos",
    obj=ObjRef(type="joint", name="joint1", entity="robot"),
    cutoff=0.01,
  )

  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=3.0,
    entities={"robot": entity_cfg},
    sensors=(sensor_cfg,),
  )

  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim_cfg = SimulationCfg(njmax=20)
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  sensor = sim.mj_model.sensor("robot/joint1_pos")
  assert sensor.cutoff[0] == 0.01
