"""Tests for builtin sensors."""

from pathlib import Path
from typing import cast

import mujoco
import pytest
import torch
import warp as wp

from mjlab.entity.entity import Entity
from mjlab.sensor import (
  BuiltinContactSensor,
  BuiltinContactSensorCfg,
  BuiltinSensorCfg,
)
from mjlab.sensor.base import Sensor
from mjlab.sim.sim import Simulation, SimulationCfg

wp.config.quiet = True

ASSET_ROOT = Path(__file__).parent.parent / "src/mjlab/asset_zoo/robots"
G1_XML_PATH = ASSET_ROOT / "unitree_g1/xmls/g1.xml"
GROUND_GEOM_NAME = "ground_plane"


def get_test_device() -> str:
  """Get device for testing, preferring CUDA if available."""
  return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def device() -> str:
  """Test device fixture."""
  return get_test_device()


def build_g1_spec() -> mujoco.MjSpec:
  """Load G1 robot as an editable specification."""
  return mujoco.MjSpec.from_file(str(G1_XML_PATH))


def add_ground_plane(spec: mujoco.MjSpec) -> str:
  """Append a ground plane to the provided spec."""
  spec.worldbody.add_geom(
    name=GROUND_GEOM_NAME,
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[10, 10, 0.1],
  )
  return GROUND_GEOM_NAME


def initialize_sensor(
  cfg: BuiltinSensorCfg | BuiltinContactSensorCfg,
  *,
  device: str,
  num_envs: int = 1,
  spec: mujoco.MjSpec | None = None,
  entities: dict[str, Entity] | None = None,
) -> tuple[Sensor, Simulation, mujoco.MjModel]:
  """Compile the spec, initialize simulation, and return sensor, sim, and model."""
  spec = spec or build_g1_spec()
  sensor = cfg.build()
  entity_map: dict[str, Entity] = entities if entities is not None else {}
  sensor.edit_spec(spec, entity_map)
  model = spec.compile()

  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  sensor.initialize(model, sim.model, sim.data, device)
  return sensor, sim, model


@pytest.mark.parametrize(
  ("cfg", "expected_enum", "num_envs", "expected_shape"),
  [
    pytest.param(
      BuiltinSensorCfg(
        name="pelvis_gyro",
        entity_name="",
        sensor_type="gyro",
        objtype="site",
        objname="imu_in_pelvis",
      ),
      mujoco.mjtSensor.mjSENS_GYRO,
      2,
      (2, 3),
      id="gyro",
    ),
    pytest.param(
      BuiltinSensorCfg(
        name="pelvis_accel",
        entity_name="",
        sensor_type="accelerometer",
        objtype="site",
        objname="imu_in_pelvis",
      ),
      mujoco.mjtSensor.mjSENS_ACCELEROMETER,
      1,
      (1, 3),
      id="accelerometer",
    ),
    pytest.param(
      BuiltinSensorCfg(
        name="pelvis_velocity",
        entity_name="",
        sensor_type="framelinvel",
        objtype="site",
        objname="imu_in_pelvis",
      ),
      mujoco.mjtSensor.mjSENS_FRAMELINVEL,
      1,
      (1, 3),
      id="framelinvel",
    ),
  ],
)
def test_builtin_sensor_registration(
  device: str,
  cfg: BuiltinSensorCfg,
  expected_enum: int,
  num_envs: int,
  expected_shape: tuple[int, ...],
) -> None:
  """Ensure builtin sensors resolve to the correct MuJoCo entry and tensor layout."""
  spec = build_g1_spec()
  sensor, _, model = initialize_sensor(cfg, device=device, num_envs=num_envs, spec=spec)
  mj_sensor = model.sensor(cfg.name)
  site_id = model.site(cfg.objname).id

  assert mj_sensor.type[0] == expected_enum
  assert mj_sensor.objtype[0] == mujoco.mjtObj.mjOBJ_SITE
  assert mj_sensor.objid[0] == site_id

  data = cast(torch.Tensor, sensor.data)
  assert data.shape == expected_shape
  assert data.device.type == torch.device(device).type


@pytest.mark.parametrize(
  ("entity_name", "objname", "site_name", "expect_success"),
  [
    pytest.param("robot", "imu", "robot/imu", True, id="prefix-applied"),
    pytest.param("", "world/imu", "world/imu", True, id="entityless"),
    pytest.param("robot", "imu", "imu", False, id="missing-prefix"),
  ],
)
def test_entity_prefixing_behavior(
  entity_name: str,
  objname: str,
  site_name: str,
  expect_success: bool,
) -> None:
  """Confirm entity prefixing logic cooperates with existing site names."""
  cfg = BuiltinSensorCfg(
    name="sensor",
    entity_name=entity_name,
    sensor_type="gyro",
    objtype="site",
    objname=objname,
  )

  spec = mujoco.MjSpec()
  body_name = f"{entity_name}/base" if entity_name else "base"
  body = spec.worldbody.add_body(name=body_name)
  body.add_site(name=site_name)

  cfg.edit_spec(spec)
  if expect_success:
    model = spec.compile()
    assert model.nsensor == 1
  else:
    with pytest.raises(Exception):  # noqa: B017
      spec.compile()


def test_contact_sensor_registration_and_shortcuts(device: str) -> None:
  """Register contact sensor and ensure shortcut properties mirror structured data.

  We seed MuJoCo's sensordata to avoid sim dynamics and check that the count/force
  accessors are exactly views over the structured buffer.
  """
  spec = build_g1_spec()
  ground = add_ground_plane(spec)
  cfg = BuiltinContactSensorCfg(
    name="foot_contact",
    entity_name="",
    geom1="left_foot1_collision",
    geom2=ground,
    data=("found", "force"),
    num=2,
  )

  sensor, sim, model = initialize_sensor(cfg, device=device, num_envs=2, spec=spec)
  assert isinstance(sensor, BuiltinContactSensor)
  contact_sensor = sensor

  mj_sensor = model.sensor("foot_contact")
  geom_id = model.geom("left_foot1_collision").id
  ground_id = model.geom(ground).id
  per_contact = 1 + 3

  assert mj_sensor.type[0] == mujoco.mjtSensor.mjSENS_CONTACT
  assert mj_sensor.objtype[0] == mujoco.mjtObj.mjOBJ_GEOM
  assert mj_sensor.objid[0] == geom_id
  assert mj_sensor.reftype[0] == mujoco.mjtObj.mjOBJ_GEOM
  assert mj_sensor.refid[0] == ground_id
  assert mj_sensor.dim[0] == cfg.num * per_contact
  assert contact_sensor.data.force.shape == (2, cfg.num, 3)

  start = mj_sensor.adr[0]
  dim = mj_sensor.dim[0]
  raw = sim.data.sensordata
  raw.zero_()
  # fmt: off
  raw[:, start : start + dim] = raw.new_tensor(
    [
      [5, 1, 2, 3, 0, 0, 0, 0],
      [2, 4, 5, 6, 0, 0, 0, 0],
    ]
  )
  # fmt: on

  assert torch.equal(contact_sensor.count, raw.new_tensor([5, 2]))
  assert torch.equal(contact_sensor.count, contact_sensor.data.count)
  assert torch.equal(contact_sensor.force, contact_sensor.data.force)
  assert torch.allclose(contact_sensor.force[0, 0], raw.new_tensor([1, 2, 3]))
  assert torch.allclose(contact_sensor.force[1, 0], raw.new_tensor([4, 5, 6]))


@pytest.mark.parametrize(
  ("case", "expected_exception", "match"),
  [
    pytest.param(
      "missing_site",
      (mujoco.FatalError, ValueError),
      "does_not_exist",
      id="missing-site",
    ),
    pytest.param(
      "conflicting_primary",
      ValueError,
      "exactly one",
      id="conflicting-primary",
    ),
    pytest.param(
      "conflicting_secondary",
      ValueError,
      "at most one",
      id="conflicting-secondary",
    ),
  ],
)
def test_sensor_configuration_errors(
  case: str,
  expected_exception: type[BaseException] | tuple[type[BaseException], ...],
  match: str,
) -> None:
  """Surface misconfigurations before running simulations.

  Covers the main integration failure modes we expect to guard in mjlab.
  """
  if case == "missing_site":
    spec = build_g1_spec()
    cfg = BuiltinSensorCfg(
      name="invalid_sensor",
      entity_name="",
      sensor_type="gyro",
      objtype="site",
      objname="does_not_exist",
    )

    sensor = cfg.build()
    sensor.edit_spec(spec, {})
    with pytest.raises(expected_exception, match=match):
      spec.compile()
    return

  if case == "conflicting_primary":
    cfg = BuiltinContactSensorCfg(
      name="invalid_contact",
      entity_name="",
      geom1="geom_a",
      subtree1="pelvis",
    )
  else:
    cfg = BuiltinContactSensorCfg(
      name="invalid_contact_secondary",
      entity_name="",
      geom1="left_foot1_collision",
      geom2="ground",
      subtree2="pelvis",
    )

  with pytest.raises(expected_exception, match=match):
    cfg.edit_spec(mujoco.MjSpec())
