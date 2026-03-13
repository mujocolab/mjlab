"""Tests for EntityData."""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from conftest import (
  create_entity_with_actuator,
  get_test_device,
  initialize_entity,
  load_fixture_xml,
)

from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

FLOATING_BASE_XML = """
<mujoco>
  <worldbody>
    <body name="object" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="object_geom" type="box" size="0.1 0.1 0.1" rgba="0.3 0.3 0.8 1" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

FLOATING_BASE_COM_OFFSET_XML = """
<mujoco>
  <worldbody>
    <body name="object" pos="0 0 1">
      <freejoint name="free_joint"/>
      <inertial pos="0.1 0.05 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def device():
  """Test device fixture."""
  return get_test_device()


def create_floating_base_entity():
  """Create a floating-base entity."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(FLOATING_BASE_XML))
  return Entity(cfg)


def initialize_entity_with_sim(entity, device, num_envs=1):
  """Initialize an entity with a simulation."""
  model = entity.compile()
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


def test_root_velocity_world_frame_roundtrip(device):
  """Test reading and writing root velocity is a no-op (world frame).

  Verifies that the API is consistent: if you write a velocity, read it
  back, and write it again, you get the same result. This ensures no
  unintended transformations happen in the read/write cycle.
  """
  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  pose = torch.tensor([0.0, 0.0, 1.0, 0.6, 0.2, 0.3, 0.7141], device=device).unsqueeze(
    0
  )
  entity.write_root_link_pose_to_sim(pose)

  vel_w = torch.tensor([1.0, 0.5, 0.0, 0.0, 0.3, 0.1], device=device).unsqueeze(0)
  entity.write_root_link_velocity_to_sim(vel_w)
  sim.forward()

  vel_w_read = entity.data.root_link_vel_w.clone()
  assert torch.allclose(vel_w_read, vel_w, atol=1e-4)

  entity.write_root_link_velocity_to_sim(vel_w_read)
  sim.forward()
  vel_w_after = entity.data.root_link_vel_w

  assert torch.allclose(vel_w_after, vel_w_read, atol=1e-4)


def test_root_velocity_frame_conversion(device):
  """Test angular velocity converts from world to body frame internally.

  The API accepts angular velocity in world frame, but MuJoCo's qvel
  stores it in body frame. This test verifies the conversion happens
  correctly by checking qvel directly.
  """
  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  quat_w = torch.tensor([0.6, 0.2, 0.3, 0.7141], device=device).unsqueeze(0)
  pose = torch.cat([torch.zeros(1, 3, device=device), quat_w], dim=-1)
  entity.write_root_link_pose_to_sim(pose)

  lin_vel_w = torch.tensor([1.0, 0.5, 0.2], device=device).unsqueeze(0)
  ang_vel_w = torch.tensor([0.1, 0.2, 0.3], device=device).unsqueeze(0)
  vel_w = torch.cat([lin_vel_w, ang_vel_w], dim=-1)
  entity.write_root_link_velocity_to_sim(vel_w)

  v_slice = entity.data.indexing.free_joint_v_adr
  qvel = sim.data.qvel[:, v_slice]

  assert torch.allclose(qvel[:, :3], lin_vel_w, atol=1e-5)

  expected_ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)
  assert torch.allclose(qvel[:, 3:], expected_ang_vel_b, atol=1e-5)


def test_write_velocity_uses_qpos_not_xquat(device):
  """Test write_root_velocity uses qpos (not stale xquat).

  Writing pose then velocity without forward() must work. This would fail
  if write_root_velocity used xquat (stale) instead of qpos (current).
  """
  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  initial_pose = torch.tensor(
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], device=device
  ).unsqueeze(0)
  entity.write_root_link_pose_to_sim(initial_pose)
  sim.forward()  # xquat now has identity orientation.

  # Write different orientation without forward() - xquat stale, qpos current.
  new_pose = torch.tensor(
    [0.0, 0.0, 1.0, 0.707, 0.0, 0.707, 0.0], device=device
  ).unsqueeze(0)
  vel_w = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device).unsqueeze(0)

  entity.write_root_link_pose_to_sim(new_pose)
  entity.write_root_link_velocity_to_sim(vel_w)

  sim.forward()
  vel_w_read = entity.data.root_link_vel_w

  assert torch.allclose(vel_w_read, vel_w, atol=1e-4)


def test_write_root_com_velocity(device):
  """COM velocity write must produce the same qvel as manual conversion."""
  num_envs = 4
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(FLOATING_BASE_COM_OFFSET_XML)
  )
  entity = Entity(cfg)
  entity, sim = initialize_entity_with_sim(entity, device, num_envs=num_envs)

  # Give each env a different non-identity orientation.
  pose = torch.zeros(num_envs, 7, device=device)
  pose[:, 2] = 1.0
  quats = torch.tensor(
    [
      [1.0, 0.0, 0.0, 0.0],
      [0.707, 0.707, 0.0, 0.0],
      [0.5, 0.5, 0.5, 0.5],
      [0.0, 0.707, 0.0, 0.707],
    ],
    device=device,
  )
  pose[:, 3:7] = quats
  entity.write_root_link_pose_to_sim(pose)

  com_vel = torch.tensor(
    [[1.0, 0.5, -0.2, 0.1, 0.2, 0.3], [-0.5, 1.0, 0.0, -0.1, 0.0, 0.4]],
    device=device,
  )
  env_ids = torch.tensor([1, 3], device=device)

  # Write COM velocity via the API.
  entity.write_root_com_velocity_to_sim(com_vel, env_ids=env_ids)
  qvel_from_api = sim.data.qvel.clone()

  # Manually convert COM velocity to link velocity and write that instead.
  com_offset_b = sim.model.body_ipos[:, entity.indexing.root_body_id]
  com_offset_w = quat_apply(quats[env_ids], com_offset_b[env_ids])
  lin_vel_link = com_vel[:, :3] - torch.cross(com_vel[:, 3:], com_offset_w, dim=-1)
  link_vel = torch.cat([lin_vel_link, com_vel[:, 3:]], dim=-1)
  entity.write_root_link_velocity_to_sim(link_vel, env_ids=env_ids)
  qvel_from_manual = sim.data.qvel.clone()

  assert torch.allclose(qvel_from_api, qvel_from_manual, atol=1e-5)

  # Untouched envs should have zero velocity.
  v_adr = entity.indexing.free_joint_v_adr
  assert torch.all(qvel_from_api[0, v_adr] == 0.0)
  assert torch.all(qvel_from_api[2, v_adr] == 0.0)


def test_read_requires_forward_to_be_current(device):
  """Test read properties are stale until forward() is called.

  Demonstrates why event order matters and why forward() is needed
  between writes and reads.
  """
  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  sim.forward()
  initial_pose = entity.data.root_link_pose_w.clone()

  new_pose = torch.tensor(
    [1.0, 2.0, 3.0, 0.707, 0.0, 0.707, 0.0], device=device
  ).unsqueeze(0)
  entity.write_root_link_pose_to_sim(new_pose)

  stale_pose = entity.data.root_link_pose_w
  assert torch.allclose(stale_pose, initial_pose, atol=1e-5)

  sim.forward()
  current_pose = entity.data.root_link_pose_w
  assert torch.allclose(current_pose, new_pose, atol=1e-4)
  assert not torch.allclose(current_pose, initial_pose, atol=1e-4)


@pytest.mark.parametrize(
  "property_name,expected_shape",
  [
    # Root properties.
    ("root_link_pose_w", (1, 7)),
    ("root_link_pos_w", (1, 3)),
    ("root_link_quat_w", (1, 4)),
    ("root_link_vel_w", (1, 6)),
    ("root_link_lin_vel_w", (1, 3)),
    ("root_link_ang_vel_w", (1, 3)),
    ("root_com_pose_w", (1, 7)),
    ("root_com_pos_w", (1, 3)),
    ("root_com_quat_w", (1, 4)),
    ("root_com_vel_w", (1, 6)),
    ("root_com_lin_vel_w", (1, 3)),
    ("root_com_ang_vel_w", (1, 3)),
    # Body properties (we only have 1 body in this test).
    ("body_link_pose_w", (1, 1, 7)),
    ("body_link_pos_w", (1, 1, 3)),
    ("body_link_quat_w", (1, 1, 4)),
    ("body_link_vel_w", (1, 1, 6)),
    ("body_com_pose_w", (1, 1, 7)),
    ("body_com_vel_w", (1, 1, 6)),
  ],
)
def test_entity_data_properties_accessible(device, property_name, expected_shape):
  """Test that all EntityData properties can be accessed without errors."""
  entity = create_floating_base_entity()
  entity, sim = initialize_entity_with_sim(entity, device)

  sim.forward()

  value = getattr(entity.data, property_name)
  assert value.shape == expected_shape, f"{property_name} has unexpected shape"


def test_entity_data_reset_clears_all_targets(device):
  """Test that EntityData.clear_state() zeros out all target buffers."""
  xml_path = Path(__file__).parent / "fixtures" / "tendon_finger.xml"

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file(str(xml_path)),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=("finger_tendon",),
          transmission_type=TransmissionType.TENDON,
          effort_limit=10.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  entity, sim = initialize_entity_with_sim(entity, device, num_envs=4)

  entity.data.joint_pos_target[:] = 1.0
  entity.data.joint_vel_target[:] = 2.0
  entity.data.joint_effort_target[:] = 3.0
  entity.data.tendon_len_target[:] = 4.0
  entity.data.tendon_vel_target[:] = 5.0
  entity.data.tendon_effort_target[:] = 6.0

  entity.data.clear_state()

  assert torch.all(entity.data.joint_pos_target == 0.0)
  assert torch.all(entity.data.joint_vel_target == 0.0)
  assert torch.all(entity.data.joint_effort_target == 0.0)
  assert torch.all(entity.data.tendon_len_target == 0.0)
  assert torch.all(entity.data.tendon_vel_target == 0.0)
  assert torch.all(entity.data.tendon_effort_target == 0.0)


def test_entity_data_reset_partial_envs(device):
  """Test that EntityData.clear_state() can reset specific environments."""
  xml_path = Path(__file__).parent / "fixtures" / "tendon_finger.xml"

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_file(str(xml_path)),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=("finger_tendon",),
          transmission_type=TransmissionType.TENDON,
          effort_limit=10.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  entity, sim = initialize_entity_with_sim(entity, device, num_envs=4)

  entity.data.tendon_len_target[:] = 7.0
  entity.data.tendon_vel_target[:] = 8.0
  entity.data.tendon_effort_target[:] = 9.0

  env_ids = torch.tensor([1, 3], device=device)
  entity.data.clear_state(env_ids)

  assert torch.all(entity.data.tendon_len_target[0] == 7.0)
  assert torch.all(entity.data.tendon_len_target[1] == 0.0)
  assert torch.all(entity.data.tendon_len_target[2] == 7.0)
  assert torch.all(entity.data.tendon_len_target[3] == 0.0)

  assert torch.all(entity.data.tendon_vel_target[0] == 8.0)
  assert torch.all(entity.data.tendon_vel_target[1] == 0.0)
  assert torch.all(entity.data.tendon_vel_target[2] == 8.0)
  assert torch.all(entity.data.tendon_vel_target[3] == 0.0)

  assert torch.all(entity.data.tendon_effort_target[0] == 9.0)
  assert torch.all(entity.data.tendon_effort_target[1] == 0.0)
  assert torch.all(entity.data.tendon_effort_target[2] == 9.0)
  assert torch.all(entity.data.tendon_effort_target[3] == 0.0)


# Joint limits tests.

MIXED_LIMITS_XML = """
<mujoco>
  <worldbody>
    <body name="base">
      <body name="cart" pos="0 0 1">
        <joint name="slider" type="slide" axis="1 0 0" limited="true" range="-2 2"/>
        <geom type="box" size="0.1 0.1 0.1" mass="1"/>
        <body name="pole">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

ALL_LIMITED_XML = """
<mujoco>
  <worldbody>
    <body name="base">
      <body name="link1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="0 0 1" limited="true" range="-1.5 1.5"/>
        <geom type="box" size="0.05 0.05 0.1" mass="1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="j2" type="hinge" axis="0 1 0" limited="true" range="-1 1"/>
          <geom type="box" size="0.05 0.05 0.1" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

ALL_UNLIMITED_XML = """
<mujoco>
  <worldbody>
    <body name="base">
      <body name="link1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.05 0.05 0.1" mass="1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="box" size="0.05 0.05 0.1" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_soft_limits_unlimited_joints_are_infinite(device):
  """Unlimited joints must have [-inf, inf] soft limits, not [0, 0]."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(MIXED_LIMITS_XML))
  entity = Entity(cfg)
  entity, _ = initialize_entity_with_sim(entity, device)

  limits = entity.data.soft_joint_pos_limits[0]
  # slider (index 0) is limited: finite limits.
  assert torch.isfinite(limits[0, 0]) and torch.isfinite(limits[0, 1])
  assert limits[0, 0] == -2.0
  assert limits[0, 1] == 2.0
  # hinge (index 1) is unlimited: infinite limits.
  assert limits[1, 0] == float("-inf")
  assert limits[1, 1] == float("inf")


def test_soft_limits_all_limited(device):
  """All-limited joints should have finite soft limits."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ALL_LIMITED_XML))
  entity = Entity(cfg)
  entity, _ = initialize_entity_with_sim(entity, device)

  limits = entity.data.soft_joint_pos_limits[0]
  for j in range(limits.shape[0]):
    assert torch.isfinite(limits[j, 0]) and torch.isfinite(limits[j, 1])


def test_soft_limits_all_unlimited(device):
  """All-unlimited joints should have infinite soft limits."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ALL_UNLIMITED_XML))
  entity = Entity(cfg)
  entity, _ = initialize_entity_with_sim(entity, device)

  limits = entity.data.soft_joint_pos_limits[0]
  for j in range(limits.shape[0]):
    assert limits[j, 0] == float("-inf")
    assert limits[j, 1] == float("inf")


def test_joint_pos_limits_match_soft_limits_for_unlimited(device):
  """joint_pos_limits and default_joint_pos_limits should also be inf for unlimited."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(MIXED_LIMITS_XML))
  entity = Entity(cfg)
  entity, _ = initialize_entity_with_sim(entity, device)

  # Hinge is joint index 1, unlimited.
  for limits_tensor in (
    entity.data.joint_pos_limits,
    entity.data.default_joint_pos_limits,
    entity.data.soft_joint_pos_limits,
  ):
    assert limits_tensor[0, 1, 0] == float("-inf")
    assert limits_tensor[0, 1, 1] == float("inf")


def test_reset_joints_by_offset_respects_unlimited(device):
  """reset_joints_by_offset must not clamp unlimited joints to [0, 0]."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(MIXED_LIMITS_XML),
    init_state=EntityCfg.InitialStateCfg(
      joint_pos={"slider": 0.0, "hinge": math.pi},
      joint_vel={".*": 0.0},
    ),
  )
  entity = Entity(cfg)
  entity, _ = initialize_entity_with_sim(entity, device)

  # Simulate what reset_joints_by_offset does.
  default_pos = entity.data.default_joint_pos
  soft_limits = entity.data.soft_joint_pos_limits
  joint_pos = default_pos.clone()
  joint_pos = joint_pos.clamp_(soft_limits[..., 0], soft_limits[..., 1])

  # Hinge should stay at pi, not get clamped to 0.
  assert torch.allclose(
    joint_pos[0, 1], torch.tensor(math.pi, device=device), atol=1e-5
  )


# Generalized force accessor tests.

TWO_LINK_ARM_XML = """
<mujoco>
  <worldbody>
    <body name="base">
      <body name="upper" pos="0 0 0.5">
        <joint name="shoulder" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.03" mass="1.0"/>
        <body name="lower" pos="0 0 0.4">
          <joint name="elbow" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.02" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_qfrc_actuator_slices_joint_dofs_only(device):
  """qfrc_actuator should expose only articulated joint DoFs."""
  xml = load_fixture_xml("floating_base_articulated")
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(xml))
  entity = Entity(cfg)
  entity, sim = initialize_entity_with_sim(entity, device)

  joint_v_adr = entity.indexing.joint_v_adr
  free_v_adr = entity.indexing.free_joint_v_adr
  nv = sim.data.qvel.shape[1]

  values = torch.arange(1, nv + 1, device=device, dtype=torch.float32)
  sim.data.qfrc_actuator[0] = values
  expected = values[joint_v_adr]
  actual = entity.data.qfrc_actuator[0]
  assert torch.equal(actual, expected)

  assert free_v_adr.numel() > 0
  assert entity.data.qfrc_actuator.shape[-1] == joint_v_adr.numel()
  assert entity.data.qfrc_actuator.shape[-1] != free_v_adr.numel()


def test_qfrc_actuator_matches_motor_command_with_gear(device):
  """qfrc_actuator should equal actuator_force projected through joint gear."""
  xml = load_fixture_xml("fixed_base_articulated")
  entity = create_entity_with_actuator(
    xml,
    BuiltinMotorActuatorCfg(
      target_names_expr=("joint.*",),
      effort_limit=100.0,
      gear=3.0,
    ),
  )
  entity, sim = initialize_entity(entity, device)

  commanded = torch.tensor([[2.0, -1.0]], device=device)
  entity.set_joint_effort_target(commanded)
  entity.write_data_to_sim()
  sim.forward()

  assert torch.allclose(entity.data.actuator_force, commanded, atol=1e-6)
  assert torch.allclose(entity.data.qfrc_actuator, 3.0 * commanded, atol=1e-6)


def test_qfrc_external_matches_mj_applyFT(device):
  """qfrc_external should match J^T * xfrc_applied computed by CPU MuJoCo."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(TWO_LINK_ARM_XML))
  entity = Entity(cfg)
  entity, sim = initialize_entity_with_sim(entity, device)

  entity.write_joint_state_to_sim(
    torch.tensor([[0.5, -0.3]], device=device),
    torch.tensor([[0.0, 0.0]], device=device),
  )
  sim.forward()

  lower_body_id = int(entity.indexing.body_ids[1].item())
  force = torch.tensor([1.0, 0.0, -2.0], device=device)
  torque = torch.tensor([0.0, 0.5, 0.0], device=device)
  sim.data.xfrc_applied[0, lower_body_id, 0:3] = force
  sim.data.xfrc_applied[0, lower_body_id, 3:6] = torque
  sim.forward()

  mjd = mujoco.MjData(sim.mj_model)
  mjd.qpos[:] = sim.data.qpos[0].cpu().numpy()
  mjd.qvel[:] = sim.data.qvel[0].cpu().numpy()
  mujoco.mj_forward(sim.mj_model, mjd)

  qfrc_reference = np.zeros(sim.mj_model.nv)
  mujoco.mj_applyFT(
    sim.mj_model,
    mjd,
    force.cpu().numpy().astype(np.float64),
    torque.cpu().numpy().astype(np.float64),
    mjd.xipos[lower_body_id],
    lower_body_id,
    qfrc_reference,
  )
  expected = torch.from_numpy(qfrc_reference).to(device=device, dtype=torch.float32)

  joint_v_adr = entity.indexing.joint_v_adr
  assert torch.allclose(
    entity.data.qfrc_external, expected[joint_v_adr].unsqueeze(0), atol=1e-5
  )
