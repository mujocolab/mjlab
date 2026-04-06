"""Tests for actuator module."""

import mujoco
import pytest
import torch
from conftest import (
  create_entity_with_actuator,
  get_test_device,
  initialize_entity,
  load_fixture_xml,
)

from mjlab.actuator import (
  BuiltinMotorActuatorCfg,
  BuiltinPositionActuatorCfg,
  BuiltinVelocityActuatorCfg,
  IdealPdActuatorCfg,
  XmlActuatorCfg,
)
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def robot_xml():
  return load_fixture_xml("floating_base_articulated")


def test_builtin_pd_actuator_compute(device, robot_xml):
  """BuiltinPositionActuator writes position targets to ctrl."""
  actuator_cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("joint.*",), stiffness=50.0, damping=5.0
  )
  entity = create_entity_with_actuator(robot_xml, actuator_cfg)
  entity, sim = initialize_entity(entity, device)

  entity.set_joint_position_target(torch.tensor([[0.5, -0.3]], device=device))
  entity.write_data_to_sim()

  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([0.5, -0.3], device=device))


def test_ideal_pd_actuator_compute(device, robot_xml):
  """IdealPdActuator computes torques via explicit PD control."""
  actuator_cfg = IdealPdActuatorCfg(
    target_names_expr=("joint.*",), effort_limit=100.0, stiffness=50.0, damping=5.0
  )
  entity = create_entity_with_actuator(robot_xml, actuator_cfg)
  entity, sim = initialize_entity(entity, device)

  entity.write_joint_state_to_sim(
    position=torch.tensor([[0.0, 0.0]], device=device),
    velocity=torch.tensor([[0.0, 0.0]], device=device),
  )

  entity.set_joint_position_target(torch.tensor([[0.1, -0.1]], device=device))
  entity.set_joint_velocity_target(torch.tensor([[0.0, 0.0]], device=device))
  entity.set_joint_effort_target(torch.tensor([[0.0, 0.0]], device=device))
  entity.write_data_to_sim()

  ctrl = sim.data.ctrl[0]
  assert torch.allclose(ctrl, torch.tensor([5.0, -5.0], device=device))


def test_targets_cleared_on_reset(device, robot_xml):
  """Entity.reset() zeros all targets."""
  actuator_cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("joint.*",), stiffness=50.0, damping=5.0
  )
  entity = create_entity_with_actuator(robot_xml, actuator_cfg)
  entity, sim = initialize_entity(entity, device)

  entity.set_joint_position_target(torch.tensor([[0.5, -0.3]], device=device))
  entity.write_data_to_sim()

  assert not torch.allclose(
    entity.data.joint_pos_target, torch.zeros(1, 2, device=device)
  )

  entity.reset()

  assert torch.allclose(entity.data.joint_pos_target, torch.zeros(1, 2, device=device))
  assert torch.allclose(entity.data.joint_vel_target, torch.zeros(1, 2, device=device))
  assert torch.allclose(
    entity.data.joint_effort_target, torch.zeros(1, 2, device=device)
  )


# ---------------------------------------------------------------------------
# Internal attach prefix tests (issue #709)
# ---------------------------------------------------------------------------


def _make_arm_spec() -> mujoco.MjSpec:
  """Helper: single-joint arm with a motor actuator."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="link")
  body.add_joint(name="elbow", type=mujoco.mjtJoint.mjJNT_HINGE)
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0])
  act = spec.add_actuator(name="motor_elbow", target="elbow")
  act.trntype = mujoco.mjtTrn.mjTRN_JOINT
  act.gainprm[:] = [1] + [0] * 9
  return spec


def _prefixed_entity_spec() -> mujoco.MjSpec:
  """Entity spec that composes a sub-model via internal attach."""
  root = mujoco.MjSpec()
  root.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01])
  frame = root.worldbody.add_frame()
  root.attach(_make_arm_spec(), prefix="arm/", frame=frame)
  return root


def test_builtin_actuator_with_internal_attach_prefix(device):
  """Builtin actuator resolves joints through internal attach prefix."""
  cfg = EntityCfg(
    spec_fn=_prefixed_entity_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(target_names_expr=("elbow",), effort_limit=100.0),
      )
    ),
  )
  entity = Entity(cfg)

  # User-facing names should be stripped.
  assert entity.joint_names == ("elbow",)

  entity, _ = initialize_entity(entity, device)

  assert len(entity.actuators) == 1
  assert entity.actuators[0].target_names == ["elbow"]


def test_xml_actuator_with_internal_attach_prefix(device):
  """XML actuator matches targets through internal attach prefix."""
  cfg = EntityCfg(
    spec_fn=_prefixed_entity_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=("elbow",)),)
    ),
  )
  entity = Entity(cfg)
  entity, sim = initialize_entity(entity, device)

  assert len(entity.actuators) == 1
  assert entity.actuators[0].target_names == ["elbow"]


# ---------------------------------------------------------------------------
# Armature / frictionloss preservation tests
# ---------------------------------------------------------------------------

_XML_WITH_JOINT_PROPERTIES = """\
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom type="box" size="0.2 0.2 0.1" mass="1.0"/>
      <body name="link1">
        <joint name="joint1" type="hinge" axis="0 0 1"
               range="-1.57 1.57" armature="0.5" frictionloss="0.3"
               damping="0.8"/>
        <geom type="box" size="0.1 0.1 0.1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_xml_armature_frictionloss_preserved_by_default(device):
  """Builtin actuator preserves XML armature/frictionloss when not set."""
  cfg = BuiltinMotorActuatorCfg(target_names_expr=("joint1",), effort_limit=100.0)
  entity = create_entity_with_actuator(_XML_WITH_JOINT_PROPERTIES, cfg)
  entity, sim = initialize_entity(entity, device)

  m = sim.mj_model
  dof_id = m.jnt_dofadr[m.joint("joint1").id]
  assert m.dof_armature[dof_id] == pytest.approx(0.5)
  assert m.dof_frictionloss[dof_id] == pytest.approx(0.3)


def test_explicit_armature_frictionloss_overrides_xml(device):
  """Explicit values override XML armature/frictionloss."""
  cfg = BuiltinMotorActuatorCfg(
    target_names_expr=("joint1",),
    effort_limit=100.0,
    armature=1.0,
    frictionloss=0.7,
  )
  entity = create_entity_with_actuator(_XML_WITH_JOINT_PROPERTIES, cfg)
  entity, sim = initialize_entity(entity, device)

  m = sim.mj_model
  dof_id = m.jnt_dofadr[m.joint("joint1").id]
  assert m.dof_armature[dof_id] == pytest.approx(1.0)
  assert m.dof_frictionloss[dof_id] == pytest.approx(0.7)


def test_explicit_zero_overrides_xml(device):
  """Explicit 0.0 overrides XML values (distinguishes None from 0.0)."""
  cfg = BuiltinMotorActuatorCfg(
    target_names_expr=("joint1",),
    effort_limit=100.0,
    armature=0.0,
    frictionloss=0.0,
  )
  entity = create_entity_with_actuator(_XML_WITH_JOINT_PROPERTIES, cfg)
  entity, sim = initialize_entity(entity, device)

  m = sim.mj_model
  dof_id = m.jnt_dofadr[m.joint("joint1").id]
  assert m.dof_armature[dof_id] == pytest.approx(0.0)
  assert m.dof_frictionloss[dof_id] == pytest.approx(0.0)


def test_partial_override_preserves_other_field(device):
  """Setting only armature preserves XML frictionloss, and vice versa."""
  cfg = BuiltinMotorActuatorCfg(
    target_names_expr=("joint1",), effort_limit=100.0, armature=1.0
  )
  entity = create_entity_with_actuator(_XML_WITH_JOINT_PROPERTIES, cfg)
  entity, sim = initialize_entity(entity, device)

  m = sim.mj_model
  dof_id = m.jnt_dofadr[m.joint("joint1").id]
  assert m.dof_armature[dof_id] == pytest.approx(1.0)
  assert m.dof_frictionloss[dof_id] == pytest.approx(0.3)


@pytest.mark.parametrize(
  "actuator_cfg",
  [
    BuiltinPositionActuatorCfg(
      target_names_expr=("joint1",), stiffness=50.0, damping=5.0
    ),
    BuiltinVelocityActuatorCfg(target_names_expr=("joint1",), damping=5.0),
    IdealPdActuatorCfg(
      target_names_expr=("joint1",),
      stiffness=50.0,
      damping=5.0,
      effort_limit=100.0,
    ),
  ],
  ids=["position", "velocity", "ideal_pd"],
)
def test_all_actuator_types_preserve_xml_properties(device, actuator_cfg):
  """All builtin actuator types preserve XML armature/frictionloss."""
  entity = create_entity_with_actuator(_XML_WITH_JOINT_PROPERTIES, actuator_cfg)
  entity, sim = initialize_entity(entity, device)

  m = sim.mj_model
  dof_id = m.jnt_dofadr[m.joint("joint1").id]
  assert m.dof_armature[dof_id] == pytest.approx(0.5)
  assert m.dof_frictionloss[dof_id] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Viscous damping tests
# ---------------------------------------------------------------------------


def test_viscous_damping_default_preserves_xml(device):
  """viscous_damping=None preserves XML joint damping (0.8)."""
  cfg = BuiltinMotorActuatorCfg(target_names_expr=("joint1",), effort_limit=100.0)
  entity = create_entity_with_actuator(_XML_WITH_JOINT_PROPERTIES, cfg)
  entity, sim = initialize_entity(entity, device)

  m = sim.mj_model
  dof_id = m.jnt_dofadr[m.joint("joint1").id]
  assert m.dof_damping[dof_id] == pytest.approx(0.8)


def test_viscous_damping_explicit_overrides_xml(device):
  """Explicit viscous_damping overrides XML damping."""
  cfg = BuiltinMotorActuatorCfg(
    target_names_expr=("joint1",), effort_limit=100.0, viscous_damping=2.5
  )
  entity = create_entity_with_actuator(_XML_WITH_JOINT_PROPERTIES, cfg)
  entity, sim = initialize_entity(entity, device)

  m = sim.mj_model
  dof_id = m.jnt_dofadr[m.joint("joint1").id]
  assert m.dof_damping[dof_id] == pytest.approx(2.5)


def test_viscous_damping_explicit_zero_overrides_xml(device):
  """Explicit 0.0 overrides XML damping (distinguishes None from 0.0)."""
  cfg = BuiltinMotorActuatorCfg(
    target_names_expr=("joint1",), effort_limit=100.0, viscous_damping=0.0
  )
  entity = create_entity_with_actuator(_XML_WITH_JOINT_PROPERTIES, cfg)
  entity, sim = initialize_entity(entity, device)

  m = sim.mj_model
  dof_id = m.jnt_dofadr[m.joint("joint1").id]
  assert m.dof_damping[dof_id] == pytest.approx(0.0)


def test_viscous_damping_produces_passive_force(device, robot_xml):
  """Nonzero viscous_damping produces a dissipative passive force."""
  damping = 3.0
  actuator_cfg = BuiltinMotorActuatorCfg(
    target_names_expr=("joint.*",), effort_limit=100.0, viscous_damping=damping
  )
  entity = create_entity_with_actuator(robot_xml, actuator_cfg)
  entity, sim = initialize_entity(entity, device)

  vel = torch.tensor([[1.0, -2.0]], device=device)
  entity.write_joint_state_to_sim(
    position=torch.zeros(1, 2, device=device),
    velocity=vel,
  )
  entity.write_data_to_sim()
  sim.step()

  qfrc_damper = sim.data.qfrc_damper[0]
  # f = -b * v
  assert qfrc_damper[6].item() == pytest.approx(-damping * 1.0, abs=1e-4)
  assert qfrc_damper[7].item() == pytest.approx(-damping * -2.0, abs=1e-4)
