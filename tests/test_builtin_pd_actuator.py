"""Tests for BuiltinPdActuator.

Covers the unique surface of the actuator: paired <position>/<velocity>
elements per target, joint/tendon-level actfrcrange sum-clamp, DR for both
gains and effort limits, delay synchronization, and the ordering invariant
that DR depends on.
"""

from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import (
  create_entity_with_actuator,
  get_test_device,
  initialize_entity,
  load_fixture_xml,
)

from mjlab.actuator import BuiltinPdActuator, BuiltinPdActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg

ROBOT_XML = load_fixture_xml("floating_base_articulated")
KP = 100.0
KD = 10.0


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def _make_entity(
  *,
  effort_limit: float | None = 50.0,
  armature: float | None = None,
  delay_max_lag: int = 0,
  delay_min_lag: int = 0,
  delay_hold_prob: float = 0.0,
) -> Entity:
  cfg = BuiltinPdActuatorCfg(
    target_names_expr=("joint.*",),
    stiffness=KP,
    damping=KD,
    effort_limit=effort_limit,
    armature=armature,
    delay_min_lag=delay_min_lag,
    delay_max_lag=delay_max_lag,
    delay_hold_prob=delay_hold_prob,
  )
  return create_entity_with_actuator(ROBOT_XML, cfg)


def _at_rest_with_targets(
  entity: Entity,
  sim,
  device: str,
  pos_target: torch.Tensor,
  vel_target: torch.Tensor,
) -> None:
  entity.write_joint_state_to_sim(
    position=torch.zeros(1, 2, device=device),
    velocity=torch.zeros(1, 2, device=device),
  )
  entity.set_joint_position_target(pos_target)
  entity.set_joint_velocity_target(vel_target)
  entity.set_joint_effort_target(torch.zeros(1, 2, device=device))
  entity.write_data_to_sim()
  sim.forward()


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_two_ctrls_per_target_with_pos_then_vel_layout(device):
  """Each target gets one <position> + one <velocity>, in halves."""
  entity, sim = initialize_entity(_make_entity(), device)
  act = entity.actuators[0]
  assert isinstance(act, BuiltinPdActuator)

  n = act.num_targets
  assert n == len(act.target_names) == 2
  assert len(act.ctrl_ids) == 2 * n
  assert len(act.global_ctrl_ids) == 2 * n

  names = [sim.mj_model.actuator(i).name for i in act.global_ctrl_ids.tolist()]
  assert names[:n] == [f"{name}_pd_pos" for name in act.target_names]
  assert names[n:] == [f"{name}_pd_vel" for name in act.target_names]


def test_site_transmission_rejected():
  with pytest.raises(ValueError, match="SITE"):
    BuiltinPdActuatorCfg(
      target_names_expr=("x",),
      stiffness=1.0,
      damping=1.0,
      transmission_type=TransmissionType.SITE,
    )


def test_armature_applied_once(device):
  """Joint armature must come from the position element only; double-applying
  would silently double dof_armature."""
  _, sim = initialize_entity(_make_entity(armature=0.7), device)
  m = sim.mj_model
  for jname in ("joint1", "joint2"):
    dof_id = m.jnt_dofadr[m.joint(jname).id]
    assert m.dof_armature[dof_id] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Force computation
# ---------------------------------------------------------------------------


def test_position_only(device):
  """Zero vel target: qfrc = kp * pos_target."""
  entity, sim = initialize_entity(_make_entity(effort_limit=None), device)
  pos = torch.tensor([[0.1, -0.05]], device=device)
  _at_rest_with_targets(entity, sim, device, pos, torch.zeros(1, 2, device=device))
  v_adr = entity.indexing.joint_v_adr
  assert torch.allclose(sim.data.qfrc_actuator[0, v_adr], KP * pos[0], atol=1e-4)


def test_velocity_only(device):
  """Zero pos target, joint at rest: qfrc = kd * vel_target."""
  entity, sim = initialize_entity(_make_entity(effort_limit=None), device)
  vel = torch.tensor([[0.3, -0.2]], device=device)
  _at_rest_with_targets(entity, sim, device, torch.zeros(1, 2, device=device), vel)
  v_adr = entity.indexing.joint_v_adr
  assert torch.allclose(sim.data.qfrc_actuator[0, v_adr], KD * vel[0], atol=1e-4)


def test_pd_superposition(device):
  """Both targets nonzero: qfrc = kp * pos_target + kd * vel_target."""
  entity, sim = initialize_entity(_make_entity(effort_limit=None), device)
  pos = torch.tensor([[0.1, -0.05]], device=device)
  vel = torch.tensor([[0.2, -0.1]], device=device)
  _at_rest_with_targets(entity, sim, device, pos, vel)
  v_adr = entity.indexing.joint_v_adr
  expected = KP * pos[0] + KD * vel[0]
  assert torch.allclose(sim.data.qfrc_actuator[0, v_adr], expected, atol=1e-4)


def test_actfrcrange_sum_clamp(device):
  """A pos error big enough to make kp*err exceed effort_limit must be
  clamped at the joint, not allowed to ride through the unbounded element."""
  entity, sim = initialize_entity(_make_entity(effort_limit=5.0), device)
  # kp * 10.0 = 1000, well over the 5.0 clamp.
  pos = torch.tensor([[10.0, 0.0]], device=device)
  _at_rest_with_targets(entity, sim, device, pos, torch.zeros(1, 2, device=device))
  v_adr = entity.indexing.joint_v_adr
  qfrc = sim.data.qfrc_actuator[0, v_adr]
  assert qfrc[0].item() == pytest.approx(5.0, abs=1e-4)
  assert qfrc[1].item() == pytest.approx(0.0, abs=1e-4)


def test_effort_limit_none_leaves_joint_unlimited(device):
  """effort_limit=None: jnt_actfrclimited stays 0 on the targeted joints."""
  _, sim = initialize_entity(_make_entity(effort_limit=None), device)
  m = sim.mj_model
  for jname in ("joint1", "joint2"):
    jid = m.joint(jname).id
    assert m.jnt_actfrclimited[jid] == 0


def test_actuator_forcerange_not_set(device):
  """We deliberately leave per-element forcerange unset; the limit lives on
  the joint. Inspection of actuator_force[i] thus shows the unclamped value."""
  entity, sim = initialize_entity(_make_entity(effort_limit=5.0), device)
  m = sim.mj_model
  for ctrl_id in entity.actuators[0].global_ctrl_ids.tolist():
    assert m.actuator_forcelimited[ctrl_id] == 0


# ---------------------------------------------------------------------------
# Delay synchronization
# ---------------------------------------------------------------------------


def test_delay_syncs_pos_and_vel(device):
  """The shared delay buffer must lag pos and vel together."""
  entity, sim = initialize_entity(
    _make_entity(effort_limit=None, delay_min_lag=2, delay_max_lag=2),
    device,
  )
  pos_targets = [
    torch.tensor([[0.1, 0.0]], device=device),
    torch.tensor([[0.3, 0.0]], device=device),
    torch.tensor([[0.5, 0.0]], device=device),
  ]
  vel_targets = [
    torch.tensor([[1.0, 0.0]], device=device),
    torch.tensor([[2.0, 0.0]], device=device),
    torch.tensor([[3.0, 0.0]], device=device),
  ]
  entity.write_joint_state_to_sim(
    position=torch.zeros(1, 2, device=device),
    velocity=torch.zeros(1, 2, device=device),
  )
  for p, v in zip(pos_targets, vel_targets, strict=True):
    entity.set_joint_position_target(p)
    entity.set_joint_velocity_target(v)
    entity.set_joint_effort_target(torch.zeros(1, 2, device=device))
    entity.write_data_to_sim()
    sim.forward()

  v_adr = entity.indexing.joint_v_adr
  # With lag=2, both halves should reference step-0 values.
  expected = KP * pos_targets[0][0] + KD * vel_targets[0][0]
  assert torch.allclose(sim.data.qfrc_actuator[0, v_adr], expected, atol=1e-4)


def test_reset_clears_delay_buffer(device):
  entity, _ = initialize_entity(_make_entity(delay_min_lag=1, delay_max_lag=3), device)
  act = entity.actuators[0]
  assert act._delay_buffer is not None
  entity.set_joint_position_target(torch.full((1, 2), 0.5, device=device))
  entity.set_joint_velocity_target(torch.zeros(1, 2, device=device))
  entity.write_data_to_sim()

  entity.reset(torch.tensor([0], device=device))
  assert act._delay_buffer.current_lags[0] == 0


# ---------------------------------------------------------------------------
# Domain randomization
# ---------------------------------------------------------------------------


def _scene_env(device, transmission=TransmissionType.JOINT, num_envs=2):
  """Build a real scene/sim with one BuiltinPd-driven entity for DR tests."""
  if transmission == TransmissionType.JOINT:
    xml = ROBOT_XML
    targets = ("joint.*",)
  else:
    xml = load_fixture_xml("tendon_finger")
    # tendon_finger ships with motor/position/velocity actuators; we need a
    # bare spec so BuiltinPd can attach to the tendon without name clashes.
    targets = ("finger_tendon",)

  def spec_fn():
    spec = mujoco.MjSpec.from_string(xml)
    # Strip any pre-existing actuators so BuiltinPd's added elements own ctrl.
    for a in list(spec.actuators):
      spec.delete(a)
    return spec

  entity_cfg = EntityCfg(
    spec_fn=spec_fn,
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPdActuatorCfg(
          target_names_expr=targets,
          stiffness=KP,
          damping=KD,
          effort_limit=50.0,
          transmission_type=transmission,
        ),
      )
    ),
  )
  scene_cfg = SceneCfg(num_envs=num_envs, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(model, sim.model, sim.data)

  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.scene = {"robot": scene["robot"]}
  env.sim = sim
  return env


def test_dr_pd_gains_scales_halves_independently(device):
  env = _scene_env(device)
  robot = env.scene["robot"]
  act = robot.actuators[0]
  assert isinstance(act, BuiltinPdActuator)
  n = act.num_targets
  pos_ids = act.global_ctrl_ids[:n]
  vel_ids = act.global_ctrl_ids[n:]

  # Expand fields so DR can write per-env.
  env.sim.expand_model_fields(("actuator_gainprm", "actuator_biasprm"))

  dr.pd_gains(
    env,
    env_ids=torch.tensor([0], device=device),
    kp_range=(2.0, 2.0),
    kd_range=(3.0, 3.0),
    asset_cfg=SceneEntityCfg("robot"),
    operation="scale",
  )

  m = env.sim.model
  # Position half: gainprm[0] and biasprm[1] both scaled by kp=2, biasprm[2]
  # must stay zero (no kd injection).
  assert torch.allclose(
    m.actuator_gainprm[0, pos_ids, 0],
    torch.full((n,), 2.0 * KP, device=device),
  )
  assert torch.allclose(
    m.actuator_biasprm[0, pos_ids, 1],
    torch.full((n,), -2.0 * KP, device=device),
  )
  assert torch.allclose(
    m.actuator_biasprm[0, pos_ids, 2], torch.zeros(n, device=device)
  )
  # Velocity half: gainprm[0] and biasprm[2] both scaled by kd=3, biasprm[1]
  # stays zero (no kp injection).
  assert torch.allclose(
    m.actuator_gainprm[0, vel_ids, 0],
    torch.full((n,), 3.0 * KD, device=device),
  )
  assert torch.allclose(
    m.actuator_biasprm[0, vel_ids, 2],
    torch.full((n,), -3.0 * KD, device=device),
  )
  assert torch.allclose(
    m.actuator_biasprm[0, vel_ids, 1], torch.zeros(n, device=device)
  )
  # The other env must be untouched.
  assert torch.allclose(m.actuator_gainprm[1, pos_ids, 0], torch.tensor(KP))
  assert torch.allclose(m.actuator_gainprm[1, vel_ids, 0], torch.tensor(KD))


def test_dr_pd_gains_abs_writes_correct_columns(device):
  env = _scene_env(device)
  robot = env.scene["robot"]
  act = robot.actuators[0]
  assert isinstance(act, BuiltinPdActuator)
  n = act.num_targets
  pos_ids = act.global_ctrl_ids[:n]
  vel_ids = act.global_ctrl_ids[n:]
  env.sim.expand_model_fields(("actuator_gainprm", "actuator_biasprm"))

  dr.pd_gains(
    env,
    env_ids=torch.tensor([0, 1], device=device),
    kp_range=(200.0, 200.0),
    kd_range=(25.0, 25.0),
    asset_cfg=SceneEntityCfg("robot"),
    operation="abs",
  )

  m = env.sim.model
  assert torch.allclose(
    m.actuator_gainprm[:, pos_ids, 0],
    torch.full((env.num_envs, n), 200.0, device=device),
  )
  assert torch.allclose(
    m.actuator_biasprm[:, pos_ids, 1],
    torch.full((env.num_envs, n), -200.0, device=device),
  )
  assert torch.allclose(
    m.actuator_biasprm[:, pos_ids, 2],
    torch.zeros(env.num_envs, n, device=device),
  )
  assert torch.allclose(
    m.actuator_gainprm[:, vel_ids, 0],
    torch.full((env.num_envs, n), 25.0, device=device),
  )
  assert torch.allclose(
    m.actuator_biasprm[:, vel_ids, 2],
    torch.full((env.num_envs, n), -25.0, device=device),
  )


def test_dr_effort_limits_writes_jnt_actfrcrange(device):
  env = _scene_env(device, transmission=TransmissionType.JOINT)
  robot = env.scene["robot"]
  act = robot.actuators[0]
  assert isinstance(act, BuiltinPdActuator)
  env.sim.expand_model_fields(
    ("actuator_forcerange", "jnt_actfrcrange", "tendon_actfrcrange")
  )

  joint_ids = robot.indexing.joint_ids[act.target_ids]
  pre_forcerange = env.sim.model.actuator_forcerange.clone()

  dr.effort_limits(
    env,
    env_ids=torch.tensor([0], device=device),
    effort_limit_range=(123.0, 123.0),
    asset_cfg=SceneEntityCfg("robot"),
    operation="abs",
  )

  m = env.sim.model
  # The joint sum-clamp was rewritten on env 0 only.
  assert torch.allclose(
    m.jnt_actfrcrange[0, joint_ids],
    torch.tensor([[-123.0, 123.0]] * len(joint_ids), device=device),
  )
  assert torch.allclose(
    m.jnt_actfrcrange[1, joint_ids],
    torch.tensor([[-50.0, 50.0]] * len(joint_ids), device=device),
  )
  # Per-element actuator_forcerange must be untouched for BuiltinPd: that
  # field belongs to the existing single-element actuator semantic.
  assert torch.allclose(m.actuator_forcerange, pre_forcerange)


def test_dr_effort_limits_scale_multiplies_default(device):
  """``scale`` multiplies the configured ``effort_limit`` (50.0) by the sample."""
  env = _scene_env(device, transmission=TransmissionType.JOINT)
  robot = env.scene["robot"]
  act = robot.actuators[0]
  assert isinstance(act, BuiltinPdActuator)
  env.sim.expand_model_fields(
    ("actuator_forcerange", "jnt_actfrcrange", "tendon_actfrcrange")
  )
  joint_ids = robot.indexing.joint_ids[act.target_ids]

  dr.effort_limits(
    env,
    env_ids=torch.tensor([0], device=device),
    effort_limit_range=(2.0, 2.0),
    asset_cfg=SceneEntityCfg("robot"),
    operation="scale",
  )

  m = env.sim.model
  # Default is [-50, 50], scaled by 2 -> [-100, 100].
  assert torch.allclose(
    m.jnt_actfrcrange[0, joint_ids],
    torch.tensor([[-100.0, 100.0]] * len(joint_ids), device=device),
  )


def test_dr_effort_limits_writes_tendon_actfrcrange(device):
  env = _scene_env(device, transmission=TransmissionType.TENDON)
  robot = env.scene["robot"]
  act = robot.actuators[0]
  assert isinstance(act, BuiltinPdActuator)
  env.sim.expand_model_fields(
    ("actuator_forcerange", "jnt_actfrcrange", "tendon_actfrcrange")
  )
  tendon_ids = robot.indexing.tendon_ids[act.target_ids]

  dr.effort_limits(
    env,
    env_ids=torch.tensor([0], device=device),
    effort_limit_range=(77.0, 77.0),
    asset_cfg=SceneEntityCfg("robot"),
    operation="abs",
  )

  m = env.sim.model
  assert torch.allclose(
    m.tendon_actfrcrange[0, tendon_ids],
    torch.tensor([[-77.0, 77.0]] * len(tendon_ids), device=device),
  )
  assert torch.allclose(
    m.tendon_actfrcrange[1, tendon_ids],
    torch.tensor([[-50.0, 50.0]] * len(tendon_ids), device=device),
  )
