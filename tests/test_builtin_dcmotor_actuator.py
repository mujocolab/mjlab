"""Tests for BuiltinDcMotorActuator.

Covers wiring of MuJoCo's native ``<dcmotor>`` element through mjlab: the
three input modes (voltage / position / velocity), torque saturation,
config validation, and DR integration.
"""

import math
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

from mjlab.actuator import (
  BuiltinDcMotorActuator,
  BuiltinDcMotorActuatorCfg,
  DcMotorDatasheetParams,
  DcMotorInputMode,
  DcMotorPhysicalParams,
)
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg

ROBOT_XML = load_fixture_xml("floating_base_articulated")

# Motor characterization used throughout (resolves to K=0.24, R=2.88).
V_NOM, TAU_STALL, OMEGA_NL = 24.0, 2.0, 100.0
K = V_NOM / OMEGA_NL
R = K * V_NOM / TAU_STALL


@pytest.fixture(scope="module")
def device():
  return get_test_device()


DATASHEET = DcMotorDatasheetParams(
  nominal_voltage=V_NOM, stall_torque=TAU_STALL, no_load_speed=OMEGA_NL
)


def _make_cfg(
  *,
  mode: DcMotorInputMode = DcMotorInputMode.POSITION,
  stiffness=5.0,
  damping=0.5,
  voltage_limit=24.0,
  **extra,
) -> BuiltinDcMotorActuatorCfg:
  """Build a cfg with sensible PID defaults. ``extra`` forwards any other
  BuiltinDcMotorActuatorCfg kwarg (effort_limit, integral_gain, thermal,
  delay_*, etc.)."""
  return BuiltinDcMotorActuatorCfg(
    target_names_expr=("joint.*",),
    mode=mode,
    motor_params=DATASHEET,
    stiffness=stiffness,
    damping=damping,
    voltage_limit=voltage_limit,
    **extra,
  )


def _make_entity(**kwargs) -> Entity:
  return create_entity_with_actuator(ROBOT_XML, _make_cfg(**kwargs))


def _make_initialized(device, **kwargs):
  """Build entity from cfg kwargs and initialize it through the sim."""
  return initialize_entity(_make_entity(**kwargs), device)


def _drive(
  entity: Entity,
  sim,
  device: str,
  *,
  pos_target=None,
  vel_target=None,
  effort_target=None,
  q0=None,
  qd0=None,
) -> None:
  zero = torch.zeros(1, 2, device=device)
  entity.write_joint_state_to_sim(
    position=q0 if q0 is not None else zero,
    velocity=qd0 if qd0 is not None else zero,
  )
  entity.set_joint_position_target(pos_target if pos_target is not None else zero)
  entity.set_joint_velocity_target(vel_target if vel_target is not None else zero)
  entity.set_joint_effort_target(effort_target if effort_target is not None else zero)
  entity.write_data_to_sim()
  sim.forward()


# Wiring sanity.


def test_kr_packed_into_gainprm(device):
  """The XML compiler derives K and R from the nominal triplet."""
  _, sim = initialize_entity(_make_entity(effort_limit=1.5), device)
  m = sim.mj_model
  for i in range(2):
    assert m.actuator_gainprm[i, 0] == pytest.approx(R, abs=1e-6)
    assert m.actuator_gainprm[i, 1] == pytest.approx(K, abs=1e-6)
    assert m.actuator_gainprm[i, 4] == pytest.approx(5.0)  # kp
    assert m.actuator_gainprm[i, 6] == pytest.approx(0.5)  # kd
    assert m.actuator_gainprm[i, 7] == pytest.approx(24.0)  # Vmax
    assert m.actuator_gainprm[i, 8] == pytest.approx(1.0)  # input_mode=position
    assert m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_DCMOTOR
    assert m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_DCMOTOR
    # No activation state: ki=0, no inductance, no thermal/lugre/slew.
    assert m.actuator_actnum[i] == 0


def test_motor_const_path(device):
  """Physical params pack K = sqrt(Kt*Ke) and R verbatim."""
  cfg = BuiltinDcMotorActuatorCfg(
    target_names_expr=("joint.*",),
    mode=DcMotorInputMode.VOLTAGE,
    motor_params=DcMotorPhysicalParams(kt=0.1, ke=0.05, resistance=2.0),
  )
  _, sim = initialize_entity(create_entity_with_actuator(ROBOT_XML, cfg), device)
  m = sim.mj_model
  for i in range(2):
    assert m.actuator_gainprm[i, 0] == pytest.approx(2.0, abs=1e-6)
    assert m.actuator_gainprm[i, 1] == pytest.approx((0.1 * 0.05) ** 0.5, abs=1e-6)


# Stateless motor physics.


def test_voltage_mode_steady_state(device):
  """At rest, ctrl = V -> tau = K * V / R."""
  cfg = BuiltinDcMotorActuatorCfg(
    target_names_expr=("joint.*",),
    mode=DcMotorInputMode.VOLTAGE,
    motor_params=DATASHEET,
  )
  entity, sim = initialize_entity(create_entity_with_actuator(ROBOT_XML, cfg), device)
  V = torch.tensor([[10.0, -5.0]], device=device)
  _drive(entity, sim, device, effort_target=V)
  v_adr = entity.indexing.joint_v_adr
  expected = K * V[0] / R
  assert torch.allclose(sim.data.qfrc_actuator[0, v_adr], expected, atol=1e-4)


def test_voltage_mode_voltage_limit_zero_is_noop(device):
  """Docstring promises ``voltage_limit=0`` disables clamping. Verify against
  MuJoCo's ``dcmotor_voltage`` (which only clamps when ``Vmax > 0``)."""
  cfg = BuiltinDcMotorActuatorCfg(
    target_names_expr=("joint.*",),
    mode=DcMotorInputMode.VOLTAGE,
    motor_params=DATASHEET,
    voltage_limit=0.0,
  )
  entity, sim = initialize_entity(create_entity_with_actuator(ROBOT_XML, cfg), device)
  V = torch.tensor([[1000.0, 0.0]], device=device)  # absurdly high voltage.
  _drive(entity, sim, device, effort_target=V)
  v_adr = entity.indexing.joint_v_adr
  expected = K * V[0] / R
  assert torch.allclose(sim.data.qfrc_actuator[0, v_adr], expected, atol=1e-2)


def test_back_emf_reduces_torque_at_velocity(device):
  """Same V, joint moving at omega: tau = K * (V - K * omega) / R."""
  cfg = BuiltinDcMotorActuatorCfg(
    target_names_expr=("joint.*",),
    mode=DcMotorInputMode.VOLTAGE,
    motor_params=DATASHEET,
  )
  entity, sim = initialize_entity(create_entity_with_actuator(ROBOT_XML, cfg), device)
  V = torch.tensor([[10.0, 0.0]], device=device)
  omega0 = torch.tensor([[2.0, 0.0]], device=device)
  _drive(entity, sim, device, effort_target=V, qd0=omega0)
  v_adr = entity.indexing.joint_v_adr
  expected = K * (V[0] - K * omega0[0]) / R
  assert torch.allclose(sim.data.qfrc_actuator[0, v_adr], expected, atol=1e-4)


def test_position_mode_pid_at_rest(device):
  """kd=0, no Vmax clamp: tau = K * kp * (target - q) / R."""
  # voltage_limit must be >0 (cfg invariant), pick it big enough not to clamp.
  entity, sim = initialize_entity(
    _make_entity(damping=0.0, voltage_limit=1000.0), device
  )
  pos = torch.tensor([[0.1, -0.05]], device=device)
  _drive(entity, sim, device, pos_target=pos)
  v_adr = entity.indexing.joint_v_adr
  expected = K * 5.0 * pos[0] / R
  assert torch.allclose(sim.data.qfrc_actuator[0, v_adr], expected, atol=1e-4)


def test_position_mode_voltage_clamp(device):
  """Huge position error -> PID voltage saturates at Vmax."""
  entity, sim = initialize_entity(
    _make_entity(stiffness=100.0, damping=0.0, voltage_limit=2.0),
    device,
  )
  # kp * err = 100 * 0.5 = 50 V, well above Vmax=2.
  pos = torch.tensor([[0.5, 0.0]], device=device)
  _drive(entity, sim, device, pos_target=pos)
  v_adr = entity.indexing.joint_v_adr
  qfrc = sim.data.qfrc_actuator[0, v_adr]
  expected_first = K * 2.0 / R  # tau at clamped V.
  assert qfrc[0].item() == pytest.approx(expected_first, abs=1e-4)
  assert qfrc[1].item() == pytest.approx(0.0, abs=1e-4)


def test_velocity_mode_pid(device):
  """P-only velocity tracking: tau = K * kp * (target - qdot) / R."""
  entity, sim = initialize_entity(
    _make_entity(mode=DcMotorInputMode.VELOCITY, damping=0.0, voltage_limit=1000.0),
    device,
  )
  qd0 = torch.tensor([[1.0, 0.0]], device=device)
  vel_target = torch.tensor([[3.0, 0.0]], device=device)
  _drive(entity, sim, device, vel_target=vel_target, qd0=qd0)
  v_adr = entity.indexing.joint_v_adr
  # back-EMF subtracts K*omega; this is folded into the dcmotor bias.
  # voltage = kp*(target - qdot); tau = K*(voltage - K*omega)/R.
  voltage = 5.0 * (vel_target[0] - qd0[0])
  expected = K * (voltage - K * qd0[0]) / R
  assert torch.allclose(sim.data.qfrc_actuator[0, v_adr], expected, atol=1e-4)


def test_effort_limit_clamps_torque(device):
  """forcerange clamps the algebraic torque output."""
  entity, sim = initialize_entity(
    _make_entity(stiffness=100.0, damping=0.0, voltage_limit=1000.0, effort_limit=0.1),
    device,
  )
  m = sim.mj_model
  for i in range(2):
    assert m.actuator_forcelimited[i] == 1
    assert m.actuator_forcerange[i, 0] == pytest.approx(-0.1)
    assert m.actuator_forcerange[i, 1] == pytest.approx(0.1)

  # Unclamped tau would be K * 100 * 0.5 / R ~= K*50/R, well above 0.1.
  pos = torch.tensor([[0.5, 0.0]], device=device)
  _drive(entity, sim, device, pos_target=pos)
  v_adr = entity.indexing.joint_v_adr
  qfrc = sim.data.qfrc_actuator[0, v_adr]
  assert qfrc[0].item() == pytest.approx(0.1, abs=1e-4)
  assert qfrc[1].item() == pytest.approx(0.0, abs=1e-4)


# Cogging.


def test_cogging_packed_into_biasprm(device):
  """``cogging=(A, Np, phi)`` packs into ``biasprm[0:3]``."""
  cfg = BuiltinDcMotorActuatorCfg(
    target_names_expr=("joint.*",),
    mode=DcMotorInputMode.VOLTAGE,
    motor_params=DATASHEET,
    cogging=(0.5, 4.0, 0.1),
  )
  _, sim = initialize_entity(create_entity_with_actuator(ROBOT_XML, cfg), device)
  m = sim.mj_model
  for i in range(2):
    assert m.actuator_biasprm[i, 0] == pytest.approx(0.5)
    assert m.actuator_biasprm[i, 1] == pytest.approx(4.0)
    assert m.actuator_biasprm[i, 2] == pytest.approx(0.1)


def test_cogging_contributes_torque(device):
  """At ctrl=0 (no electromagnetic torque), qfrc_actuator equals the cogging
  term ``A * sin(Np * q + phi)`` evaluated at the joint angle."""
  A, Np, phi = 0.5, 4.0, 0.1
  cfg = BuiltinDcMotorActuatorCfg(
    target_names_expr=("joint.*",),
    mode=DcMotorInputMode.VOLTAGE,
    motor_params=DATASHEET,
    cogging=(A, Np, phi),
  )
  entity, sim = initialize_entity(create_entity_with_actuator(ROBOT_XML, cfg), device)
  q0, q1 = 0.3, -0.2
  _drive(
    entity,
    sim,
    device,
    q0=torch.tensor([[q0, q1]], device=device),
    effort_target=torch.zeros(1, 2, device=device),
  )
  v_adr = entity.indexing.joint_v_adr
  qfrc = sim.data.qfrc_actuator[0, v_adr]
  assert qfrc[0].item() == pytest.approx(A * math.sin(Np * q0 + phi), abs=1e-5)
  assert qfrc[1].item() == pytest.approx(A * math.sin(Np * q1 + phi), abs=1e-5)


def test_cogging_bypasses_effort_limit(device):
  """Cogging is added *after* the forcerange clamp (MuJoCo's intentional
  model: ``effort_limit`` bounds electromagnetic torque, cogging is
  mechanical). Total torque can exceed ``effort_limit`` by up to the
  cogging amplitude."""
  A, Np, phi = 0.5, 0.0, math.pi / 2  # sin(pi/2)=1, so cogging = A at any q.
  cfg = BuiltinDcMotorActuatorCfg(
    target_names_expr=("joint.*",),
    mode=DcMotorInputMode.VOLTAGE,
    motor_params=DATASHEET,
    cogging=(A, Np, phi),
    effort_limit=0.05,  # An order of magnitude below A.
  )
  entity, sim = initialize_entity(create_entity_with_actuator(ROBOT_XML, cfg), device)
  # Pick a voltage large enough that the electromagnetic torque alone
  # would saturate forcerange at +/- 0.05.
  V = torch.tensor([[100.0, 0.0]], device=device)
  _drive(entity, sim, device, effort_target=V)
  v_adr = entity.indexing.joint_v_adr
  qfrc = sim.data.qfrc_actuator[0, v_adr]
  # joint1: electromagnetic clamped to +0.05, plus cogging A=0.5.
  assert qfrc[0].item() == pytest.approx(0.05 + A, abs=1e-5)
  # joint2: zero voltage, electromagnetic=0, only cogging.
  assert qfrc[1].item() == pytest.approx(A, abs=1e-5)


# Optional stateful extensions (integral, slew, inductance, thermal, LuGre).
# Each behavior check compares against a baseline with the feature disabled
# so that removing the wiring in edit_spec causes the comparison to fail.


def _step_n(entity, sim, device, n: int, *, pos_target=None, eff_target=None):
  zero = torch.zeros(1, 2, device=device)
  entity.write_joint_state_to_sim(position=zero, velocity=zero)
  for _ in range(n):
    entity.set_joint_position_target(pos_target if pos_target is not None else zero)
    entity.set_joint_velocity_target(zero)
    entity.set_joint_effort_target(eff_target if eff_target is not None else zero)
    entity.write_data_to_sim()
    sim.step()


def _qfrc(entity, sim) -> torch.Tensor:
  return sim.data.qfrc_actuator[0, entity.indexing.joint_v_adr].clone()


def test_integral_gain_ramps_torque(device):
  """Integrator in position mode ramps torque over time even with ``kp``
  and ``kd`` near zero."""
  # stiffness must be > 0 (validation); choose tiny so ki dominates.
  base = dict(
    mode=DcMotorInputMode.POSITION, stiffness=1e-4, damping=0.0, voltage_limit=24.0
  )
  ent_off, sim_off = _make_initialized(device, **base, integral_gain=0.0)
  ent_on, sim_on = _make_initialized(device, **base, integral_gain=10.0)

  target = torch.tensor([[0.5, 0.0]], device=device)
  for sim, ent in ((sim_off, ent_off), (sim_on, ent_on)):
    _step_n(ent, sim, device, n=20, pos_target=target)
  assert _qfrc(ent_on, sim_on)[0].abs() > 100.0 * _qfrc(ent_off, sim_off)[0].abs()


def test_slew_rate_limits_voltage(device):
  """``slew_rate`` rate-limits ``ctrl``: after one step, effective voltage
  is far below the requested input."""
  base = dict(
    mode=DcMotorInputMode.VOLTAGE, stiffness=0.0, damping=0.0, voltage_limit=0.0
  )
  ent_off, sim_off = _make_initialized(device, **base, slew_rate=0.0)
  ent_on, sim_on = _make_initialized(device, **base, slew_rate=10.0)

  V = torch.tensor([[100.0, 0.0]], device=device)
  for sim, ent in ((sim_off, ent_off), (sim_on, ent_on)):
    _step_n(ent, sim, device, n=1, eff_target=V)
  assert _qfrc(ent_off, sim_off)[0] > 100.0 * _qfrc(ent_on, sim_on)[0]


def test_inductance_lags_current(device):
  """Large ``inductance`` (te >> dt) suppresses early-step torque."""
  base = dict(
    mode=DcMotorInputMode.VOLTAGE, stiffness=0.0, damping=0.0, voltage_limit=0.0
  )
  ent_off, sim_off = _make_initialized(device, **base, inductance=0.0)
  ent_on, sim_on = _make_initialized(device, **base, inductance=1.0)

  V = torch.tensor([[10.0, 0.0]], device=device)
  for sim, ent in ((sim_off, ent_off), (sim_on, ent_on)):
    _step_n(ent, sim, device, n=2, eff_target=V)
  assert _qfrc(ent_off, sim_off)[0].abs() > 10.0 * _qfrc(ent_on, sim_on)[0].abs()


def test_thermal_decays_torque(device):
  """I^2R heating raises T, which raises effective resistance and decays
  torque over time."""
  # Params chosen for visible effect in a handful of steps without going
  # numerically unstable: small C (fast heating) and modest alpha.
  base = dict(
    mode=DcMotorInputMode.VOLTAGE, stiffness=0.0, damping=0.0, voltage_limit=0.0
  )
  ent_off, sim_off = _make_initialized(device, **base)
  ent_on, sim_on = _make_initialized(
    device, **base, thermal=(1.0, 0.1, 0.0, 0.01, 0.0, 0.0)
  )

  V = torch.tensor([[100.0, 0.0]], device=device)
  for sim, ent in ((sim_off, ent_off), (sim_on, ent_on)):
    _step_n(ent, sim, device, n=5, eff_target=V)
  assert _qfrc(ent_on, sim_on)[0].abs() < 0.5 * _qfrc(ent_off, sim_off)[0].abs()


def test_lugre_subtracts_friction(device):
  """LuGre friction subtracts a velocity-dependent force after the
  ``effort_limit`` clamp (mechanical, like cogging)."""
  # Static comparison at v>0, ctrl=0; avoids feedback between LuGre slowing
  # the joint and back-EMF easing off under sim.step().
  #   no LuGre:  qfrc = -K^2 * v / R              (back-EMF only)
  #   w/ LuGre:  qfrc = -K^2 * v / R - sigma1*v - ...
  base = dict(
    mode=DcMotorInputMode.VOLTAGE, stiffness=0.0, damping=0.0, voltage_limit=0.0
  )
  ent_off, sim_off = _make_initialized(device, **base)
  ent_on, sim_on = _make_initialized(
    device, **base, lugre=(1e4, 100.0, 0.1, 0.15, 0.01)
  )

  zero = torch.zeros(1, 2, device=device)
  v0 = torch.tensor([[1.0, 0.0]], device=device)
  for sim, ent in ((sim_off, ent_off), (sim_on, ent_on)):
    ent.write_joint_state_to_sim(position=zero, velocity=v0)
    ent.set_joint_position_target(zero)
    ent.set_joint_velocity_target(zero)
    ent.set_joint_effort_target(zero)
    ent.write_data_to_sim()
    sim.forward()
  assert abs(_qfrc(ent_on, sim_on)[0]) > 100.0 * abs(_qfrc(ent_off, sim_off)[0])


# Config validation.


def test_pid_mode_requires_gains():
  with pytest.raises(ValueError, match="stiffness"):
    BuiltinDcMotorActuatorCfg(
      target_names_expr=("j",),
      mode=DcMotorInputMode.POSITION,
      motor_params=DATASHEET,
      voltage_limit=1.0,
    )
  with pytest.raises(ValueError, match="voltage_limit"):
    BuiltinDcMotorActuatorCfg(
      target_names_expr=("j",),
      mode=DcMotorInputMode.POSITION,
      motor_params=DATASHEET,
      stiffness=1.0,
    )


def test_voltage_mode_rejects_pid_gains():
  with pytest.raises(ValueError, match="VOLTAGE"):
    BuiltinDcMotorActuatorCfg(
      target_names_expr=("j",),
      mode=DcMotorInputMode.VOLTAGE,
      motor_params=DATASHEET,
      stiffness=1.0,
    )


def test_site_rejected():
  with pytest.raises(ValueError, match="SITE"):
    BuiltinDcMotorActuatorCfg(
      target_names_expr=("j",),
      motor_params=DATASHEET,
      stiffness=1.0,
      voltage_limit=1.0,
      transmission_type=TransmissionType.SITE,
    )


# Joint-level passthrough.


def test_armature_applied(device):
  _, sim = initialize_entity(_make_entity(armature=0.7), device)
  m = sim.mj_model
  for jname in ("joint1", "joint2"):
    dof_id = m.jnt_dofadr[m.joint(jname).id]
    assert m.dof_armature[dof_id] == pytest.approx(0.7)


# Domain randomization.


def _scene_env(
  device,
  num_envs=2,
  mode: DcMotorInputMode = DcMotorInputMode.POSITION,
):
  def spec_fn():
    spec = mujoco.MjSpec.from_string(ROBOT_XML)
    for a in list(spec.actuators):
      spec.delete(a)
    return spec

  entity_cfg = EntityCfg(
    spec_fn=spec_fn,
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinDcMotorActuatorCfg(
          target_names_expr=("joint.*",),
          mode=mode,
          motor_params=DATASHEET,
          stiffness=5.0 if mode != DcMotorInputMode.VOLTAGE else 0.0,
          damping=0.5 if mode != DcMotorInputMode.VOLTAGE else 0.0,
          voltage_limit=24.0 if mode != DcMotorInputMode.VOLTAGE else 0.0,
          effort_limit=50.0,
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


@pytest.mark.parametrize(
  "operation, kp_in, kd_in, kp_expected, kd_expected",
  [
    # scale: multiplies the configured defaults (kp=5.0, kd=0.5).
    ("scale", 2.0, 3.0, 2.0 * 5.0, 3.0 * 0.5),
    # abs: writes the value directly.
    ("abs", 10.0, 2.0, 10.0, 2.0),
  ],
)
def test_dr_pd_gains_position_mode(
  device, operation, kp_in, kd_in, kp_expected, kd_expected
):
  env = _scene_env(device)
  robot = env.scene["robot"]
  act = robot.actuators[0]
  assert isinstance(act, BuiltinDcMotorActuator)
  ctrl_ids = act.global_ctrl_ids
  env.sim.expand_model_fields(("actuator_gainprm", "actuator_biasprm"))

  dr.pd_gains(
    env,
    env_ids=torch.tensor([0], device=device),
    kp_range=(kp_in, kp_in),
    kd_range=(kd_in, kd_in),
    asset_cfg=SceneEntityCfg("robot"),
    operation=operation,
  )

  m = env.sim.model
  n = len(ctrl_ids)
  assert torch.allclose(
    m.actuator_gainprm[0, ctrl_ids, 4], torch.full((n,), kp_expected, device=device)
  )
  assert torch.allclose(
    m.actuator_gainprm[0, ctrl_ids, 6], torch.full((n,), kd_expected, device=device)
  )
  # Other env untouched (cfg defaults).
  assert torch.allclose(m.actuator_gainprm[1, ctrl_ids, 4], torch.tensor(5.0))
  assert torch.allclose(m.actuator_gainprm[1, ctrl_ids, 6], torch.tensor(0.5))


def test_dr_pd_gains_voltage_mode_rejected(device):
  env = _scene_env(device, mode=DcMotorInputMode.VOLTAGE)
  env.sim.expand_model_fields(("actuator_gainprm", "actuator_biasprm"))
  with pytest.raises(ValueError, match="VOLTAGE"):
    dr.pd_gains(
      env,
      env_ids=torch.tensor([0], device=device),
      kp_range=(1.0, 1.0),
      kd_range=(1.0, 1.0),
      asset_cfg=SceneEntityCfg("robot"),
    )


def test_dr_effort_limits_writes_forcerange(device):
  env = _scene_env(device)
  robot = env.scene["robot"]
  act = robot.actuators[0]
  ctrl_ids = act.global_ctrl_ids
  env.sim.expand_model_fields(
    ("actuator_forcerange", "jnt_actfrcrange", "tendon_actfrcrange")
  )

  dr.effort_limits(
    env,
    env_ids=torch.tensor([0], device=device),
    effort_limit_range=(123.0, 123.0),
    asset_cfg=SceneEntityCfg("robot"),
    operation="abs",
  )

  m = env.sim.model
  n = len(ctrl_ids)
  assert torch.allclose(
    m.actuator_forcerange[0, ctrl_ids, 0],
    torch.full((n,), -123.0, device=device),
  )
  assert torch.allclose(
    m.actuator_forcerange[0, ctrl_ids, 1],
    torch.full((n,), 123.0, device=device),
  )
  # Env 1 keeps the configured default of 50.
  assert torch.allclose(m.actuator_forcerange[1, ctrl_ids, 1], torch.tensor(50.0))


# Delay.


def test_delay_position_mode(device):
  """A 2-step lag should make position-mode torque reference step-0 target."""
  entity, sim = initialize_entity(
    _make_entity(
      stiffness=10.0,
      damping=0.0,
      voltage_limit=1000.0,
      delay_min_lag=2,
      delay_max_lag=2,
    ),
    device,
  )
  zero = torch.zeros(1, 2, device=device)
  entity.write_joint_state_to_sim(position=zero, velocity=zero)
  targets = [
    torch.tensor([[0.1, 0.0]], device=device),
    torch.tensor([[0.3, 0.0]], device=device),
    torch.tensor([[0.5, 0.0]], device=device),
  ]
  for p in targets:
    entity.set_joint_position_target(p)
    entity.set_joint_velocity_target(zero)
    entity.set_joint_effort_target(zero)
    entity.write_data_to_sim()
    sim.forward()

  v_adr = entity.indexing.joint_v_adr
  # With lag=2 and three writes, the effective target is targets[0].
  expected = K * 10.0 * targets[0][0] / R
  assert torch.allclose(sim.data.qfrc_actuator[0, v_adr], expected, atol=1e-4)
