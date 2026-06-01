"""Tests for FusedActuatorGroup (fused stateless-control-law actuators)."""

import mujoco
import pytest
import torch
from conftest import get_test_device, load_fixture_xml

from mjlab.actuator import (
  DcMotorActuator,
  DcMotorActuatorCfg,
  IdealPdActuator,
  IdealPdActuatorCfg,
  LearnedMlpActuator,
)
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg

ROBOT_XML = load_fixture_xml("floating_base_articulated")


@pytest.fixture(scope="module")
def device():
  return get_test_device()


def make_entity(actuator_cfgs, num_envs, device):
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML),
    articulation=EntityArticulationInfoCfg(actuators=actuator_cfgs),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


def test_idealpd_actuators_fused(device):
  """Ideal PD actuators with matching delay config fuse into one group."""
  cfg1 = IdealPdActuatorCfg(target_names_expr=("joint1",), stiffness=50.0, damping=5.0)
  cfg2 = IdealPdActuatorCfg(target_names_expr=("joint2",), stiffness=30.0, damping=3.0)
  entity, _ = make_entity((cfg1, cfg2), num_envs=2, device=device)

  assert len(entity._fused_actuator_group._groups) == 1
  assert entity._fused_actuator_group._groups[0].target_ids.numel() == 2
  assert len(entity._custom_actuators) == 0


def test_different_delay_configs_separate_groups(device):
  """Ideal PD actuators with different delay configs get separate groups."""
  cfg1 = IdealPdActuatorCfg(
    target_names_expr=("joint1",), stiffness=50.0, damping=5.0, delay_max_lag=3
  )
  cfg2 = IdealPdActuatorCfg(
    target_names_expr=("joint2",), stiffness=50.0, damping=5.0, delay_max_lag=5
  )
  entity, _ = make_entity((cfg1, cfg2), num_envs=2, device=device)

  assert len(entity._fused_actuator_group._groups) == 2


def test_fusable_detection_by_compute():
  """Fusability is keeping the shared stateless-law compute, not a flag."""
  from mjlab.actuator import BuiltinPositionActuator

  # IdealPd defines the shared law-applying compute; DcMotor inherits it.
  assert DcMotorActuator.compute is IdealPdActuator.compute
  # Custom compute (learned network, built-in field passthrough) opts out, with
  # no flag and immune to the DcMotor -> LearnedMlp inheritance trap.
  assert LearnedMlpActuator.compute is not IdealPdActuator.compute
  assert BuiltinPositionActuator.compute is not IdealPdActuator.compute


def test_dcmotor_fuses_separately_and_matches(device):
  """DcMotor fuses into its own group (distinct law) and matches per-actuator."""
  ideal = IdealPdActuatorCfg(target_names_expr=("joint1",), stiffness=50.0, damping=5.0)
  dc = DcMotorActuatorCfg(
    target_names_expr=("joint2",),
    stiffness=50.0,
    damping=5.0,
    effort_limit=20.0,
    saturation_effort=40.0,
    velocity_limit=30.0,
  )
  entity, _ = make_entity((ideal, dc), num_envs=4, device=device)
  fused = entity._fused_actuator_group

  # Different control laws (clamp vs torque-speed curve) -> separate groups,
  # nothing left on the custom path.
  assert len(fused._groups) == 2
  assert len(entity._custom_actuators) == 0

  data = entity.data
  # Large targets so the DC torque-speed clip is actually active.
  data.joint_pos_target[:] = 10.0 * torch.randn_like(data.joint_pos_target)
  data.joint_vel_target[:] = torch.randn_like(data.joint_vel_target)

  fused.apply_controls(data)
  for group in fused._groups:
    got = data.data.ctrl[:, data.indexing.ctrl_ids[group.ctrl_ids]]
    ref = torch.cat(
      [act.compute(act.get_command(data)) for act in group.absorbed_actuators], dim=1
    )
    assert torch.equal(got, ref)


def test_fused_matches_per_actuator(device):
  """Fused control output is identical to per-actuator compute (no delay)."""
  cfg1 = IdealPdActuatorCfg(
    target_names_expr=("joint1",), stiffness=50.0, damping=5.0, effort_limit=100.0
  )
  cfg2 = IdealPdActuatorCfg(
    target_names_expr=("joint2",), stiffness=30.0, damping=3.0, effort_limit=80.0
  )
  entity, _ = make_entity((cfg1, cfg2), num_envs=4, device=device)
  data = entity.data
  group = entity._fused_actuator_group._groups[0]

  data.joint_pos_target[:] = torch.randn_like(data.joint_pos_target)
  data.joint_vel_target[:] = torch.randn_like(data.joint_vel_target)

  entity._fused_actuator_group.apply_controls(data)
  fused_ctrl = data.data.ctrl[:, data.indexing.ctrl_ids[group.ctrl_ids]].clone()

  reference = torch.cat(
    [act.compute(act.get_command(data)) for act in group.absorbed_actuators], dim=1
  )
  assert torch.equal(fused_ctrl, reference)


def test_set_gains_writes_through_view(device):
  """Per-actuator set_gains mutates the fused gain tensor and its output."""
  cfg1 = IdealPdActuatorCfg(
    target_names_expr=("joint1",), stiffness=50.0, damping=5.0, effort_limit=1e6
  )
  cfg2 = IdealPdActuatorCfg(
    target_names_expr=("joint2",), stiffness=30.0, damping=3.0, effort_limit=1e6
  )
  entity, _ = make_entity((cfg1, cfg2), num_envs=4, device=device)
  data = entity.data
  group = entity._fused_actuator_group._groups[0]
  act0 = group.absorbed_actuators[0]
  assert isinstance(act0, IdealPdActuator)
  n0 = act0.target_ids.numel()

  env_ids = torch.arange(4, device=device)
  new_kp = torch.full((4, n0), 123.0, device=device)
  act0.set_gains(env_ids, kp=new_kp)

  # The view aliases the fused tensor in place.
  assert torch.equal(group.params["stiffness"][:, :n0], new_kp)

  data.joint_pos_target[:] = torch.randn_like(data.joint_pos_target)
  entity._fused_actuator_group.apply_controls(data)
  fused_ctrl = data.data.ctrl[:, data.indexing.ctrl_ids[group.ctrl_ids]]
  expected0 = act0.compute(act0.get_command(data))
  assert torch.equal(fused_ctrl[:, :n0], expected0)


def test_fused_delay_applies_lag(device):
  """A fused group with constant lag returns the command from `lag` steps ago."""
  lag = 3
  cfg = IdealPdActuatorCfg(
    target_names_expr=("joint.*",),
    stiffness=0.0,  # isolate the feedforward effort term.
    damping=0.0,
    effort_limit=1e9,
    delay_min_lag=lag,
    delay_max_lag=lag,
  )
  entity, _ = make_entity((cfg,), num_envs=2, device=device)
  data = entity.data
  group = entity._fused_actuator_group._groups[0]
  assert group.delay_buffer is not None

  # First append happened during a prior apply; drive a ramp and read it back.
  data.joint_effort_target[:] = 0.0
  entity._fused_actuator_group.apply_controls(data)  # seed history with 0.
  seen = []
  for step in range(1, 6):
    data.joint_effort_target[:] = float(step)
    entity._fused_actuator_group.apply_controls(data)
    seen.append(data.data.ctrl[:, data.indexing.ctrl_ids[group.ctrl_ids]][0, 0].item())

  # ctrl reflects the effort target from `lag` steps earlier; history starts at
  # the seeded 0 and the buffer fills before the ramp shows through.
  assert seen == [0.0, 0.0, 0.0, 1.0, 2.0]


def test_no_idealpd_actuators_empty_group(device):
  """With no ideal PD actuators the fused group is empty and harmless."""
  from mjlab.actuator import BuiltinPositionActuatorCfg

  cfg = BuiltinPositionActuatorCfg(
    target_names_expr=("joint.*",), stiffness=50.0, damping=5.0
  )
  entity, _ = make_entity((cfg,), num_envs=2, device=device)
  assert len(entity._fused_actuator_group._groups) == 0
  entity._fused_actuator_group.apply_controls(entity.data)  # no-op, must not raise.
