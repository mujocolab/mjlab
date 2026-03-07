"""Tests for DelayedBuiltinActuatorGroup."""

import mujoco
import pytest
import torch
from conftest import get_test_device, load_fixture_xml

from mjlab.actuator import (
  BuiltinPositionActuatorCfg,
  DelayedActuatorCfg,
  IdealPdActuatorCfg,
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


def test_fused_group_created(device):
  """Delayed builtins with same config are fused into one group."""
  cfg1 = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
      target_names_expr=("joint1",), stiffness=50.0, damping=5.0
    ),
    delay_min_lag=1,
    delay_max_lag=3,
  )
  cfg2 = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
      target_names_expr=("joint2",), stiffness=50.0, damping=5.0
    ),
    delay_min_lag=1,
    delay_max_lag=3,
  )
  entity, _ = make_entity((cfg1, cfg2), num_envs=2, device=device)

  assert len(entity._delayed_builtin_group._groups) == 1
  assert len(entity._custom_actuators) == 0


def test_different_delay_configs_separate_groups(device):
  """Delayed builtins with different delay configs get separate groups."""
  cfg1 = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
      target_names_expr=("joint1",), stiffness=50.0, damping=5.0
    ),
    delay_min_lag=1,
    delay_max_lag=3,
  )
  cfg2 = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
      target_names_expr=("joint2",), stiffness=50.0, damping=5.0
    ),
    delay_min_lag=5,
    delay_max_lag=10,
  )
  entity, _ = make_entity((cfg1, cfg2), num_envs=2, device=device)

  assert len(entity._delayed_builtin_group._groups) == 2


def test_non_builtin_delayed_not_fused(device):
  """Delayed non-builtin actuators remain in custom_actuators."""
  delayed_builtin = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
      target_names_expr=("joint1",), stiffness=50.0, damping=5.0
    ),
    delay_min_lag=1,
    delay_max_lag=3,
  )
  delayed_custom = DelayedActuatorCfg(
    base_cfg=IdealPdActuatorCfg(
      target_names_expr=("joint2",),
      stiffness=50.0,
      damping=5.0,
      effort_limit=100.0,
    ),
    delay_min_lag=1,
    delay_max_lag=3,
  )
  entity, _ = make_entity((delayed_builtin, delayed_custom), num_envs=2, device=device)

  assert len(entity._delayed_builtin_group._groups) == 1
  assert len(entity._custom_actuators) == 1


def test_delayed_controls_written(device):
  """Fused delayed builtins write controls to sim correctly."""
  cfg = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
      target_names_expr=("joint.*",), stiffness=50.0, damping=5.0
    ),
    delay_min_lag=0,
    delay_max_lag=0,
  )
  entity, sim = make_entity((cfg,), num_envs=1, device=device)

  target = torch.tensor([[0.5, -0.3]], device=device)
  entity.set_joint_position_target(target)
  entity.write_data_to_sim()

  # With lag=0, output equals input.
  assert torch.allclose(sim.data.ctrl[0], target[0])


def test_delay_actually_delays(device):
  """With nonzero fixed lag, a new target takes lag steps to appear."""
  cfg = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
      target_names_expr=("joint.*",), stiffness=50.0, damping=5.0
    ),
    delay_min_lag=2,
    delay_max_lag=2,
  )
  entity, sim = make_entity((cfg,), num_envs=1, device=device)

  target_a = torch.tensor([[1.0, 2.0]], device=device)
  target_b = torch.tensor([[5.0, 6.0]], device=device)

  # Fill the buffer with target_a so lag clamp doesn't mask the test.
  for _ in range(3):
    entity.set_joint_position_target(target_a)
    entity.write_data_to_sim()
  assert torch.allclose(sim.data.ctrl[0], target_a[0])

  # Now send target_b. With lag=2, it takes 2 more steps to arrive.
  entity.set_joint_position_target(target_b)
  entity.write_data_to_sim()
  assert torch.allclose(sim.data.ctrl[0], target_a[0])  # still old

  entity.set_joint_position_target(target_b)
  entity.write_data_to_sim()
  assert torch.allclose(sim.data.ctrl[0], target_a[0])  # still old

  entity.set_joint_position_target(target_b)
  entity.write_data_to_sim()
  assert torch.allclose(sim.data.ctrl[0], target_b[0])  # now arrives


def test_reset_clears_buffer(device):
  """Reset clears delay buffers so old values don't leak through."""
  cfg = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
      target_names_expr=("joint.*",), stiffness=50.0, damping=5.0
    ),
    delay_min_lag=2,
    delay_max_lag=2,
  )
  entity, sim = make_entity((cfg,), num_envs=1, device=device)

  target_a = torch.tensor([[1.0, 2.0]], device=device)
  target_b = torch.tensor([[5.0, 6.0]], device=device)

  # Fill buffer with target_a.
  for _ in range(3):
    entity.set_joint_position_target(target_a)
    entity.write_data_to_sim()
  assert torch.allclose(sim.data.ctrl[0], target_a[0])

  # Reset, then fill with target_b. Old target_a must not appear.
  entity.reset(torch.tensor([0], device=device))
  for _ in range(3):
    entity.set_joint_position_target(target_b)
    entity.write_data_to_sim()
  assert torch.allclose(sim.data.ctrl[0], target_b[0])
