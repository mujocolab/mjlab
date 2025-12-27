"""Tests for actions."""

from pathlib import Path
from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device, load_fixture_xml

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import BuiltinMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  SiteEffortActionCfg,
  TendonEffortActionCfg,
  TendonLengthActionCfg,
  TendonVelocityActionCfg,
)
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.fixture
def device():
  return get_test_device()


@pytest.fixture
def fixtures_dir():
  return Path(__file__).parent / "fixtures"


def make_entity(xml_or_path, target_expr, transmission_type, device, from_file=False):
  """Create and initialize entity."""

  def spec_fn():
    if from_file:
      return mujoco.MjSpec.from_file(str(xml_or_path))
    return mujoco.MjSpec.from_string(xml_or_path)

  cfg = EntityCfg(
    spec_fn=spec_fn,
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=target_expr,
          transmission_type=transmission_type,
          effort_limit=10.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(num_envs=4, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity


def make_env(entity, name, device):
  """Create mock environment."""
  env = Mock(spec=ManagerBasedRlEnv)
  env.num_envs = 4
  env.device = device
  env.scene = {name: entity}
  return env


def test_base_action_applies_scale_and_offset(fixtures_dir, device):
  """BaseAction: processed = raw * scale + offset."""
  entity = make_entity(
    fixtures_dir / "tendon_finger.xml",
    ("finger_tendon",),
    TransmissionType.TENDON,
    device,
    from_file=True,
  )
  env = make_env(entity, "finger", device)

  cfg = TendonLengthActionCfg(
    entity_name="finger",
    actuator_names=("finger_tendon",),
    scale=2.0,
    offset=0.5,
  )
  action = cfg.build(env)

  raw = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
  action.process_actions(raw)

  assert torch.allclose(action._processed_actions, raw * 2.0 + 0.5)


def test_base_action_reset_zeros_specific_envs(fixtures_dir, device):
  """BaseAction.reset() zeros raw_action for specified env_ids only."""
  entity = make_entity(
    fixtures_dir / "tendon_finger.xml",
    ("finger_tendon",),
    TransmissionType.TENDON,
    device,
    from_file=True,
  )
  env = make_env(entity, "finger", device)

  cfg = TendonLengthActionCfg(entity_name="finger", actuator_names=("finger_tendon",))
  action = cfg.build(env)

  action.process_actions(torch.ones(4, 1, device=device))
  action.reset(env_ids=torch.tensor([0, 2], device=device))

  assert torch.all(action.raw_action[[0, 2]] == 0.0)
  assert torch.all(action.raw_action[[1, 3]] == 1.0)


@pytest.mark.parametrize(
  "cfg_cls,target_attr,fixture,target_expr,transmission,entity_name",
  [
    # Joints.
    (
      JointPositionActionCfg,
      "joint_pos_target",
      "floating_base_articulated",
      ("joint.*",),
      TransmissionType.JOINT,
      "robot",
    ),
    # Tendons.
    (
      TendonLengthActionCfg,
      "tendon_len_target",
      "tendon_finger.xml",
      ("finger_tendon",),
      TransmissionType.TENDON,
      "finger",
    ),
    (
      TendonVelocityActionCfg,
      "tendon_vel_target",
      "tendon_finger.xml",
      ("finger_tendon",),
      TransmissionType.TENDON,
      "finger",
    ),
    (
      TendonEffortActionCfg,
      "tendon_effort_target",
      "tendon_finger.xml",
      ("finger_tendon",),
      TransmissionType.TENDON,
      "finger",
    ),
    # Sites.
    (
      SiteEffortActionCfg,
      "site_effort_target",
      "quadcopter.xml",
      ("rotor_.*",),
      TransmissionType.SITE,
      "drone",
    ),
  ],
)
def test_action_sets_entity_target(
  fixtures_dir,
  device,
  cfg_cls,
  target_attr,
  fixture,
  target_expr,
  transmission,
  entity_name,
):
  """Each action type writes to correct entity.data field."""
  if fixture.endswith(".xml"):
    entity = make_entity(
      fixtures_dir / fixture, target_expr, transmission, device, from_file=True
    )
  else:
    entity = make_entity(
      load_fixture_xml(fixture), target_expr, transmission, device, from_file=False
    )

  env = make_env(entity, entity_name, device)
  cfg = cfg_cls(entity_name=entity_name, actuator_names=target_expr)
  action = cfg.build(env)

  target = torch.arange(4 * action.action_dim, device=device, dtype=torch.float32)
  target = target.reshape(4, action.action_dim) * 0.1

  action.process_actions(target)
  action.apply_actions()

  entity_target = getattr(entity.data, target_attr)
  assert torch.allclose(entity_target, target)
