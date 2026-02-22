"""Tests for EventManager, special dr.* functions, and recomputation."""

from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab import actuator
from mjlab.entity import EntityCfg
from mjlab.envs.mdp import dr, events
from mjlab.managers.event_manager import (
  EventManager,
  EventTermCfg,
  RecomputeLevel,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg

pytestmark = pytest.mark.filterwarnings(
  "ignore:Use of index_put_ on expanded tensors is deprecated:UserWarning"
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="5.0"/>
      <body name="link1" pos="0.2 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"
          armature="0.1"/>
        <geom name="link1_geom" type="box" size="0.05 0.05 0.2" mass="1.0"/>
      </body>
      <body name="link2" pos="-0.2 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="0 1.57"
          armature="0.1"/>
        <geom name="link2_geom" type="box" size="0.05 0.05 0.2" mass="1.0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

NUM_ENVS = 2


@pytest.fixture(scope="module")
def device():
  return get_test_device()


class Env:
  def __init__(self, scene, sim, device):
    self.scene = scene
    self.sim = sim
    self.num_envs = scene.num_envs
    self.device = device


def create_env(device, fields, num_envs=NUM_ENVS):
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML))
  scene_cfg = SceneCfg(num_envs=num_envs, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()

  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(model, sim.model, sim.data)
  sim.expand_model_fields(fields)

  return Env(scene, sim, device)


# ===========================================================================
# Section 1: EventManager
# ===========================================================================


def test_dr_fields_registered_in_event_manager(device):
  """@requires_model_fields functions populate manager.domain_randomization_fields."""
  env = Mock()
  env.num_envs = 4
  env.device = device
  env.scene = {}
  env.sim = Mock()

  cfg = {
    "friction_dr": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={"ranges": (0.3, 1.2)},
    ),
    "damping_dr": EventTermCfg(
      mode="reset",
      func=dr.joint_damping,
      params={"ranges": (0.1, 0.5)},
    ),
    "pd_gains": EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={"kp_range": (0.8, 1.2), "kd_range": (0.8, 1.2)},
    ),
    "effort_limits": EventTermCfg(
      mode="reset",
      func=dr.effort_limits,
      params={"effort_limit_range": (0.8, 1.2)},
    ),
    "regular_event": EventTermCfg(
      mode="reset",
      func=events.reset_joints_by_offset,
      params={"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0)},
    ),
  }

  manager = EventManager(cfg, env)

  assert "geom_friction" in manager.domain_randomization_fields
  assert "dof_damping" in manager.domain_randomization_fields
  assert "actuator_gainprm" in manager.domain_randomization_fields
  assert "actuator_biasprm" in manager.domain_randomization_fields
  assert "actuator_forcerange" in manager.domain_randomization_fields
  assert len(manager.domain_randomization_fields) == 5


def test_recompute_level_ordering():
  """IntEnum max() gives strongest level."""
  L = RecomputeLevel
  assert max(L.none, L.set_const_0) == L.set_const_0
  assert max(L.set_const_0, L.set_const) == L.set_const
  assert max(L.set_const, L.set_const_fixed) == L.set_const
  assert max(L.none, L.none) == L.none


def test_recompute_not_called_when_no_dr_fired(device):
  """Non-DR events don't trigger recompute."""
  env = Mock()
  env.num_envs = 2
  env.device = device
  env.scene = {}
  env.sim = Mock()
  env.sim.recompute_constants = Mock()

  cfg = {
    "regular_event": EventTermCfg(
      mode="reset",
      func=lambda env, env_ids: None,
      params={},
    ),
  }

  manager = EventManager(cfg, env)
  manager.apply(
    "reset",
    env_ids=torch.tensor([0], device=device),
    global_env_step_count=1,
  )

  env.sim.recompute_constants.assert_not_called()


# ===========================================================================
# Section 2: PD gains & effort limits
# ===========================================================================


def _make_pd_env(device, num_envs=2):
  """Create a mock env with Builtin, Xml, and Ideal actuators for PD tests."""
  env = Mock()
  env.num_envs = num_envs
  env.device = device

  mock_entity = Mock()

  builtin = Mock(spec=actuator.BuiltinPositionActuator)
  builtin.ctrl_ids = torch.tensor([0, 1], device=device)
  builtin.global_ctrl_ids = torch.tensor([0, 1], device=device)

  xml = Mock(spec=actuator.XmlPositionActuator)
  xml.ctrl_ids = torch.tensor([2, 3], device=device)
  xml.global_ctrl_ids = torch.tensor([2, 3], device=device)

  ideal = Mock(spec=actuator.IdealPdActuator)
  ideal.ctrl_ids = torch.tensor([4, 5], device=device)
  ideal.global_ctrl_ids = torch.tensor([4, 5], device=device)
  ideal.stiffness = torch.tensor([[100.0, 100.0]] * num_envs, device=device)
  ideal.damping = torch.tensor([[10.0, 10.0]] * num_envs, device=device)
  ideal.default_stiffness = torch.tensor([[100.0, 100.0]] * num_envs, device=device)
  ideal.default_damping = torch.tensor([[10.0, 10.0]] * num_envs, device=device)
  ideal.set_gains = actuator.IdealPdActuator.set_gains.__get__(ideal)

  mock_entity.actuators = [builtin, xml, ideal]
  env.scene = {"robot": mock_entity}

  env.sim = Mock()
  env.sim.model = Mock()
  env.sim.model.actuator_gainprm = torch.ones((num_envs, 6, 10), device=device) * 50.0
  env.sim.model.actuator_biasprm = torch.zeros((num_envs, 6, 10), device=device)
  env.sim.model.actuator_biasprm[:, :, 1] = -50.0
  env.sim.model.actuator_biasprm[:, :, 2] = -5.0

  default_gainprm = torch.ones((6, 10), device=device) * 50.0
  default_biasprm = torch.zeros((6, 10), device=device)
  default_biasprm[:, 1] = -50.0
  default_biasprm[:, 2] = -5.0
  defaults = {
    "actuator_gainprm": default_gainprm,
    "actuator_biasprm": default_biasprm,
  }
  env.sim.get_default_field = lambda f: defaults[f]

  return env, ideal


def _make_effort_env(device, num_envs=2):
  """Create a mock env with Builtin, Xml, and Ideal actuators for effort tests."""
  env = Mock()
  env.num_envs = num_envs
  env.device = device

  mock_entity = Mock()

  builtin = Mock(spec=actuator.BuiltinPositionActuator)
  builtin.ctrl_ids = torch.tensor([0, 1], device=device)
  builtin.global_ctrl_ids = torch.tensor([0, 1], device=device)

  xml = Mock(spec=actuator.XmlPositionActuator)
  xml.ctrl_ids = torch.tensor([2, 3], device=device)
  xml.global_ctrl_ids = torch.tensor([2, 3], device=device)

  ideal = Mock(spec=actuator.IdealPdActuator)
  ideal.ctrl_ids = torch.tensor([4, 5], device=device)
  ideal.global_ctrl_ids = torch.tensor([4, 5], device=device)
  ideal.force_limit = torch.tensor([[50.0, 50.0]] * num_envs, device=device)
  ideal.default_force_limit = torch.tensor([[50.0, 50.0]] * num_envs, device=device)
  ideal.set_effort_limit = actuator.IdealPdActuator.set_effort_limit.__get__(ideal)

  mock_entity.actuators = [builtin, xml, ideal]
  env.scene = {"robot": mock_entity}

  env.sim = Mock()
  env.sim.model = Mock()
  env.sim.model.actuator_forcerange = torch.zeros((num_envs, 6, 2), device=device)
  env.sim.model.actuator_forcerange[:, :, 0] = -100.0
  env.sim.model.actuator_forcerange[:, :, 1] = 100.0

  default_forcerange = torch.zeros((6, 2), device=device)
  default_forcerange[:, 0] = -100.0
  default_forcerange[:, 1] = 100.0
  env.sim.get_default_field = lambda field: default_forcerange

  return env, ideal


@pytest.mark.parametrize(
  "operation, env_id, kp, kd, expected_gainprm, expected_biasprm_1,"
  " expected_biasprm_2, expected_stiffness, expected_damping",
  [
    ("scale", 0, 1.5, 2.0, 75.0, -75.0, -10.0, 150.0, 20.0),
    ("abs", 1, 200.0, 25.0, 200.0, -200.0, -25.0, 200.0, 25.0),
  ],
)
def test_pd_gains(
  device,
  operation,
  env_id,
  kp,
  kd,
  expected_gainprm,
  expected_biasprm_1,
  expected_biasprm_2,
  expected_stiffness,
  expected_damping,
):
  """PD gains scale/abs on Builtin, Xml, and Ideal actuators."""
  env, ideal = _make_pd_env(device)

  dr.pd_gains(
    env,
    torch.tensor([env_id], device=device),
    kp_range=(kp, kp),
    kd_range=(kd, kd),
    asset_cfg=SceneEntityCfg("robot"),
    operation=operation,
  )

  d = device
  # Builtin actuator.
  assert torch.allclose(
    env.sim.model.actuator_gainprm[env_id, [0, 1], 0],
    torch.tensor([expected_gainprm] * 2, device=d),
  )
  assert torch.allclose(
    env.sim.model.actuator_biasprm[env_id, [0, 1], 1],
    torch.tensor([expected_biasprm_1] * 2, device=d),
  )
  assert torch.allclose(
    env.sim.model.actuator_biasprm[env_id, [0, 1], 2],
    torch.tensor([expected_biasprm_2] * 2, device=d),
  )
  # Xml actuator.
  assert torch.allclose(
    env.sim.model.actuator_gainprm[env_id, [2, 3], 0],
    torch.tensor([expected_gainprm] * 2, device=d),
  )
  # Ideal actuator.
  assert torch.allclose(
    ideal.stiffness[env_id],
    torch.tensor([expected_stiffness] * 2, device=d),
  )
  assert torch.allclose(
    ideal.damping[env_id],
    torch.tensor([expected_damping] * 2, device=d),
  )


def test_pd_gains_multi_env(device):
  """Independent values per environment."""
  env, _ = _make_pd_env(device)

  torch.manual_seed(42)
  dr.pd_gains(
    env,
    torch.tensor([0, 1], device=device),
    kp_range=(0.5, 2.0),
    kd_range=(0.5, 2.0),
    asset_cfg=SceneEntityCfg("robot"),
    operation="scale",
  )

  gains = env.sim.model.actuator_gainprm[:, :2, 0]
  assert (gains != 50.0).all()
  assert not torch.allclose(gains[0], gains[1])


@pytest.mark.parametrize(
  "operation, env_id, limit, expected_lower, expected_upper, expected_ideal",
  [
    ("scale", 0, 2.0, -200.0, 200.0, 100.0),
    ("abs", 1, 150.0, -150.0, 150.0, 150.0),
  ],
)
def test_effort_limits(
  device, operation, env_id, limit, expected_lower, expected_upper, expected_ideal
):
  """Effort limits scale/abs on Builtin, Xml, and Ideal actuators."""
  env, ideal = _make_effort_env(device)

  dr.effort_limits(
    env,
    torch.tensor([env_id], device=device),
    effort_limit_range=(limit, limit),
    asset_cfg=SceneEntityCfg("robot"),
    operation=operation,
  )

  d = device
  # Builtin.
  assert torch.allclose(
    env.sim.model.actuator_forcerange[env_id, [0, 1], 0],
    torch.tensor([expected_lower] * 2, device=d),
  )
  assert torch.allclose(
    env.sim.model.actuator_forcerange[env_id, [0, 1], 1],
    torch.tensor([expected_upper] * 2, device=d),
  )
  # Xml.
  assert torch.allclose(
    env.sim.model.actuator_forcerange[env_id, [2, 3], 0],
    torch.tensor([expected_lower] * 2, device=d),
  )
  # Ideal.
  assert torch.allclose(
    ideal.force_limit[env_id],
    torch.tensor([expected_ideal] * 2, device=d),
  )


def test_effort_limits_multi_env(device):
  """Independent values per environment."""
  env, _ = _make_effort_env(device)

  torch.manual_seed(42)
  dr.effort_limits(
    env,
    torch.tensor([0, 1], device=device),
    effort_limit_range=(0.5, 2.0),
    asset_cfg=SceneEntityCfg("robot"),
    operation="scale",
  )

  upper = env.sim.model.actuator_forcerange[:, :2, 1]
  assert (upper != 100.0).all()
  assert not torch.allclose(upper[0], upper[1])


def test_effort_limits_scale_no_accumulation(device):
  """Scale operation uses defaults (no accumulation)."""
  env, _ = _make_effort_env(device)

  for _ in range(3):
    dr.effort_limits(
      env,
      env_ids=None,
      effort_limit_range=(2.0, 2.0),
      asset_cfg=SceneEntityCfg("robot"),
      operation="scale",
    )

  actual_upper = env.sim.model.actuator_forcerange[0, 0, 1].item()
  assert abs(actual_upper - 200.0) < 1e-5


# ===========================================================================
# Section 3: Other events
# ===========================================================================


def test_reset_joints_by_offset(device):
  """reset_joints_by_offset applies offsets and respects joint limits."""
  env = Mock()
  env.num_envs = 2
  env.device = device

  mock_entity = Mock()
  mock_entity.data.default_joint_pos = torch.zeros((2, 3), device=device)
  mock_entity.data.default_joint_vel = torch.zeros((2, 3), device=device)
  mock_entity.data.soft_joint_pos_limits = torch.tensor(
    [
      [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
      [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
    ],
    device=device,
  )
  mock_entity.write_joint_state_to_sim = Mock()
  env.scene = {"robot": mock_entity}

  # Normal offset.
  events.reset_joints_by_offset(
    env,
    torch.tensor([0], device=device),
    position_range=(0.3, 0.3),
    velocity_range=(0.2, 0.2),
    asset_cfg=SceneEntityCfg("robot", joint_ids=slice(None)),
  )

  call_args = mock_entity.write_joint_state_to_sim.call_args
  joint_pos, joint_vel = call_args[0][0], call_args[0][1]
  assert torch.allclose(joint_pos, torch.ones_like(joint_pos) * 0.3)
  assert torch.allclose(joint_vel, torch.ones_like(joint_vel) * 0.2)

  # Clamping when offset exceeds limits.
  events.reset_joints_by_offset(
    env,
    torch.tensor([1], device=device),
    position_range=(1.0, 1.0),
    velocity_range=(0.0, 0.0),
    asset_cfg=SceneEntityCfg("robot", joint_ids=slice(None)),
  )

  call_args = mock_entity.write_joint_state_to_sim.call_args
  joint_pos = call_args[0][0]
  assert torch.allclose(joint_pos, torch.ones_like(joint_pos) * 0.5)


def test_sync_actuator_delays(device):
  """Samples lag in range and applies to all delayed actuators."""
  from mjlab.actuator.delayed_actuator import DelayedActuator

  env = Mock()
  env.num_envs = 4
  env.device = device

  delayed_1 = Mock(spec=DelayedActuator)
  delayed_2 = Mock(spec=DelayedActuator)
  non_delayed = Mock(spec=actuator.BuiltinPositionActuator)

  mock_entity = Mock()
  mock_entity.actuators = [delayed_1, non_delayed, delayed_2]
  env.scene = {"robot": mock_entity}

  torch.manual_seed(42)
  dr.sync_actuator_delays(
    env,
    env_ids=None,
    lag_range=(1, 5),
    asset_cfg=SceneEntityCfg("robot"),
  )

  delayed_1.set_lags.assert_called_once()
  delayed_2.set_lags.assert_called_once()
  assert not hasattr(non_delayed, "set_lags")

  # Both calls should receive the same lags (same sample).
  lags_1 = delayed_1.set_lags.call_args[0][0]
  lags_2 = delayed_2.set_lags.call_args[0][0]
  torch.testing.assert_close(lags_1, lags_2)

  assert torch.all(lags_1 >= 1)
  assert torch.all(lags_1 <= 5)
  assert len(lags_1) == 4


# ===========================================================================
# Section 4: Recomputation integration
# ===========================================================================


def test_body_mass_recompute_subtreemass(device):
  """body_subtreemass is correctly recomputed after body_mass change."""
  env = create_env(device, ("body_mass", "body_subtreemass"))
  robot = env.scene["robot"]
  body_ids = robot.indexing.body_ids

  original_subtreemass = env.sim.model.body_subtreemass[:, body_ids].clone()

  env.sim.model.body_mass[0, body_ids] *= 2.0
  env.sim.recompute_constants(RecomputeLevel.set_const)

  new_subtreemass = env.sim.model.body_subtreemass[:, body_ids]

  assert not torch.allclose(new_subtreemass[0], original_subtreemass[0])
  torch.testing.assert_close(new_subtreemass[1], original_subtreemass[1])


def test_body_mass_recompute_invweight(device):
  """dof_invweight0 changes after body_mass modification."""
  env = create_env(
    device,
    ("body_mass", "body_subtreemass", "dof_invweight0", "body_invweight0"),
  )
  robot = env.scene["robot"]
  body_ids = robot.indexing.body_ids
  dof_adr = robot.indexing.joint_v_adr

  original_invweight = env.sim.model.dof_invweight0[:, dof_adr].clone()

  env.sim.model.body_mass[0, body_ids] *= 2.0
  env.sim.recompute_constants(RecomputeLevel.set_const)

  new_invweight = env.sim.model.dof_invweight0[:, dof_adr]

  assert not torch.allclose(new_invweight[0], original_invweight[0])
  torch.testing.assert_close(new_invweight[1], original_invweight[1])


def test_body_com_offset_recompute(device):
  """body_ipos modification triggers correct recomputation."""
  env = create_env(
    device,
    ("body_ipos", "body_subtreemass", "dof_invweight0", "body_invweight0"),
  )
  robot = env.scene["robot"]
  body_ids = robot.indexing.body_ids
  dof_adr = robot.indexing.joint_v_adr

  original_invweight = env.sim.model.dof_invweight0[:, dof_adr].clone()

  link1_idx = body_ids[1]
  env.sim.model.body_ipos[0, link1_idx, 0] += 0.05
  env.sim.recompute_constants(RecomputeLevel.set_const)

  new_invweight = env.sim.model.dof_invweight0[:, dof_adr]

  assert not torch.allclose(new_invweight[0], original_invweight[0])
  torch.testing.assert_close(new_invweight[1], original_invweight[1])


def test_dof_armature_recompute(device):
  """dof_armature modification updates dof_invweight0."""
  env = create_env(
    device,
    (
      "dof_armature",
      "dof_invweight0",
      "body_invweight0",
      "tendon_length0",
      "tendon_invweight0",
    ),
  )
  robot = env.scene["robot"]
  dof_adr = robot.indexing.joint_v_adr

  original_invweight = env.sim.model.dof_invweight0[:, dof_adr].clone()

  env.sim.model.dof_armature[0, dof_adr] *= 10.0
  env.sim.recompute_constants(RecomputeLevel.set_const_0)

  new_invweight = env.sim.model.dof_invweight0[:, dof_adr]

  assert not torch.allclose(new_invweight[0], original_invweight[0])
  torch.testing.assert_close(new_invweight[1], original_invweight[1])


# ===========================================================================
# Section 5: Recompute vs mj_setConst / recompiled MjSpec
# ===========================================================================


def test_body_mass_recompute_matches_set_const(device):
  """Modify body_mass: GPU recompute matches CPU mj_setConst."""
  gpu_model = mujoco.MjModel.from_xml_string(ROBOT_XML)
  cpu_model = mujoco.MjModel.from_xml_string(ROBOT_XML)

  link1_id = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_BODY, "link1")

  # GPU path.
  expand_fields = (
    "body_mass",
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
  )
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=gpu_model, device=device)
  sim.expand_model_fields(expand_fields)
  sim.model.body_mass[0, link1_id] = 5.0
  sim.recompute_constants(RecomputeLevel.set_const)

  # CPU ground truth: apply same cfg mutations as Simulation.__init__.
  SimulationCfg().mujoco.apply(cpu_model)
  cpu_data = mujoco.MjData(cpu_model)
  mujoco.mj_forward(cpu_model, cpu_data)
  cpu_model.body_mass[link1_id] = 5.0
  mujoco.mj_setConst(cpu_model, cpu_data)

  for field in ("body_subtreemass", "dof_invweight0", "body_invweight0"):
    gpu_val = getattr(sim.model, field)[0].cpu()
    ref_val = torch.tensor(getattr(cpu_model, field), dtype=torch.float32)
    torch.testing.assert_close(gpu_val, ref_val, atol=1e-4, rtol=1e-4)


def test_body_ipos_recompute_matches_set_const(device):
  """Modify body_ipos: GPU recompute matches CPU mj_setConst."""
  gpu_model = mujoco.MjModel.from_xml_string(ROBOT_XML)
  cpu_model = mujoco.MjModel.from_xml_string(ROBOT_XML)

  link1_id = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_BODY, "link1")

  # GPU path.
  expand_fields = (
    "body_ipos",
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
  )
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=gpu_model, device=device)
  sim.expand_model_fields(expand_fields)
  sim.model.body_ipos[0, link1_id, 0] += 0.05
  sim.recompute_constants(RecomputeLevel.set_const)

  # CPU ground truth: apply same cfg mutations as Simulation.__init__.
  SimulationCfg().mujoco.apply(cpu_model)
  cpu_data = mujoco.MjData(cpu_model)
  mujoco.mj_forward(cpu_model, cpu_data)
  cpu_model.body_ipos[link1_id, 0] += 0.05
  # Clear body_sameframe so mj_kinematics recomputes xipos from body_ipos.
  # (When sameframe != 0, mj_local2Global skips the ipos→xipos transform.)
  cpu_model.body_sameframe[link1_id] = 0
  mujoco.mj_setConst(cpu_model, cpu_data)

  for field in ("dof_invweight0", "body_invweight0"):
    gpu_val = getattr(sim.model, field)[0].cpu()
    ref_val = torch.tensor(getattr(cpu_model, field), dtype=torch.float32)
    torch.testing.assert_close(gpu_val, ref_val, atol=1e-4, rtol=1e-4)


def test_dof_armature_recompute_matches_recompile(device):
  """Modify dof_armature: GPU recompute matches fresh MjSpec.compile()."""
  spec = mujoco.MjSpec.from_string(ROBOT_XML)
  original_model = spec.compile()

  joint1_id = mujoco.mj_name2id(original_model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")

  # Ground truth: modify spec armature, recompile.
  spec.joint("joint1").armature = 1.0
  recompiled_model = spec.compile()

  # GPU path.
  expand_fields = (
    "dof_armature",
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
  )
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=original_model, device=device)
  sim.expand_model_fields(expand_fields)

  dof_adr = original_model.jnt_dofadr[joint1_id]
  sim.model.dof_armature[0, dof_adr] = 1.0
  sim.recompute_constants(RecomputeLevel.set_const_0)

  gpu_dof_invweight = sim.model.dof_invweight0[0].cpu()
  ref_dof_invweight = torch.tensor(recompiled_model.dof_invweight0, dtype=torch.float32)
  torch.testing.assert_close(gpu_dof_invweight, ref_dof_invweight, atol=1e-4, rtol=1e-4)
