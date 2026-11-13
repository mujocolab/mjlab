"""Tests for EventManager, special dr.* functions, and recomputation."""

from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device
from mujoco_warp import SetConstGroup

from mjlab import actuator
from mjlab.entity import EntityCfg
from mjlab.envs.mdp import dr, events
from mjlab.managers.event_manager import (
  EventManager,
  EventTermCfg,
  recompute_groups,
  requires_model_fields,
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
  assert "jnt_actfrcrange" in manager.domain_randomization_fields
  assert "tendon_actfrcrange" in manager.domain_randomization_fields
  assert len(manager.domain_randomization_fields) == 7


def test_recompute_fields_registered_in_event_manager(device):
  """Recomputed model fields are expanded per environment."""
  env = Mock()
  env.num_envs = 4
  env.device = device
  env.scene = {}
  env.sim = Mock()

  cfg = {
    "body_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={"ranges": (-0.01, 0.01)},
    ),
  }

  manager = EventManager(cfg, env)

  assert manager.domain_randomization_fields == (
    "body_ipos",
    "tendon_length0",
    "eq_data",
    "cam_pos0",
    "cam_poscom0",
    "cam_mat0",
    "light_pos0",
    "light_poscom0",
    "light_dir0",
    "dof_invweight0",
    "body_invweight0",
    "tendon_invweight0",
    "actuator_acc0",
  )


def test_recompute_groups_of_fired_terms_are_combined(device):
  """Groups inferred from each fired term's fields are OR'ed into one call."""
  env = Mock()
  env.num_envs = 2
  env.device = device
  env.scene = {}
  env.sim = Mock()

  @requires_model_fields("eq_data")
  def kin(env, env_ids):
    del env, env_ids

  @requires_model_fields("dof_armature")
  def inert(env, env_ids):
    del env, env_ids

  cfg = {
    "kin": EventTermCfg(mode="startup", func=kin, params={}),
    "inert": EventTermCfg(mode="startup", func=inert, params={}),
  }
  EventManager(cfg, env).apply("startup")

  env.sim.recompute_constants.assert_called_once_with(
    SetConstGroup.KIN0 | SetConstGroup.INERT0
  )


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

  xml = Mock(spec=actuator.XmlActuator)
  xml.command_field = "position"
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

  xml = Mock(spec=actuator.XmlActuator)
  xml.command_field = "position"
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


def test_pd_gains_accepts_operation_object(device):
  """dr.scale / dr.abs Operation objects produce the same result as strings."""
  env_str, ideal_str = _make_pd_env(device)
  env_obj, ideal_obj = _make_pd_env(device)

  ids = torch.tensor([0], device=device)
  kwargs = dict(
    kp_range=(1.5, 1.5), kd_range=(2.0, 2.0), asset_cfg=SceneEntityCfg("robot")
  )

  torch.manual_seed(0)
  dr.pd_gains(env_str, ids, operation="scale", **kwargs)
  torch.manual_seed(0)
  dr.pd_gains(env_obj, ids, operation=dr.scale, **kwargs)

  assert torch.allclose(
    env_str.sim.model.actuator_gainprm[0], env_obj.sim.model.actuator_gainprm[0]
  )
  assert torch.allclose(ideal_str.stiffness, ideal_obj.stiffness)


def test_effort_limits_accepts_operation_object(device):
  """dr.abs Operation object produces the same result as the string."""
  env_str, ideal_str = _make_effort_env(device)
  env_obj, ideal_obj = _make_effort_env(device)

  ids = torch.tensor([0], device=device)
  kwargs = dict(effort_limit_range=(150.0, 150.0), asset_cfg=SceneEntityCfg("robot"))

  dr.effort_limits(env_str, ids, operation="abs", **kwargs)
  dr.effort_limits(env_obj, ids, operation=dr.abs, **kwargs)

  assert torch.allclose(
    env_str.sim.model.actuator_forcerange[0], env_obj.sim.model.actuator_forcerange[0]
  )
  assert torch.allclose(ideal_str.force_limit, ideal_obj.force_limit)


def test_pd_gains_rejects_unsupported_operation(device):
  """Operations other than scale/abs raise ValueError."""
  env, _ = _make_pd_env(device)
  ids = torch.tensor([0], device=device)

  with pytest.raises(ValueError, match="only supports 'scale' and 'abs'"):
    dr.pd_gains(env, ids, kp_range=(1.0, 1.0), kd_range=(1.0, 1.0), operation=dr.add)


def test_effort_limits_rejects_unsupported_operation(device):
  """Operations other than scale/abs raise ValueError."""
  env, _ = _make_effort_env(device)
  ids = torch.tensor([0], device=device)

  with pytest.raises(ValueError, match="only supports 'scale' and 'abs'"):
    dr.effort_limits(env, ids, effort_limit_range=(1.0, 1.0), operation=dr.add)


# ===========================================================================
# Section 3: Other events
# ===========================================================================


def test_reset_joints_by_offset(device):
  """reset_joints_by_offset applies offsets and respects joint limits."""
  env = Mock()
  env.num_envs = 2
  env.device = device

  mock_entity = Mock()
  mock_entity.indexing.has_ball_joints = False
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


# ===========================================================================
# Section 4: Step mode and apply_body_impulse
# ===========================================================================


def test_step_mode_fires_every_call(device):
  """Step-mode events fire unconditionally on every apply() call."""
  call_count = [0]

  def counter(env, env_ids):
    call_count[0] += 1

  env = Mock()
  env.num_envs = 2
  env.device = device
  env.scene = {}
  env.sim = Mock()

  cfg = {
    "step_counter": EventTermCfg(
      mode="step",
      func=counter,
      params={},
    ),
  }
  manager = EventManager(cfg, env)

  for _ in range(5):
    manager.apply(mode="step", dt=0.02)

  assert call_count[0] == 5


def _make_impulse_env(
  device, num_envs=2, num_bodies=1, body_ids=None, cooldown_s=(0.0, 0.0)
):
  """Create a mock env for apply_body_impulse tests."""
  if body_ids is None:
    body_ids = [0]
  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.step_dt = 0.02

  mock_entity = Mock()
  mock_entity.num_bodies = num_bodies
  mock_entity.data = Mock()
  mock_entity.data.body_link_quat_w = torch.zeros(
    (num_envs, num_bodies, 4), device=device
  )
  mock_entity.data.body_link_quat_w[..., 0] = 1.0
  # The inertia frame is rotated 90 degrees about x relative to the body frame.
  mock_entity.data.body_com_quat_w = torch.zeros(
    (num_envs, num_bodies, 4), device=device
  )
  mock_entity.data.body_com_quat_w[..., [0, 1]] = 0.5**0.5
  env.scene = {"robot": mock_entity}

  asset_cfg = SceneEntityCfg("robot", body_ids=body_ids)
  term_cfg = Mock()
  term_cfg.params = {"asset_cfg": asset_cfg, "cooldown_s": cooldown_s}
  impulse = events.apply_body_impulse(cfg=term_cfg, env=env)
  return env, mock_entity, asset_cfg, impulse


def test_apply_body_impulse_basic(device):
  """Impulse is applied and cleared after duration expires."""
  env, mock_entity, asset_cfg, impulse = _make_impulse_env(
    device, num_envs=2, num_bodies=3, body_ids=[1], cooldown_s=(10.0, 10.0)
  )

  # Skip the initial cooldown so the first call triggers immediately;
  # the trigger/sustain/expire cycle is what's under test here.
  impulse._interval_time_left[:] = 0.0

  impulse(
    env,
    None,
    force_range=(10.0, 10.0),
    torque_range=(0.0, 0.0),
    duration_s=(0.05, 0.05),
    cooldown_s=(10.0, 10.0),
    asset_cfg=asset_cfg,
  )
  assert impulse._active.any()
  mock_entity.write_external_wrench_to_sim.assert_called()

  def step():
    impulse(
      env,
      None,
      force_range=(10.0, 10.0),
      torque_range=(0.0, 0.0),
      duration_s=(0.05, 0.05),
      cooldown_s=(10.0, 10.0),
      asset_cfg=asset_cfg,
    )

  # Step a few times (duration is 0.05s, step_dt is 0.02s).
  mock_entity.write_external_wrench_to_sim.reset_mock()
  step()  # t=0.02, remaining=0.03
  assert impulse._active.all()

  step()  # t=0.04, remaining=0.01
  assert impulse._active.all()

  step()  # t=0.06, remaining=-0.01 -> cleared
  assert not impulse._active.any()

  # Verify write_external_wrench_to_sim was called with zeros to clear.
  clear_call = None
  for call in mock_entity.write_external_wrench_to_sim.call_args_list:
    forces = call[0][0]
    if torch.all(forces == 0):
      clear_call = call
  assert clear_call is not None


def test_apply_body_impulse_with_offset(device):
  """body_point_offset, in the body frame, adds cross-product torque."""
  env, mock_entity, asset_cfg, impulse = _make_impulse_env(
    device, num_envs=1, num_bodies=1, body_ids=[0]
  )

  impulse(
    env,
    None,
    force_range=(1.0, 1.0),
    torque_range=(0.0, 0.0),
    duration_s=(1.0, 1.0),
    cooldown_s=(0.0, 0.0),
    asset_cfg=asset_cfg,
    body_point_offset=(0.0, 0.0, 0.5),
  )

  call_args = mock_entity.write_external_wrench_to_sim.call_args
  forces = call_args[0][0]
  torques = call_args[0][1]

  # force is (1,1,1) applied at offset (0,0,0.5).
  # cross((0,0,0.5), (1,1,1)) = (-0.5, 0.5, 0)
  offset = torch.tensor([0.0, 0.0, 0.5], device=device)
  expected_extra_torque = torch.cross(offset.unsqueeze(0), forces.squeeze(0), dim=-1)
  torch.testing.assert_close(
    torques.squeeze(0), expected_extra_torque, atol=1e-5, rtol=1e-5
  )


def test_apply_body_impulse_reset_clears(device):
  """reset() zeros forces and resets internal state."""
  env, mock_entity, asset_cfg, impulse = _make_impulse_env(
    device, num_envs=2, num_bodies=1, body_ids=[0]
  )

  # Trigger impulse.
  impulse(
    env,
    None,
    force_range=(10.0, 10.0),
    torque_range=(0.0, 0.0),
    duration_s=(1.0, 1.0),
    cooldown_s=(0.0, 0.0),
    asset_cfg=asset_cfg,
  )
  assert impulse._active.all()

  # Reset env 0.
  mock_entity.write_external_wrench_to_sim.reset_mock()
  impulse.reset(env_ids=torch.tensor([0], device=device))

  assert not impulse._active[0]
  assert impulse._active[1]
  assert impulse._time_remaining[0] == 0.0

  # Verify clearing write was called for env 0.
  mock_entity.write_external_wrench_to_sim.assert_called_once()
  call_args = mock_entity.write_external_wrench_to_sim.call_args
  env_ids_arg = call_args[1]["env_ids"]
  assert len(env_ids_arg) == 1
  assert env_ids_arg[0].item() == 0


def test_apply_body_impulse_initial_cooldown(device):
  """The first call after init/reset enters cooldown, not an immediate impulse.

  Regression test for #973.
  """
  env, mock_entity, asset_cfg, impulse = _make_impulse_env(
    device, num_envs=1, num_bodies=1, body_ids=[0], cooldown_s=(0.05, 0.05)
  )

  def step():
    impulse(
      env,
      None,
      force_range=(10.0, 10.0),
      torque_range=(0.0, 0.0),
      duration_s=(1.0, 1.0),
      cooldown_s=(0.05, 0.05),  # ~2.5 steps at dt=0.02
      asset_cfg=asset_cfg,
    )

  # First two steps consume the sampled cooldown; impulse must not fire yet.
  step()
  assert not impulse._active.any()
  step()
  assert not impulse._active.any()

  # Third step crosses the cooldown boundary and triggers.
  step()
  assert impulse._active.all()

  # Reset re-enters cooldown: next step should not immediately re-trigger.
  impulse.reset(env_ids=torch.tensor([0], device=device))
  assert not impulse._active.any()
  step()
  assert not impulse._active.any()


# ===========================================================================
# Section 5: Recomputation vs a recompiled MjSpec
# ===========================================================================

SET_CONST_XML = """
<mujoco>
  <asset><hfield name="h" nrow="3" ncol="3" size="1 1 0.1 0.1"/></asset>
  <worldbody>
    <geom type="hfield" hfield="h"/>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <inertial pos="0.01 0.02 0" quat="1 0.1 0 0" mass="5" diaginertia="0.1 0.2 0.3"/>
      <camera name="cam" mode="trackcom" pos="0 -1 0.5" quat="1 0.3 0 0"/>
      <light name="light" mode="trackcom" pos="0 0 1" dir="0 0.2 -1"/>
      <site name="s0" pos="0.1 0 0.1"/>
      <body name="link1" pos="0.2 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" armature="0.05"/>
        <inertial pos="0.05 0 0.01" quat="1 0 0.2 0" mass="1"
          diaginertia="0.01 0.02 0.015"/>
        <geom name="wrap" type="cylinder" size="0.04 0.05" contype="0"
          conaffinity="0"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <inertial pos="0.03 0 0" mass="0.5" diaginertia="0.005 0.005 0.005"/>
          <site name="s1" pos="0.1 0 -0.05"/>
        </body>
      </body>
      <body name="link3" pos="-0.2 0 0">
        <joint name="j3" type="hinge" axis="0 1 0"/>
        <inertial pos="-0.03 0 0" mass="0.5" diaginertia="0.005 0.005 0.005"/>
        <site name="s2" pos="-0.05 0 0"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t"><site site="s0"/><geom geom="wrap"/><site site="s1"/></spatial>
    <fixed name="tf" armature="0.02">
      <joint joint="j2" coef="1"/><joint joint="j3" coef="-1"/>
    </fixed>
  </tendon>
  <equality><connect body1="link2" body2="link3" anchor="0.1 0 0"/></equality>
  <actuator>
    <position name="a1" joint="j1" kp="10" gear="2" armature="0.01"/>
    <motor name="a2" tendon="t" gear="1.5"/>
    <motor name="a3" site="s2" gear="0 0 0 0 1 0"/>
  </actuator>
</mujoco>
"""


def _set(obj, attr, value):
  setattr(obj, attr, value)


def _set_anchor(equality, anchor):
  data = list(equality.data)
  data[:3] = anchor
  equality.data = data


# One spec edit per model field whose change requires set_const.
SPEC_EDITS = {
  "body_mass": lambda s: _set(s.body("link1"), "mass", 2.0),
  "body_ipos": lambda s: _set(s.body("link1"), "ipos", [0.08, 0.01, 0.02]),
  "body_pos": lambda s: _set(s.body("link1"), "pos", [0.25, 0.03, 0.0]),
  "body_quat": lambda s: _set(s.body("link1"), "quat", [0.98, 0.1, 0.17, 0.0]),
  "qpos0": lambda s: _set(s.joint("j1"), "ref", 0.3),
  "jnt_pos": lambda s: _set(s.joint("j1"), "pos", [0.02, 0.0, 0.01]),
  "jnt_axis": lambda s: _set(s.joint("j1"), "axis", [0.0, 0.6, 0.8]),
  "body_inertia": lambda s: _set(s.body("link1"), "inertia", [0.02, 0.03, 0.025]),
  "body_iquat": lambda s: _set(s.body("link1"), "iquat", [0.95, 0.0, 0.0, 0.3]),
  "dof_armature": lambda s: _set(s.joint("j1"), "armature", 0.2),
  "tendon_armature": lambda s: _set(s.tendon("tf"), "armature", 0.1),
  "actuator_armature": lambda s: _set(s.actuator("a1"), "armature", 0.1),
  "actuator_gear": lambda s: _set(s.actuator("a1"), "gear", [3.0, 0, 0, 0, 0, 0]),
  "eq_data": lambda s: _set_anchor(s.equalities[0], [0.12, 0.01, 0.0]),
  "cam_pos": lambda s: _set(s.camera("cam"), "pos", [0.1, -1.2, 0.4]),
  "cam_quat": lambda s: _set(s.camera("cam"), "quat", [1.0, 0.0, 0.3, 0.0]),
  "light_pos": lambda s: _set(s.light("light"), "pos", [0.2, 0.0, 1.2]),
  "light_dir": lambda s: _set(s.light("light"), "dir", [0.3, 0.0, -1.0]),
  "geom_size": lambda s: _set(s.geom("wrap"), "size", [0.06, 0.05, 0.0]),
  "geom_pos": lambda s: _set(s.geom("wrap"), "pos", [0.02, 0.0, 0.03]),
  "geom_quat": lambda s: _set(s.geom("wrap"), "quat", [1.0, 0.4, 0.0, 0.0]),
  "site_pos": lambda s: _set(s.site("s1"), "pos", [0.12, 0.0, -0.08]),
  "site_quat": lambda s: _set(s.site("s2"), "quat", [1.0, 0.0, 0.0, 0.4]),
}

# Every field mujoco_warp.set_const writes.
SET_CONST_OUTPUTS = (
  "body_subtreemass",
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  "tendon_length0",
  "eq_data",
  "cam_pos0",
  "cam_poscom0",
  "cam_mat0",
  "light_pos0",
  "light_poscom0",
  "light_dir0",
  "dof_invweight0",
  "body_invweight0",
  "tendon_invweight0",
  "actuator_acc0",
)


def test_model_dependent_groups_are_skipped_when_unused():
  """Without tendons, site actuators, or tracking cameras, these fields stay cheap."""
  model = mujoco.MjModel.from_xml_string(ROBOT_XML)
  assert recompute_groups(("geom_size",), model) == SetConstGroup.FIXED
  for field in ("geom_pos", "site_pos", "site_quat", "cam_pos", "light_pos"):
    assert recompute_groups((field,), model) == SetConstGroup.NONE


@pytest.mark.parametrize("field", sorted(SPEC_EDITS))
def test_recompute_matches_recompiled_spec(device, field):
  """A DR write to world 0 recomputes exactly what recompiling the spec does,
  and world 1 keeps the original values."""
  spec = mujoco.MjSpec.from_string(SET_CONST_XML)
  original = spec.compile()
  SPEC_EDITS[field](spec)
  edited = spec.compile()

  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=original, device=device)

  @requires_model_fields(field)
  def write_world_0(env, env_ids):
    del env_ids
    target = getattr(env.sim.model, field)
    value = torch.as_tensor(getattr(edited, field), dtype=target.dtype)
    if field == "eq_data":  # Only the anchor is user-set; KIN0 derives the rest.
      value = torch.as_tensor(original.eq_data, dtype=target.dtype)
      value[:, :3] = torch.as_tensor(edited.eq_data[:, :3])
    target[0] = value.reshape(target[0].shape).to(device)

  env = Mock(num_envs=2, device=device, scene={}, sim=sim)
  cfg = {"dr": EventTermCfg(mode="startup", func=write_world_0, params={})}
  manager = EventManager(cfg, env)
  sim.expand_model_fields(manager.domain_randomization_fields)
  manager.apply("startup")

  for name in (field, *SET_CONST_OUTPUTS):
    arr = getattr(sim.model, name)
    for world, ref in ((0, edited), (1, original)):
      actual = arr[min(world, arr.shape[0] - 1)].cpu().flatten()
      expected = torch.as_tensor(getattr(ref, name), dtype=actual.dtype).flatten()
      torch.testing.assert_close(
        actual, expected, atol=1e-5, rtol=1e-4, msg=f"{name}, world {world}"
      )


# ===========================================================================
# Section: interval timer reset (issue: reset iterated the class-only list)
# ===========================================================================


def _noop_interval_event(env, env_ids):
  del env, env_ids


class _ClassIntervalEvent:
  """Class-based interval event with a reset hook."""

  def __init__(self, cfg, env):
    del cfg, env

  def reset(self, env_ids=None):
    del env_ids

  def __call__(self, env, env_ids):
    del env, env_ids


def _make_mock_env(device, num_envs=4):
  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.scene = {}
  env.sim = Mock()
  return env


def test_interval_timer_resets_for_function_terms(device):
  """Function-based interval terms get their per-env timer resampled on reset."""
  env = _make_mock_env(device)
  cfg = {
    "push": EventTermCfg(
      mode="interval", func=_noop_interval_event, interval_range_s=(5.0, 10.0)
    ),
  }
  manager = EventManager(cfg, env)

  timer = manager._interval_term_time_left[0]
  # Zero the reset env's timer so the in-range assertion below can only pass
  # if reset actually resampled it (deterministic, not clone-and-compare).
  timer[1] = 0.0
  before = timer.clone()
  reset_ids = torch.tensor([1], dtype=torch.int64, device=device)
  manager.reset(env_ids=reset_ids)

  assert 5.0 <= timer[1].item() <= 10.0
  untouched = torch.tensor([0, 2, 3], device=device)
  assert torch.equal(timer[untouched], before[untouched])


def test_interval_timer_reset_with_mixed_terms(device):
  """Mixed function and class interval terms reset the right timer slots."""
  env = _make_mock_env(device)
  cfg = {
    "push": EventTermCfg(
      mode="interval", func=_noop_interval_event, interval_range_s=(5.0, 10.0)
    ),
    "impulse": EventTermCfg(
      mode="interval", func=_ClassIntervalEvent, interval_range_s=(100.0, 200.0)
    ),
  }
  manager = EventManager(cfg, env)

  push_timer = manager._interval_term_time_left[0]
  impulse_timer = manager._interval_term_time_left[1]
  # Zero both timers for the reset env so the in-range assertions can only
  # pass if reset resampled each slot (the init draw is from the same range,
  # so comparing against it would be vacuous).
  push_timer[1] = 0.0
  impulse_timer[1] = 0.0
  reset_ids = torch.tensor([1], dtype=torch.int64, device=device)
  manager.reset(env_ids=reset_ids)

  # Each timer slot is resampled from its own term's range.
  assert 5.0 <= push_timer[1].item() <= 10.0
  assert 100.0 <= impulse_timer[1].item() <= 200.0


def test_interval_timer_reset_all_envs(device):
  """reset(env_ids=None) resamples every env's timer."""
  env = _make_mock_env(device)
  cfg = {
    "push": EventTermCfg(
      mode="interval", func=_noop_interval_event, interval_range_s=(5.0, 10.0)
    ),
  }
  manager = EventManager(cfg, env)

  timer = manager._interval_term_time_left[0]
  timer[:] = 0.0
  manager.reset(env_ids=None)
  assert ((timer >= 5.0) & (timer <= 10.0)).all()
