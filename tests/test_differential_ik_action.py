"""Tests for differential IK action space."""

from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import BuiltinPositionActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import DifferentialIKAction, DifferentialIKActionCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_mul

ARM_XML = """\
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="base_geom" type="cylinder" size="0.05 0.02" mass="1.0"
            contype="0" conaffinity="0"/>
      <body name="link1" pos="0 0 0.02">
        <joint name="joint1" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.3"
              size="0.02" mass="0.5" contype="0" conaffinity="0"/>
        <body name="link2" pos="0 0 0.3">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0 0.3"
                size="0.02" mass="0.5" contype="0" conaffinity="0"/>
          <body name="ee" pos="0 0 0.3">
            <joint name="joint3" type="hinge" axis="0 1 0"
                   range="-3.14 3.14"/>
            <geom name="ee_geom" type="sphere" size="0.03" mass="0.1"
                  contype="0" conaffinity="0"/>
            <site name="ee_site" pos="0 0 0.05"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

TIGHT_ARM_XML = """\
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="base_geom" type="cylinder" size="0.05 0.02" mass="1.0"
            contype="0" conaffinity="0"/>
      <body name="link1" pos="0 0 0.02">
        <joint name="joint1" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.3"
              size="0.02" mass="0.5" contype="0" conaffinity="0"/>
        <body name="link2" pos="0 0 0.3">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0 0 0.3"
                size="0.02" mass="0.5" contype="0" conaffinity="0"/>
          <body name="ee" pos="0 0 0.3">
            <joint name="joint3" type="hinge" axis="0 1 0"
                   range="-0.5 0.5"/>
            <geom name="ee_geom" type="sphere" size="0.03" mass="0.1"
                  contype="0" conaffinity="0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

NUM_ENVS = 4


@pytest.fixture
def device():
  return get_test_device()


def _make_entity(device, xml=ARM_XML):
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("joint.*",),
          transmission_type=TransmissionType.JOINT,
          stiffness=1.0,
          damping=1.0,
          effort_limit=1.0,
        ),
      )
    ),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim_cfg = SimulationCfg(mujoco=MujocoCfg(gravity=(0, 0, 0)))
  sim = Simulation(num_envs=NUM_ENVS, cfg=sim_cfg, model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


def _make_env(entity, sim, device):
  env = Mock(spec=ManagerBasedRlEnv)
  env.num_envs = NUM_ENVS
  env.device = device
  env.scene = {"robot": entity}
  env.sim = sim
  return env


@pytest.mark.parametrize(
  "orientation_weight,relative,expected_dim",
  [
    (0.0, True, 3),
    (0.0, False, 3),
    (1.0, True, 6),
    (1.0, False, 7),
  ],
)
def test_action_dim(device, orientation_weight, relative, expected_dim):
  entity, sim = _make_entity(device)
  env = _make_env(entity, sim, device)
  cfg = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    frame_name="ee",
    frame_type="body",
    orientation_weight=orientation_weight,
    use_relative_mode=relative,
  )
  action = cfg.build(env)
  assert action.action_dim == expected_dim


@pytest.mark.parametrize(
  "frame_type,frame_name",
  [
    ("body", "ee"),
    ("site", "ee_site"),
    ("geom", "ee_geom"),
  ],
)
def test_frame_types(device, frame_type, frame_name):
  entity, sim = _make_entity(device)
  env = _make_env(entity, sim, device)
  cfg = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    frame_name=frame_name,
    frame_type=frame_type,
    orientation_weight=0.0,
  )
  action: DifferentialIKAction = cfg.build(env)
  assert action.action_dim == 3

  pos, quat = action._get_frame_pose()
  assert pos.shape == (NUM_ENVS, 3)
  assert quat.shape == (NUM_ENVS, 4)


def test_ik_convergence_position(device):
  """Verify IK converges kinematically by writing q+dq directly to qpos."""
  entity, sim = _make_entity(device)
  env = _make_env(entity, sim, device)
  cfg = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    frame_name="ee",
    frame_type="body",
    orientation_weight=0.0,
    use_relative_mode=False,
    damping=0.05,
  )
  action: DifferentialIKAction = cfg.build(env)

  initial_pos, _ = action._get_frame_pose()
  target = initial_pos.clone()
  target[:, 0] += 0.05

  action.process_actions(target)
  for _ in range(10):
    dq = action.compute_dq()
    q = entity.data.joint_pos[:, action._joint_ids] + dq
    entity.write_joint_position_to_sim(q, joint_ids=action._joint_ids)
    sim.forward()

  final_pos, _ = action._get_frame_pose()
  error = (final_pos - target).norm(dim=-1)
  assert (error < 0.003).all(), f"Position error too large: {error}"


def test_reset_clears_buffers(device):
  entity, sim = _make_entity(device)
  env = _make_env(entity, sim, device)
  cfg = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    frame_name="ee",
    frame_type="body",
  )
  action: DifferentialIKAction = cfg.build(env)

  actions = torch.randn(NUM_ENVS, 6, device=device)
  action.process_actions(actions)

  reset_ids = torch.tensor([0, 2], device=device)
  action.reset(env_ids=reset_ids)

  assert torch.all(action.raw_action[[0, 2]] == 0.0)
  assert not torch.all(action.raw_action[[1, 3]] == 0.0)


def test_joint_limits_respected(device):
  """IK with joint_limit_weight keeps joints within bounds."""
  entity, sim = _make_entity(device, xml=TIGHT_ARM_XML)
  env = _make_env(entity, sim, device)

  action_no_lim: DifferentialIKAction = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    frame_name="ee",
    frame_type="body",
    orientation_weight=0.0,
    use_relative_mode=False,
    damping=0.05,
  ).build(env)
  action_lim: DifferentialIKAction = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    frame_name="ee",
    frame_type="body",
    orientation_weight=0.0,
    use_relative_mode=False,
    damping=0.05,
    joint_limit_weight=10.0,
  ).build(env)

  initial_pos, _ = action_no_lim._get_frame_pose()
  target = initial_pos.clone()
  target[:, 0] += 0.15

  limits = entity.data.soft_joint_pos_limits[:, action_lim._joint_ids]
  for action in (action_no_lim, action_lim):
    entity.write_joint_position_to_sim(
      torch.zeros(NUM_ENVS, 3, device=device),
      joint_ids=action._joint_ids,
    )
    sim.forward()
    action.process_actions(target)
    for _ in range(10):
      dq = action.compute_dq()
      q = entity.data.joint_pos[:, action._joint_ids] + dq
      entity.write_joint_position_to_sim(q, joint_ids=action._joint_ids)
      sim.forward()

  q_lim = entity.data.joint_pos[:, action_lim._joint_ids]
  upper_viol = (q_lim - limits[..., 1]).clamp(min=0).sum()
  lower_viol = (limits[..., 0] - q_lim).clamp(min=0).sum()
  total_violation = upper_viol + lower_viol
  assert total_violation < 0.01, f"Joint limit violation too large: {total_violation}"


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _central_fd_position(action, entity, sim, q0, eps):
  """Central finite-difference position Jacobian."""
  nj = action._num_joints
  jacp_fd = torch.zeros(q0.shape[0], 3, nj, device=q0.device)
  for i in range(nj):
    q_plus = q0.clone()
    q_plus[:, i] += eps
    entity.write_joint_position_to_sim(q_plus, joint_ids=action._joint_ids)
    sim.forward()
    pos_plus = action._get_frame_pose()[0].clone()

    q_minus = q0.clone()
    q_minus[:, i] -= eps
    entity.write_joint_position_to_sim(q_minus, joint_ids=action._joint_ids)
    sim.forward()
    pos_minus = action._get_frame_pose()[0].clone()

    jacp_fd[:, :, i] = (pos_plus - pos_minus) / (2 * eps)
  return jacp_fd


def _central_fd_orientation(action, entity, sim, q0, quat0, eps):
  """Central finite-difference rotation Jacobian via axis-angle."""
  nj = action._num_joints
  jacr_fd = torch.zeros(q0.shape[0], 3, nj, device=q0.device)
  for i in range(nj):
    q_plus = q0.clone()
    q_plus[:, i] += eps
    entity.write_joint_position_to_sim(q_plus, joint_ids=action._joint_ids)
    sim.forward()
    _, quat_plus = action._get_frame_pose()
    q_diff_plus = quat_mul(quat_plus, _quat_conjugate(quat0))
    aa_plus = axis_angle_from_quat(q_diff_plus)

    q_minus = q0.clone()
    q_minus[:, i] -= eps
    entity.write_joint_position_to_sim(q_minus, joint_ids=action._joint_ids)
    sim.forward()
    _, quat_minus = action._get_frame_pose()
    q_diff_minus = quat_mul(quat_minus, _quat_conjugate(quat0))
    aa_minus = axis_angle_from_quat(q_diff_minus)

    jacr_fd[:, :, i] = (aa_plus - aa_minus) / (2 * eps)
  return jacr_fd


# Float32 central FD: truncation O(eps^2) ~ 1e-6, round-off O(u/eps) ~ 1e-4.
_FD_EPS = 1e-3
_FD_ATOL = 1e-3


def test_jacobian_finite_difference_position(device):
  """Verify position Jacobian against central finite differences."""
  entity, sim = _make_entity(device)
  env = _make_env(entity, sim, device)
  cfg = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    frame_name="ee",
    frame_type="body",
    orientation_weight=0.0,
    use_relative_mode=False,
    damping=0.05,
  )
  action: DifferentialIKAction = cfg.build(env)

  q0 = 0.3 * torch.tensor([[0.5, -0.3, 0.8]] * NUM_ENVS, device=device)
  entity.write_joint_position_to_sim(q0, joint_ids=action._joint_ids)
  sim.forward()

  pos0 = action._get_frame_pose()[0].clone()
  action._point_torch[:] = pos0
  action._compute_jacobian()
  jacp_analytic = action._jacp_torch[:, :, action._joint_dof_ids].clone()

  jacp_fd = _central_fd_position(action, entity, sim, q0, _FD_EPS)

  assert torch.allclose(jacp_analytic, jacp_fd, atol=_FD_ATOL), (
    f"Position Jacobian mismatch.\n"
    f"Max error: {(jacp_analytic - jacp_fd).abs().max():.2e}"
  )


def test_jacobian_finite_difference_orientation(device):
  """Verify rotation Jacobian against central finite differences."""
  entity, sim = _make_entity(device)
  env = _make_env(entity, sim, device)
  cfg = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    frame_name="ee",
    frame_type="body",
    use_relative_mode=False,
    damping=0.05,
  )
  action: DifferentialIKAction = cfg.build(env)

  q0 = 0.3 * torch.tensor([[0.5, -0.3, 0.8]] * NUM_ENVS, device=device)
  entity.write_joint_position_to_sim(q0, joint_ids=action._joint_ids)
  sim.forward()

  _, quat0 = action._get_frame_pose()
  quat0 = quat0.clone()

  pos0 = action._get_frame_pose()[0].clone()
  action._point_torch[:] = pos0
  action._compute_jacobian()
  jacr_analytic = action._jacr_torch[:, :, action._joint_dof_ids].clone()

  jacr_fd = _central_fd_orientation(action, entity, sim, q0, quat0, _FD_EPS)

  assert torch.allclose(jacr_analytic, jacr_fd, atol=_FD_ATOL), (
    f"Rotation Jacobian mismatch.\n"
    f"Max error: {(jacr_analytic - jacr_fd).abs().max():.2e}"
  )


def test_posture_regularization(device):
  """Posture weight biases joints toward target when task error is zero."""
  entity, sim = _make_entity(device)
  env = _make_env(entity, sim, device)

  cfg = DifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint.*",),
    frame_name="ee",
    frame_type="body",
    orientation_weight=0.0,
    use_relative_mode=False,
    damping=0.05,
    posture_weight=0.1,
    posture_target={".*": 0.0},
  )
  action: DifferentialIKAction = cfg.build(env)

  q0 = torch.tensor([[0.3, -0.2, 0.5]] * NUM_ENVS, device=device)
  entity.write_joint_position_to_sim(q0, joint_ids=action._joint_ids)
  sim.forward()

  # Target = current frame position, so only posture term drives motion.
  frame_pos, _ = action._get_frame_pose()
  action.process_actions(frame_pos.clone())

  for _ in range(10):
    dq = action.compute_dq()
    q = entity.data.joint_pos[:, action._joint_ids] + dq
    entity.write_joint_position_to_sim(q, joint_ids=action._joint_ids)
    sim.forward()

  q_final = entity.data.joint_pos[:, action._joint_ids]
  assert q_final.abs().mean() < q0.abs().mean(), (
    f"Posture regularization did not pull toward target.\n"
    f"Initial mean |q|: {q0.abs().mean():.4f}, "
    f"Final mean |q|: {q_final.abs().mean():.4f}"
  )
