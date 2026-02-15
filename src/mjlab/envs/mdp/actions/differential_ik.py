"""Differential IK action space for task-space control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import (
  apply_delta_pose,
  compute_pose_error,
  quat_from_matrix,
)
from mjlab.utils.string import resolve_expr

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DifferentialIKActionCfg(ActionTermCfg):
  """Configuration for differential IK action space.

  Converts task-space commands (position and/or orientation) into
  joint-space targets via damped least-squares (DLS) IK, applied
  every decimation substep.

  The action dimension is determined by the active objectives:

  - ``position_weight > 0, orientation_weight == 0`` -> 3D
  - ``orientation_weight > 0, use_relative_mode=True`` -> 6D
  - ``orientation_weight > 0, use_relative_mode=False`` -> 7D
  """

  actuator_names: tuple[str, ...] | list[str]
  """Actuator name expressions to resolve the controlled joints."""

  frame_type: Literal["body", "site", "geom"] = "body"
  """Element type of the target frame."""

  frame_name: str
  """Name of the target frame element on the entity."""

  use_relative_mode: bool = True
  """If True, actions are deltas applied to the current frame pose.
  If False, actions are absolute targets."""

  delta_pos_scale: float = 1.0
  """Scaling factor for position components in relative mode.

  Maps raw policy outputs to position deltas (meters).
  Ignored in absolute mode.
  """

  delta_ori_scale: float = 1.0
  """Scaling factor for orientation components in relative mode.

  Maps raw policy outputs to orientation deltas (radians).
  Ignored in absolute mode.
  """

  damping: float = 0.05
  """Damping coefficient (lambda) for the DLS pseudoinverse."""

  max_dq: float = 0.5
  """Maximum joint displacement per IK solve (rad/step)."""

  position_weight: float = 1.0
  """Weight for the position residual in the DLS system."""

  orientation_weight: float = 1.0
  """Weight for the orientation residual (0 = position-only mode)."""

  joint_limit_weight: float = 0.0
  """Weight for soft joint-limit residuals (0 = disabled).

  When positive, joint-limit violations are added as extra rows
  in the DLS system so the solver steers away from limits.
  """

  posture_weight: float = 0.0
  """Weight for posture regularization (0 = disabled).

  When positive, a residual ``w * (q_target - q_current)`` is added
  to bias the solver toward the posture target. Useful with
  redundant manipulators or position-only tracking to control
  null-space behaviour.
  """

  posture_target: dict[str, float] | None = None
  """Target joint positions for posture regularization.

  Maps joint name patterns to target values (same format as
  ``EntityCfg.InitialStateCfg.joint_pos``). Joints not listed
  default to their initial (qpos0) value. Only used when
  ``posture_weight > 0``.
  """

  def build(self, env: ManagerBasedRlEnv) -> DifferentialIKAction:
    return DifferentialIKAction(self, env)


class DifferentialIKAction(ActionTerm):
  """Converts task-space commands into joint position targets via damped least-squares
  IK each decimation substep.
  """

  cfg: DifferentialIKActionCfg
  _entity: Entity

  def __init__(self, cfg: DifferentialIKActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    joint_ids, _ = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._num_joints = len(joint_ids)
    self._joint_dof_ids = self._entity.indexing.joint_v_adr[self._joint_ids]

    self._frame_type = cfg.frame_type
    self._resolve_frame(cfg.frame_name)

    if cfg.orientation_weight > 0:
      self._action_dim = 6 if cfg.use_relative_mode else 7
    else:
      self._action_dim = 3
    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

    self._desired_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self._desired_quat = torch.zeros(self.num_envs, 4, device=self.device)
    self._desired_quat[:, 0] = 1.0

    # Joint-limit buffers.
    limits = self._entity.data.soft_joint_pos_limits
    self._joint_lower = limits[:, self._joint_ids, 0]
    self._joint_upper = limits[:, self._joint_ids, 1]

    # Posture regularization target.
    q_target = self._entity.data.default_joint_pos[:, self._joint_ids].clone()
    if cfg.posture_target is not None:
      joint_names = tuple(self._entity.joint_names[i] for i in joint_ids)
      overrides = resolve_expr(cfg.posture_target, joint_names)
      for j, val in enumerate(overrides):
        if val is not None:
          q_target[:, j] = val
    self._posture_target = q_target

    nworld = self.num_envs
    nv = self._env.sim.mj_model.nv

    with wp.ScopedDevice(self._env.sim.wp_device):
      self._jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._jacr_wp = wp.zeros((nworld, 3, nv), dtype=float)
      self._point_wp = wp.zeros(nworld, dtype=wp.vec3)
      self._body_wp = wp.zeros(nworld, dtype=wp.int32)
      self._body_wp.fill_(self._body_id)

    self._jacp_torch = wp.to_torch(self._jacp_wp)
    self._jacr_torch = wp.to_torch(self._jacr_wp)
    self._point_torch = wp.to_torch(self._point_wp).view(nworld, 3)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions

    frame_pos, frame_quat = self._get_frame_pose()
    if self._action_dim == 3:
      if self.cfg.use_relative_mode:
        self._desired_pos[:] = frame_pos + actions * self.cfg.delta_pos_scale
      else:
        self._desired_pos[:] = actions
      self._desired_quat[:] = frame_quat
    elif self._action_dim == 6:
      delta = actions.clone()
      delta[:, :3] *= self.cfg.delta_pos_scale
      delta[:, 3:] *= self.cfg.delta_ori_scale
      target_pos, target_quat = apply_delta_pose(frame_pos, frame_quat, delta)
      self._desired_pos[:] = target_pos
      self._desired_quat[:] = target_quat
    else:
      assert self._action_dim == 7
      self._desired_pos[:] = actions[:, :3]
      self._desired_quat[:] = actions[:, 3:7]

  def compute_dq(self) -> torch.Tensor:
    """Run one DLS IK step and return joint displacement.

    Returns:
      Joint displacement tensor of shape ``(num_envs, num_joints)``.
    """
    frame_pos, frame_quat = self._get_frame_pose()
    pos_error, rot_error = compute_pose_error(
      frame_pos, frame_quat, self._desired_pos, self._desired_quat
    )

    self._point_torch[:] = frame_pos
    self._compute_jacobian()
    jacp = self._jacp_torch[:, :, self._joint_dof_ids]
    jacr = self._jacr_torch[:, :, self._joint_dof_ids]

    w_pos = self.cfg.position_weight
    w_ori = self.cfg.orientation_weight
    w_lim = self.cfg.joint_limit_weight
    w_post = self.cfg.posture_weight
    lam = max(self.cfg.damping, 1e-6)

    # Joint-space normal equations: (J^T W J + λ²I) dq = J^T W dx.
    # Equivalent to task-space DLS but solves a smaller n×n system instead of the
    # (6 + 2n)-square task-space system.
    wp2, wo2 = w_pos * w_pos, w_ori * w_ori
    JTJ = wp2 * torch.einsum("bti,btj->bij", jacp, jacp) + wo2 * torch.einsum(
      "bti,btj->bij", jacr, jacr
    )
    JTdx = wp2 * torch.einsum("bti,bt->bi", jacp, pos_error) + wo2 * torch.einsum(
      "bti,bt->bi", jacr, rot_error
    )

    # Joint-limit penalty (diagonal contribution).
    q = self._entity.data.joint_pos[:, self._joint_ids]
    r_limit = (self._joint_upper - q).clamp(max=0) + (self._joint_lower - q).clamp(
      min=0
    )
    violated = (r_limit != 0).float()
    wl2 = w_lim * w_lim
    JTJ.diagonal(dim1=-2, dim2=-1).add_(wl2 * violated)
    JTdx.add_(wl2 * violated * r_limit)

    # Posture regularization (identity Jacobian contribution).
    r_posture = self._posture_target - q
    wpost2 = w_post * w_post
    JTJ.diagonal(dim1=-2, dim2=-1).add_(wpost2)
    JTdx.add_(wpost2 * r_posture)

    # Damping.
    JTJ.diagonal(dim1=-2, dim2=-1).add_(lam * lam)

    dq = torch.linalg.solve(JTJ, JTdx)
    return dq.clamp(-self.cfg.max_dq, self.cfg.max_dq)

  def apply_actions(self) -> None:
    dq = self.compute_dq()
    q_current = self._entity.data.joint_pos[:, self._joint_ids]
    q_target = q_current + dq
    self._entity.set_joint_position_target(q_target, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._desired_pos[env_ids] = 0.0
    self._desired_quat[env_ids] = 0.0
    self._desired_quat[env_ids, 0] = 1.0

  # Private

  def _resolve_frame(self, frame_name: str) -> None:
    """Resolve the frame element to its global ID and parent body ID."""
    if self._frame_type == "body":
      ids, _ = self._entity.find_bodies(frame_name)
      local_id = ids[0]
      self._frame_id = int(self._entity.indexing.body_ids[local_id].item())
      self._body_id = self._frame_id
    elif self._frame_type == "site":
      ids, _ = self._entity.find_sites(frame_name)
      local_id = ids[0]
      self._frame_id = int(self._entity.indexing.site_ids[local_id].item())
      self._body_id = int(self._env.sim.mj_model.site_bodyid[self._frame_id])
    elif self._frame_type == "geom":
      ids, _ = self._entity.find_geoms(frame_name)
      local_id = ids[0]
      self._frame_id = int(self._entity.indexing.geom_ids[local_id].item())
      self._body_id = int(self._env.sim.mj_model.geom_bodyid[self._frame_id])
    else:
      raise ValueError(f"Unknown frame_type: {self._frame_type}")

  def _get_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Return current frame (position, quaternion) from sim data."""
    data = self._env.sim.data
    if self._frame_type == "body":
      pos = data.xpos[:, self._frame_id]
      quat = data.xquat[:, self._frame_id]
    elif self._frame_type == "site":
      pos = data.site_xpos[:, self._frame_id]
      xmat = data.site_xmat[:, self._frame_id]
      quat = quat_from_matrix(xmat)
    else:
      assert self._frame_type == "geom"
      pos = data.geom_xpos[:, self._frame_id]
      xmat = data.geom_xmat[:, self._frame_id]
      quat = quat_from_matrix(xmat)
    return pos, quat

  def _compute_jacobian(self) -> None:
    """Compute the frame Jacobian."""
    with wp.ScopedDevice(self._env.sim.wp_device):
      mjwarp.jac(
        self._env.sim.wp_model,
        self._env.sim.wp_data,
        self._jacp_wp,
        self._jacr_wp,
        self._point_wp,
        self._body_wp,
      )
