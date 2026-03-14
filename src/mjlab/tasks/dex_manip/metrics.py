from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

from .mdp.numerics import sanitize_to_range

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_OBJECT_CFG = SceneEntityCfg("object", geom_names=("object_geom",))
_DEFAULT_OBJECT_BODY_CFG = SceneEntityCfg("object")


def _resolve_object_geom_id(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg,
) -> int:
  cache_key = f"_dex_manip_object_geom_id::{object_cfg.name}::{object_cfg.geom_names}"
  cached = getattr(env, cache_key, None)
  if cached is not None:
    return int(cached)

  object_cfg.resolve(env.scene)
  geom_ids = env.scene[object_cfg.name].indexing.geom_ids[object_cfg.geom_ids]
  geom_id = int(geom_ids[0].item())
  setattr(env, cache_key, geom_id)
  return geom_id


def _resolve_mesh_id(env: ManagerBasedRlEnv, mesh_name: str) -> int:
  cache_key = f"_dex_manip_mesh_id::{mesh_name}"
  cached = getattr(env, cache_key, None)
  if cached is not None:
    return int(cached)

  available_names = [mesh.name for mesh in env.scene.spec.meshes if mesh.name is not None]
  exact_name_to_id = {name: idx for idx, name in enumerate(available_names)}

  if mesh_name in exact_name_to_id:
    mesh_id = exact_name_to_id[mesh_name]
  else:
    suffix_matches = [
      idx
      for idx, full_name in enumerate(available_names)
      if full_name.rsplit("/", 1)[-1] == mesh_name
    ]
    if len(suffix_matches) == 1:
      mesh_id = suffix_matches[0]
    elif len(suffix_matches) > 1:
      matched = [available_names[idx] for idx in suffix_matches]
      raise ValueError(
        f"Mesh name {mesh_name!r} is ambiguous. Matches: {matched}. "
        "Use a fully qualified mesh name."
      )
    else:
      raise ValueError(
        f"Unknown mesh name {mesh_name!r}. Available: {sorted(available_names)}"
      )

  setattr(env, cache_key, mesh_id)
  return mesh_id


def _object_dataid_per_env(
  env: ManagerBasedRlEnv,
  object_cfg: SceneEntityCfg,
) -> torch.Tensor:
  geom_id = _resolve_object_geom_id(env, object_cfg)
  dataid = env.sim.model.geom_dataid
  if dataid.ndim == 2:
    return dataid[:, geom_id].to(dtype=torch.int64, device=env.device)
  value = int(dataid[geom_id].item())
  return torch.full((env.num_envs,), value, dtype=torch.int64, device=env.device)


def reward_mean(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.reward_buf


def reward_for_mesh(
  env: ManagerBasedRlEnv,
  mesh_name: str,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  per_env_dataid = _object_dataid_per_env(env, object_cfg)
  mesh_id = _resolve_mesh_id(env, mesh_name)
  mask = per_env_dataid == mesh_id
  if not torch.any(mask):
    value = torch.tensor(0.0, device=env.device, dtype=env.reward_buf.dtype)
  else:
    value = env.reward_buf[mask].mean()
  return torch.full_like(env.reward_buf, value)


def object_linear_speed(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_BODY_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=-1)
  return sanitize_to_range(speed, 0.0, 1e6, nan_default=0.0)


class object_rotation_progress:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
    self._asset: Entity = env.scene[self._asset_cfg.name]
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    self._init_pos_w[env_ids] = pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_BODY_CFG,
    target_yaw_rate: float = 0.20,
    position_threshold: float = 0.02,
    tilt_threshold: float = 0.35,
  ) -> torch.Tensor:
    del env, asset_cfg
    yaw_rate = -self._asset.data.root_link_ang_vel_w[:, 2]
    yaw_score = torch.clamp(yaw_rate / max(target_yaw_rate, 1e-6), min=0.0, max=1.0)

    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(torch.stack([roll_error, pitch_error], dim=-1), dim=-1)

    pos_score = torch.clamp(1.0 - pos_error / max(position_threshold, 1e-6), min=0.0, max=1.0)
    tilt_score = torch.clamp(1.0 - tilt_error / max(tilt_threshold, 1e-6), min=0.0, max=1.0)
    progress = sanitize_to_range(yaw_score * pos_score * tilt_score, 0.0, 1.0, nan_default=0.0)
    return torch.where(self._has_init, progress, torch.zeros_like(progress))


class object_rotation_success:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
    self._asset: Entity = env.scene[self._asset_cfg.name]
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    self._init_pos_w[env_ids] = pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_BODY_CFG,
    target_yaw_rate: float = 0.20,
    position_threshold: float = 0.02,
    tilt_threshold: float = 0.35,
  ) -> torch.Tensor:
    del env, asset_cfg
    yaw_rate = -self._asset.data.root_link_ang_vel_w[:, 2]
    yaw_ok = yaw_rate >= target_yaw_rate

    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(torch.stack([roll_error, pitch_error], dim=-1), dim=-1)

    stable = (pos_error <= position_threshold) & (tilt_error <= tilt_threshold)
    success = self._has_init & yaw_ok & stable
    return success.float()


class object_pose_rp_error_from_reset:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
    self._component = cast(Literal["position", "tilt"], cfg.params.get("component", "position"))
    if self._component not in ("position", "tilt"):
      raise ValueError(
        f"Unknown component '{self._component}'. Expected 'position' or 'tilt'."
      )
    self._asset: Entity = env.scene[self._asset_cfg.name]
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    self._init_pos_w[env_ids] = pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    component: Literal["position", "tilt"] = "position",
    asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_BODY_CFG,
  ) -> torch.Tensor:
    del env, component, asset_cfg
    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(torch.stack([roll_error, pitch_error], dim=-1), dim=-1)

    metric = pos_error if self._component == "position" else tilt_error
    metric = sanitize_to_range(metric, 0.0, 1e6, nan_default=0.0)
    return torch.where(self._has_init, metric, torch.zeros_like(metric))
