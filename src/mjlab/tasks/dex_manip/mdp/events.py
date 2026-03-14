from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_gaussian, sample_log_uniform, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_OBJECT_CFG = SceneEntityCfg("object", geom_names=("object_geom",))
_DEFAULT_HAND_GEOM_CFG = SceneEntityCfg("robot", geom_names=(".*",))


def _resolve_env_ids(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None
) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if isinstance(env_ids, slice):
    start, stop, step = env_ids.indices(env.num_envs)
    return torch.arange(start, stop, step, device=env.device, dtype=torch.int)
  return env_ids.to(env.device, dtype=torch.int)


def _resolve_geom_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  if asset_cfg.geom_ids is None:
    raise ValueError("asset_cfg.geom_ids must be resolved.")
  if isinstance(asset_cfg.geom_ids, list):
    return torch.tensor(asset_cfg.geom_ids, device=env.device, dtype=torch.int)
  if isinstance(asset_cfg.geom_ids, torch.Tensor):
    return asset_cfg.geom_ids.to(env.device, dtype=torch.int)
  if isinstance(asset_cfg.geom_ids, slice):
    entity: Entity = env.scene[asset_cfg.name]
    return entity.indexing.geom_ids[asset_cfg.geom_ids]
  return torch.tensor([asset_cfg.geom_ids], device=env.device, dtype=torch.int)


def _resolve_body_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  if asset_cfg.body_ids is None:
    raise ValueError("asset_cfg.body_ids must be resolved.")
  if isinstance(asset_cfg.body_ids, list):
    return torch.tensor(asset_cfg.body_ids, device=env.device, dtype=torch.int)
  if isinstance(asset_cfg.body_ids, torch.Tensor):
    return asset_cfg.body_ids.to(env.device, dtype=torch.int)
  if isinstance(asset_cfg.body_ids, slice):
    entity: Entity = env.scene[asset_cfg.name]
    return entity.indexing.body_ids[asset_cfg.body_ids]
  return torch.tensor([asset_cfg.body_ids], device=env.device, dtype=torch.int)


def _sample_distribution(
  distribution: str,
  lower: float,
  upper: float,
  shape: tuple[int, ...],
  device: str,
) -> torch.Tensor:
  low = torch.tensor(lower, device=device, dtype=torch.float32)
  high = torch.tensor(upper, device=device, dtype=torch.float32)
  if distribution == "uniform":
    return sample_uniform(low, high, shape, device=device)
  if distribution == "log_uniform":
    return sample_log_uniform(low, high, shape, device=device)
  if distribution == "gaussian":
    return sample_gaussian(low, high, shape, device=device)
  raise ValueError(f"Unsupported distribution: {distribution}")


@requires_model_fields("geom_friction")
def randomize_shared_contact_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  friction_range: tuple[float, float],
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_GEOM_CFG,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  axes: tuple[int, ...] = (0,),
) -> None:
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return

  hand_geom_ids = _resolve_geom_ids(env, hand_cfg)
  object_geom_ids = _resolve_geom_ids(env, object_cfg)
  if len(hand_geom_ids) == 0 or len(object_geom_ids) == 0:
    return

  friction = sample_uniform(
    friction_range[0],
    friction_range[1],
    (len(env_ids), 1),
    device=env.device,
  )

  for axis in axes:
    env.sim.model.geom_friction[env_ids[:, None], hand_geom_ids[None, :], axis] = friction
    env.sim.model.geom_friction[env_ids[:, None], object_geom_ids[None, :], axis] = friction


@requires_model_fields("body_mass")
def randomize_body_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  mass_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
  distribution: str = "uniform",
  operation: str = "scale",
  min_mass: float = 1e-5,
) -> None:
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  body_ids = _resolve_body_ids(env, asset_cfg)
  if len(body_ids) == 0:
    return

  current = env.sim.model.body_mass[env_ids[:, None], body_ids[None, :]]
  if operation == "scale":
    base = env.sim.get_default_field("body_mass")[body_ids].unsqueeze(0).expand_as(current)
    scale = _sample_distribution(
      distribution=distribution,
      lower=mass_range[0],
      upper=mass_range[1],
      shape=current.shape,
      device=env.device,
    )
    out = base * scale
  elif operation == "abs":
    out = _sample_distribution(
      distribution=distribution,
      lower=mass_range[0],
      upper=mass_range[1],
      shape=current.shape,
      device=env.device,
    )
  elif operation == "add":
    base = env.sim.get_default_field("body_mass")[body_ids].unsqueeze(0).expand_as(current)
    delta = _sample_distribution(
      distribution=distribution,
      lower=mass_range[0],
      upper=mass_range[1],
      shape=current.shape,
      device=env.device,
    )
    out = base + delta
  else:
    raise ValueError(f"Unsupported operation '{operation}'.")

  env.sim.model.body_mass[env_ids[:, None], body_ids[None, :]] = torch.clamp(
    out, min=min_mass
  )
