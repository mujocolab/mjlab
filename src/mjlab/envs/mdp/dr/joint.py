"""Domain randomization functions for joint and DOF fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

from ._core import (
  _DEFAULT_ASSET_CFG,
  Ranges,
  _randomize_model_field,
)
from ._types import Distribution, Operation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("dof_damping")
def joint_damping(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize joint damping (dof_damping)."""
  _randomize_model_field(
    env,
    env_ids,
    "dof_damping",
    entity_type="dof",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    use_address=True,
  )


# Raw alias.
dof_damping = joint_damping


@requires_model_fields("dof_armature", recompute=RecomputeLevel.set_const_0)
def joint_armature(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize joint armature. Triggers ``set_const_0``."""
  _randomize_model_field(
    env,
    env_ids,
    "dof_armature",
    entity_type="dof",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    use_address=True,
  )


# Raw alias.
dof_armature = joint_armature


@requires_model_fields("dof_frictionloss")
def joint_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize joint friction loss (dof_frictionloss)."""
  _randomize_model_field(
    env,
    env_ids,
    "dof_frictionloss",
    entity_type="dof",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    use_address=True,
  )


# Raw alias.
dof_frictionloss = joint_friction


@requires_model_fields("jnt_stiffness")
def joint_stiffness(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize joint stiffness (jnt_stiffness)."""
  _randomize_model_field(
    env,
    env_ids,
    "jnt_stiffness",
    entity_type="joint",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
  )


# Raw alias.
jnt_stiffness = joint_stiffness


@requires_model_fields("jnt_range")
def joint_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize joint limits (jnt_range)."""
  _randomize_model_field(
    env,
    env_ids,
    "jnt_range",
    entity_type="joint",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
  )


# Raw alias.
jnt_range = joint_limits


@requires_model_fields("qpos0", recompute=RecomputeLevel.set_const_0)
def joint_default_pos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "add",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize default joint positions (qpos0). Triggers ``set_const_0``."""
  _randomize_model_field(
    env,
    env_ids,
    "qpos0",
    entity_type="joint",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    use_address=True,
  )


# Raw alias.
qpos0 = joint_default_pos


def encoder_bias(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  bias_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize encoder bias to simulate joint encoder calibration errors."""
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    num_joints = asset.num_joints
    joint_ids_tensor = torch.arange(num_joints, device=env.device)
  else:
    joint_ids_tensor = torch.tensor(joint_ids, device=env.device)

  num_joints = len(joint_ids_tensor)
  bias_samples = sample_uniform(
    torch.tensor(bias_range[0], device=env.device),
    torch.tensor(bias_range[1], device=env.device),
    (len(env_ids), num_joints),
    env.device,
  )

  if isinstance(joint_ids, slice):
    asset.data.encoder_bias[env_ids] = bias_samples
  else:
    asset.data.encoder_bias[env_ids[:, None], joint_ids_tensor] = bias_samples
