"""Domain randomization functions for camera fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._core import (
  _DEFAULT_ASSET_CFG,
  Ranges,
  _randomize_model_field,
  _randomize_quat_field,
)
from ._types import Distribution, Operation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("cam_fovy")
def cam_fovy(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize camera field-of-view (vertical, degrees)."""
  _randomize_model_field(
    env,
    env_ids,
    "cam_fovy",
    entity_type="camera",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
  )


@requires_model_fields("cam_pos")
def cam_pos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize camera positions."""
  _randomize_model_field(
    env,
    env_ids,
    "cam_pos",
    entity_type="camera",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2],
  )


@requires_model_fields("cam_quat")
def cam_quat(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  roll_range: tuple[float, float] = (0.0, 0.0),
  pitch_range: tuple[float, float] = (0.0, 0.0),
  yaw_range: tuple[float, float] = (0.0, 0.0),
  distribution: Distribution | str = "uniform",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize camera orientation by composing an RPY perturbation.

  Ranges are in radians. The sampled perturbation is composed with the default
  quaternion (not the current one), so repeated calls do not accumulate. The result is
  always a valid unit quaternion.
  """
  _randomize_quat_field(
    env,
    env_ids,
    "cam_quat",
    entity_type="camera",
    roll_range=roll_range,
    pitch_range=pitch_range,
    yaw_range=yaw_range,
    distribution=distribution,
    asset_cfg=asset_cfg,
  )


@requires_model_fields("cam_intrinsic")
def cam_intrinsic(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize camera intrinsic parameters."""
  _randomize_model_field(
    env,
    env_ids,
    "cam_intrinsic",
    entity_type="camera",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2, 3],
  )
