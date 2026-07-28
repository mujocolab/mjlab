"""Domain randomization functions for light fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._core import (
  _DEFAULT_ASSET_CFG,
  Ranges,
  _randomize_model_field,
)
from ._types import Distribution, Operation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("light_pos")
def light_pos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize light positions."""
  _randomize_model_field(
    env,
    env_ids,
    "light_pos",
    entity_type="light",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2],
  )


@requires_model_fields("light_dir")
def light_dir(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize light direction vectors."""
  _randomize_model_field(
    env,
    env_ids,
    "light_dir",
    entity_type="light",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2],
  )


@requires_model_fields("light_diffuse")
def light_diffuse(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize light diffuse RGB color."""
  _randomize_model_field(
    env,
    env_ids,
    "light_diffuse",
    entity_type="light",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2],
  )


@requires_model_fields("light_specular")
def light_specular(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize light specular RGB color."""
  _randomize_model_field(
    env,
    env_ids,
    "light_specular",
    entity_type="light",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2],
  )


@requires_model_fields("light_ambient")
def light_ambient(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize light ambient RGB color."""
  _randomize_model_field(
    env,
    env_ids,
    "light_ambient",
    entity_type="light",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2],
  )


@requires_model_fields("light_attenuation")
def light_attenuation(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize light attenuation coefficients."""
  _randomize_model_field(
    env,
    env_ids,
    "light_attenuation",
    entity_type="light",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2],
  )


@requires_model_fields("light_cutoff")
def light_cutoff(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize spot light cutoff angles in degrees."""
  _randomize_model_field(
    env,
    env_ids,
    "light_cutoff",
    entity_type="light",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
  )


@requires_model_fields("light_exponent")
def light_exponent(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize spot light angular falloff exponents."""
  _randomize_model_field(
    env,
    env_ids,
    "light_exponent",
    entity_type="light",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
  )
