"""Domain randomization functions for material fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._core import _DEFAULT_ASSET_CFG, Ranges, _randomize_model_field
from ._types import Distribution, Operation

if TYPE_CHECKING:
  import torch

  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("mat_rgba")
def mat_rgba(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize material RGBA color.

  In MuJoCo, a material's RGBA plays two roles depending on whether a texture is
  assigned:

  - **With texture**: ``mat_rgba`` is a multiplicative modulator. The rendered color
    equals ``texture_color * mat_rgba`` per channel. ``(1, 1, 1, 1)`` leaves the
    texture unchanged; values below 1 darken it.
  - **Without texture**: ``mat_rgba`` directly sets the material's surface color,
    similar to how :func:`~mjlab.envs.mdp.dr.geom_rgba` sets geom colors.

  Note: If a geom has no material assigned (``matid < 0``), its color is controlled by
  ``geom_rgba``, not ``mat_rgba``.

  Args:
    env: The environment instance.
    env_ids: Environment indices to randomize. ``None`` means all.
    ranges: Value range(s) for sampling.
    asset_cfg: Entity and material selection. Use
      ``SceneEntityCfg("entity", material_names=(...))`` to target specific materials.
    distribution: Sampling distribution.
    operation: How to combine sampled values with the base.
    axes: Which RGBA channels to randomize. Defaults to ``[0, 1, 2, 3]``.
    shared_random: If ``True``, all selected materials receive the same sampled
      value per environment.
  """
  _randomize_model_field(
    env,
    env_ids,
    "mat_rgba",
    entity_type="material",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1, 2, 3],
  )


@requires_model_fields("mat_emission")
def mat_emission(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize material emission.

  MuJoCo Warp's RGB renderer adds ``mat_emission * base_color`` per shaded
  pixel, so higher values make the material appear self-illuminated.
  """
  _randomize_model_field(
    env,
    env_ids,
    "mat_emission",
    entity_type="material",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
  )


@requires_model_fields("mat_specular")
def mat_specular(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize material specular reflection strength.

  MuJoCo stores specular in ``[0, 1]``. MuJoCo Warp uses it to scale the
  specular component of RGB rendering.
  """
  _randomize_model_field(
    env,
    env_ids,
    "mat_specular",
    entity_type="material",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
  )


@requires_model_fields("mat_shininess")
def mat_shininess(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize material shininess.

  MuJoCo stores shininess in ``[0, 1]``. MuJoCo Warp maps it to a Phong
  exponent in ``[0, 128]`` during RGB rendering.
  """
  _randomize_model_field(
    env,
    env_ids,
    "mat_shininess",
    entity_type="material",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
  )


@requires_model_fields("mat_texrepeat")
def mat_texrepeat(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize material texture repeat in the S/T directions.

  Only affects textured materials. Values should remain positive; zero or
  negative texture repeats are not meaningful for rendering.
  """
  _randomize_model_field(
    env,
    env_ids,
    "mat_texrepeat",
    entity_type="material",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0, 1],
  )
