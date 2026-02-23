"""Domain randomization functions for material fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._core import _DEFAULT_ASSET_CFG, _randomize_model_field
from ._types import Distribution, Operation

if TYPE_CHECKING:
  import torch

  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("mat_rgba")
def mat_rgba(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[int, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
) -> None:
  """Randomize material RGBA color.

  Args:
    env: The environment instance.
    env_ids: Environment indices to randomize. ``None`` means all.
    ranges: Value range(s) for sampling.
    asset_cfg: Entity and material selection. Use
      ``SceneEntityCfg("entity", material_names=(...))`` to target
      specific materials.
    distribution: Sampling distribution.
    operation: How to combine sampled values with the base.
    axes: Which RGBA channels to randomize. Defaults to ``[0, 1, 2, 3]``.
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
    default_axes=[0, 1, 2, 3],
  )
