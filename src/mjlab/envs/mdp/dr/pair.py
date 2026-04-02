"""Domain randomization functions for geom pair fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._core import _DEFAULT_ASSET_CFG, Ranges, _randomize_model_field
from ._types import Distribution, Operation

if TYPE_CHECKING:
  import torch

  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("pair_friction")
def pair_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize geom-pair friction overrides.

  Pair friction has 5 components: ``[tangent1, tangent2, spin, roll1, roll2]``.
  Default axis is 0 (tangent1 only). Axis 0 requires ``condim >= 3``
  (the default); axes 1 and 2 require ``condim >= 4``; axes 3 and 4
  require ``condim = 6``.

  Args:
    env: The environment instance.
    env_ids: Environment indices to randomize. ``None`` means all.
    ranges: Value range(s) for sampling.
    asset_cfg: Entity and pair selection.
    distribution: Sampling distribution.
    operation: How to combine sampled values with the base.
    axes: Which friction components to randomize. Defaults to ``[0]`` (tangent1).
    shared_random: If ``True``, all selected pairs receive the same sampled value per
      environment.
  """
  _randomize_model_field(
    env,
    env_ids,
    "pair_friction",
    entity_type="pair",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0],
    valid_axes=[0, 1, 2, 3, 4],
  )
