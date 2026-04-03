"""Domain randomization functions for geom pair fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._core import _DEFAULT_ASSET_CFG, Ranges, _randomize_model_field
from ._types import Distribution, Operation

if TYPE_CHECKING:
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
  isotropic: bool = False,
) -> None:
  """Randomize geom-pair friction overrides.

  Pair friction has 5 components: ``[tangent1, tangent2, spin, roll1, roll2]``.
  Default axis is 0 (tangent1 only). Axis 0 requires ``condim >= 3``
  (the default); axes 1 and 2 require ``condim >= 4``; axes 3 and 4
  require ``condim = 6``.

  When ``isotropic=True``, tangent2 is set equal to tangent1 after sampling
  (axis 1 is overwritten with axis 0). If axes 3 or 4 are targeted, roll2 is
  also set equal to roll1 (axis 4 is overwritten with axis 3). This matches
  MuJoCo's standard geom friction convention where both tangent directions and
  both roll directions share one coefficient.

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
    isotropic: If ``True``, mirror tangent2 = tangent1 after sampling (and
      roll2 = roll1 if roll axes are targeted). Use this to maintain isotropic
      friction, matching MuJoCo's geom friction convention.
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

  if isotropic:
    _env_ids = (
      torch.arange(env.num_envs, device=env.device, dtype=torch.int)
      if env_ids is None
      else env_ids.to(env.device, dtype=torch.int)
    )
    asset = env.scene[asset_cfg.name]
    pair_indices = asset.indexing.pair_ids[asset_cfg.pair_ids]
    env_grid, pair_grid = torch.meshgrid(_env_ids, pair_indices, indexing="ij")
    pf = env.sim.model.pair_friction
    pf[env_grid, pair_grid, 1] = pf[env_grid, pair_grid, 0]
    if axes is not None:
      effective_axes = axes
    elif isinstance(ranges, dict) and ranges and isinstance(next(iter(ranges)), int):
      effective_axes = list(ranges.keys())
    else:
      effective_axes = [0]
    if 3 in effective_axes or 4 in effective_axes:
      pf[env_grid, pair_grid, 4] = pf[env_grid, pair_grid, 3]
