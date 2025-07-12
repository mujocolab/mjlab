from typing import Sequence
from mjlab.managers.manager_base import ManagerBase
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg

from mjlab.utils.dataclasses import get_terms


class ObservationManager(ManagerBase):
  def __init__(self, cfg: object, env):
    super().__init__(cfg=cfg, env=env)

    self._group_obs_dim = dict()

    self._obs_buffer = None

  # Properties.

  @property
  def active_terms(self) -> list[str] | dict[str, list[str]]:
    pass

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    pass

  def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
    return {}

  def _prepare_terms(self) -> None:
    group_cfg_items = get_terms(self.cfg, ObservationGroupCfg).items()
    for group_name, group_cfg in group_cfg_items:
      if group_cfg is None:
        print(f"group: {group_name} set to None, skipping...")
        continue
      print(f"Group: {group_name}:")

      group_cfg_items = get_terms(group_cfg, ObservationTermCfg).items()
      for term_name, term_cfg in group_cfg_items:
        if term_cfg is None:
          print(f"term: {term_name} set to None, skipping...")
          continue
        print(f"\t{term_name}")
