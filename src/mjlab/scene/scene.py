from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.terrains.terrain_importer_cfg import TerrainImporterCfg
from mjlab.utils.spec_editor import spec_editor as common_editors
from mjlab.utils.spec_editor.spec_editor_config import OptionCfg

_SCENE_XML = Path(__file__).parent / "scene.xml"


@dataclass(frozen=True, kw_only=True)
class SceneCfg:
  num_envs: int = 1
  """Number of environment instances in the scene."""
  env_spacing: float = 2.0
  """Spacing between environments. Only used when the number of environments is > 1."""
  terrain: TerrainImporterCfg | None = None
  """Configuration for the terrain."""
  entities: dict[str, EntityCfg] = field(default_factory=dict)
  """Dictionary of entity configurations."""


class Scene:
  def __init__(self, scene_cfg: SceneCfg) -> None:
    self._cfg = scene_cfg
    self._entities: dict[str, Entity] = {}
    self._spec = mujoco.MjSpec.from_file(str(_SCENE_XML))
    self._attach_entities()

  def compile(self) -> mujoco.MjModel:
    return self._spec.compile()

  def configure_sim_options(self, cfg: OptionCfg) -> None:
    common_editors.OptionEditor(cfg).edit_spec(self._spec)

  # Attributes.

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  @property
  def entities(self) -> dict[str, Entity]:
    return self._entities

  def __getitem__(self, key: str) -> Any:
    all_keys = []
    for asset_family in [
      self._entities,
    ]:
      out = asset_family.get(key)
      if out is not None:
        return out
      all_keys += list(asset_family.keys())
    raise KeyError(
      f"Scene entity with key '{key}' not found. Available entities: '{all_keys}'"
    )

  # Methods.

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ):
    for ent in self._entities.values():
      ent.initialize(mj_model, model, data, device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    for ent in self._entities.values():
      ent.reset(env_ids)

  def update(self, dt: float) -> None:
    for ent in self._entities.values():
      ent.update(dt)

  def write_data_to_sim(self) -> None:
    for ent in self._entities.values():
      ent.write_data_to_sim()

  # Private methods.

  def _attach_entities(self) -> None:
    for ent_name, ent_cfg in self._cfg.entities.items():
      ent = Entity(ent_cfg)
      self._entities[ent_name] = ent
      frame = self._spec.worldbody.add_frame()
      self._spec.attach(ent.spec, prefix=f"{ent_name}/", frame=frame)


if __name__ == "__main__":
  from dataclasses import replace

  from mjlab.asset_zoo.robots.unitree_g1 import g1_constants

  SCENE_CFG = SceneCfg(
    entities={"robot": replace(g1_constants.G1_ROBOT_CFG)},
  )
  scene = Scene(SCENE_CFG)
