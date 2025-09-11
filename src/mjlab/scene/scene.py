from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg
from mjlab.utils.spec_editor import spec_editor as common_editors
from mjlab.utils.spec_editor.spec_editor_config import OptionCfg

_SCENE_XML = Path(__file__).parent / "scene.xml"


@dataclass(kw_only=True)
class SceneCfg:
  num_envs: int = 1
  env_spacing: float = 2.0
  terrain: TerrainImporterCfg | None = None
  entities: dict[str, EntityCfg] = field(default_factory=dict)
  device: str = "cuda:0"


class Scene:
  def __init__(self, scene_cfg: SceneCfg) -> None:
    self._cfg = scene_cfg
    self._device = scene_cfg.device
    self._entities: dict[str, Entity] = {}
    self._terrain: TerrainImporter | None = None
    self._default_env_origins: torch.Tensor | None = None

    self._spec = mujoco.MjSpec.from_file(str(_SCENE_XML))
    self._attach_entities()
    self._attach_terrain()

  def compile(self) -> mujoco.MjModel:
    return self._spec.compile()

  def to_zip(self, path: Path) -> None:
    # TODO(kevin): This is buggy and the generated zip file is not reloadable.
    # I had to add assetdir="assets" in the compiler directive to make it work.
    # Check again in a future MuJoCo release if this has been resolved.
    with path.open("wb") as f:
      mujoco.MjSpec.to_zip(self._spec, f)

  def configure_sim_options(self, cfg: OptionCfg) -> None:
    common_editors.OptionEditor(cfg).edit_spec(self._spec)

  # Attributes.

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  @property
  def env_origins(self) -> torch.Tensor:
    if self._terrain is not None:
      assert self._terrain.env_origins is not None
      return self._terrain.env_origins
    assert self._default_env_origins is not None
    return self._default_env_origins

  @property
  def env_spacing(self) -> float:
    return self._cfg.env_spacing

  @property
  def entities(self) -> dict[str, Entity]:
    return self._entities

  @property
  def terrain(self) -> TerrainImporter | None:
    return self._terrain

  @property
  def num_envs(self) -> int:
    return self._cfg.num_envs

  def __getitem__(self, key: str) -> Any:
    all_keys = ["terrain"]
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
    self._default_env_origins = torch.zeros(
      (self._cfg.num_envs, 3), device=device, dtype=torch.float32
    )
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

  def _attach_terrain(self) -> None:
    if self._cfg.terrain is None:
      return
    self._cfg.terrain.num_envs = self._cfg.num_envs
    self._cfg.terrain.env_spacing = self._cfg.env_spacing
    self._terrain = TerrainImporter(self._cfg.terrain, self._device)
    frame = self._spec.worldbody.add_frame()
    self._spec.attach(self._terrain.spec, prefix="terrain/", frame=frame)


if __name__ == "__main__":
  from dataclasses import replace

  from mjlab.asset_zoo.robots.unitree_g1 import g1_constants

  SCENE_CFG = SceneCfg(
    entities={"robot": replace(g1_constants.G1_ROBOT_CFG)},
  )
  scene = Scene(SCENE_CFG)
