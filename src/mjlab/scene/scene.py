# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg

_SCENE_XML = Path(__file__).parent / "scene.xml"


@dataclass(kw_only=True)
class SceneCfg:
  num_envs: int = 1
  env_spacing: float = 2.0
  terrain: TerrainImporterCfg | None = None
  entities: dict[str, EntityCfg] = field(default_factory=dict)
  extent: float | None = None


class Scene:
  def __init__(self, scene_cfg: SceneCfg, device: str) -> None:
    self._cfg = scene_cfg
    self._device = device
    self._entities: dict[str, Entity] = {}
    self._terrain: TerrainImporter | None = None
    self._default_env_origins: torch.Tensor | None = None

    self._spec = mujoco.MjSpec.from_file(str(_SCENE_XML))
    if self._cfg.extent is not None:
      self._spec.stat.extent = self._cfg.extent
    self._attach_terrain()
    self._attach_entities()

  def compile(self) -> mujoco.MjModel:
    return self._spec.compile()

  def to_zip(self, path: Path) -> None:
    """Export the scene to a zip file.

    Warning: The generated zip may require manual adjustment of asset paths
    to be reloadable. Specifically, you may need to add assetdir="assets"
    to the compiler directive in the XML.

    Args:
      path: Output path for the zip file.

    TODO: Verify if this is fixed in future MuJoCo releases.
    """
    with path.open("wb") as f:
      mujoco.MjSpec.to_zip(self._spec, f)

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

  @property
  def device(self) -> str:
    return self._device

  def __getitem__(self, key: str) -> Any:
    if key == "terrain":
      if self._terrain is None:
        raise KeyError("No terrain configured in this scene.")
      return self._terrain

    if key in self._entities:
      return self._entities[key]

    # Not found, raise helpful error.
    available = list(self._entities.keys())
    if self._terrain is not None:
      available.append("terrain")
    raise KeyError(f"Scene element '{key}' not found. Available: {available}")

  # Methods.

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
  ):
    self._default_env_origins = torch.zeros(
      (self._cfg.num_envs, 3), device=self._device, dtype=torch.float32
    )
    for ent in self._entities.values():
      ent.initialize(mj_model, model, data, self._device)

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
    self._spec.attach(self._terrain.spec, frame=frame)
