from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg

_SCENE_XML = Path(__file__).parent / "scene.xml"

from mjlab.utils import spec_config as spec_cfg


@dataclass(kw_only=True)
class SceneCfg:
  num_envs: int = 1
  env_spacing: float = 2.0
  terrain: TerrainImporterCfg | None = None
  entities: dict[str, EntityCfg] = field(default_factory=dict)
  extent: float | None = None
  
  sensors: tuple[spec_cfg.SensorCfg | spec_cfg.ContactSensorCfg, ...] = field(
    default_factory=tuple
  )


class Scene:
  def __init__(self, scene_cfg: SceneCfg, device: str) -> None:
    self._cfg = scene_cfg
    self._device = device
    self._entities: dict[str, Entity] = {}
    self._terrain: TerrainImporter | None = None
    self._default_env_origins: torch.Tensor | None = None
    # Runtime handles
    self._data: mjwarp.Data | None = None
    self._sensor_adr: dict[str, torch.Tensor] | None = None

    self._spec = mujoco.MjSpec.from_file(str(_SCENE_XML))
    if self._cfg.extent is not None:
      self._spec.stat.extent = self._cfg.extent
    self._attach_terrain()
    self._attach_entities()
    self._apply_spec_editors()

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
    # Keep runtime data handle for sensor access.
    self._data = data
    for ent in self._entities.values():
      ent.initialize(mj_model, model, data, self._device)

    # Build scene-wide sensor address map (indices into sensordata).
    sensor_adr: dict[str, torch.Tensor] = {}
    for sensor in self._spec.sensors:
      sensor_name = sensor.name
      sns = mj_model.sensor(sensor_name)
      dim = sns.dim[0]
      start_adr = sns.adr[0]
      sensor_adr[sensor_name] = (
        start_adr, start_adr + dim,
      )
    self._sensor_adr = sensor_adr

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    for ent in self._entities.values():
      ent.reset(env_ids)

  def update(self, dt: float) -> None:
    for ent in self._entities.values():
      ent.update(dt)

  def write_data_to_sim(self) -> None:
    for ent in self._entities.values():
      ent.write_data_to_sim()

  @property
  def sensor_data(self) -> dict[str, torch.Tensor]:
    """All sensor data in the scene keyed by full sensor name.

    Returns a dict mapping each sensor's full name (including entity prefix
    if applicable) to a tensor of shape (num_envs, sensor_dim).
    """
    return {
      name: self._data.sensordata[:, indices[0]:indices[1]].clone()
      for name, indices in self._sensor_adr.items()
    }

  # Private methods.

  def _attach_entities(self) -> None:
    for ent_name, ent_cfg in self._cfg.entities.items():
      ent_cls = ent_cfg.class_type
      if ent_cls is None:
        ent_cls = Entity
      if not issubclass(ent_cls, Entity):
        raise TypeError(f"Entity class for '{ent_name}' must inherit from Entity, got {ent_cls!r}")

      ent = ent_cls(ent_cfg)
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

  def _apply_spec_editors(self) -> None:
    for cfg_list in [
      # self.cfg.lights,
      # self.cfg.cameras,
      # self.cfg.textures,
      # self.cfg.materials,
      self._cfg.sensors,
      # self.cfg.collisions,
    ]:
      cfg_list: Sequence[spec_cfg.SpecCfg]
      for cfg in cfg_list:
        cfg.edit_spec(self._spec)
