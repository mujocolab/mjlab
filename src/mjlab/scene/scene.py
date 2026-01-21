import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.sensor import BuiltinSensor, Sensor, SensorCfg
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg

SceneElementCfgType = EntityCfg | TerrainEntityCfg

_SCENE_XML = Path(__file__).parent / "scene.xml"


def _default_entities() -> dict[str, SceneElementCfgType]:
  return {"terrain": TerrainEntityCfg(terrain_type="plane")}


@dataclass(kw_only=True)
class SceneCfg:
  num_envs: int = 1
  env_spacing: float = 2.0
  entities: dict[str, SceneElementCfgType] = field(default_factory=_default_entities)
  sensors: tuple[SensorCfg, ...] = field(default_factory=tuple)
  extent: float | None = None
  spec_fn: Callable[[mujoco.MjSpec], None] | None = None

  @property
  def terrain(self) -> TerrainEntityCfg | None:
    """Helper to get terrain config from entities dict."""
    for cfg in self.entities.values():
      if isinstance(cfg, TerrainEntityCfg):
        return cfg
    return None


class Scene:
  def __init__(self, scene_cfg: SceneCfg, device: str) -> None:
    self._cfg = scene_cfg
    self._device = device
    self._entities: dict[str, Entity] = {}
    self._terrain: TerrainEntity | None = None
    self._sensors: dict[str, Sensor] = {}

    self._spec = mujoco.MjSpec.from_file(str(_SCENE_XML))
    if self._cfg.extent is not None:
      self._spec.stat.extent = self._cfg.extent
    self._add_entities()
    self._add_sensors()
    if self._cfg.spec_fn is not None:
      self._cfg.spec_fn(self._spec)

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
    return self._env_origins

  @property
  def env_spacing(self) -> float:
    return self._cfg.env_spacing

  @property
  def entities(self) -> dict[str, Entity]:
    return self._entities

  @property
  def sensors(self) -> dict[str, Sensor]:
    return self._sensors

  @property
  def terrain(self) -> TerrainEntity | None:
    return self._terrain

  @property
  def num_envs(self) -> int:
    return self._cfg.num_envs

  @property
  def device(self) -> str:
    return self._device

  def __getitem__(self, key: str) -> Any:
    if key in self._sensors:
      return self._sensors[key]
    if key in self._entities:
      return self._entities[key]
    if self._terrain is not None and key in self._cfg.entities:
      if isinstance(self._cfg.entities[key], TerrainEntityCfg):
        return self._terrain

    # Not found, raise helpful error.
    available = list(self._cfg.entities.keys()) + list(self._sensors.keys())
    raise KeyError(f"Scene element '{key}' not found. Available: {available}")

  # Methods.

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
  ):
    if self._terrain is not None:
      self._terrain.initialize(mj_model, model, data, self._device)
      self._env_origins = self._terrain.env_origins
    else:
      self._env_origins = self._compute_env_origins()

    for ent in self._entities.values():
      ent.initialize(mj_model, model, data, self._device)
    for sensor in self._sensors.values():
      sensor.initialize(mj_model, model, data, self._device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    for ent in self._entities.values():
      ent.reset(env_ids)
    for sensor in self._sensors.values():
      sensor.reset(env_ids)

  def update(self, dt: float) -> None:
    for ent in self._entities.values():
      ent.update(dt)
    for sensor in self._sensors.values():
      sensor.update(dt)

  def write_data_to_sim(self) -> None:
    for ent in self._entities.values():
      ent.write_data_to_sim()

  # Private methods.

  def _add_entities(self) -> None:
    # Collect keyframes from each entity to merge into a single scene keyframe.
    # Order matters: qpos/ctrl are concatenated in entity iteration order.
    key_qpos: list[np.ndarray] = []
    key_ctrl: list[np.ndarray] = []

    for ent_name, ent_cfg in self._cfg.entities.items():
      # Terrain entity.
      if isinstance(ent_cfg, TerrainEntityCfg):
        ent_cfg.num_envs = self._cfg.num_envs
        ent_cfg.env_spacing = self._cfg.env_spacing
        self._terrain = ent_cfg.build(self._device)
        frame = self._spec.worldbody.add_frame()
        self._spec.attach(self._terrain.spec, prefix="", frame=frame)
        continue

      # Regular entity.
      ent = ent_cfg.build()
      self._entities[ent_name] = ent
      # Extract keyframe before attach (must delete before attach to avoid corruption).
      if ent.spec.keys:
        if len(ent.spec.keys) > 1:
          warnings.warn(
            f"Entity '{ent_name}' has {len(ent.spec.keys)} keyframes; only the "
            "first one will be used.",
            stacklevel=2,
          )
        key_qpos.append(np.array(ent.spec.keys[0].qpos))
        key_ctrl.append(np.array(ent.spec.keys[0].ctrl))
        ent.spec.delete(ent.spec.keys[0])
      frame = self._spec.worldbody.add_frame()
      self._spec.attach(ent.spec, prefix=f"{ent_name}/", frame=frame)

    # Add merged keyframe to scene spec.
    if key_qpos:
      combined_qpos = np.concatenate(key_qpos)
      combined_ctrl = np.concatenate(key_ctrl)
      self._spec.add_key(name="init_state", qpos=combined_qpos, ctrl=combined_ctrl)

  def _add_sensors(self) -> None:
    for sensor_cfg in self._cfg.sensors:
      sns = sensor_cfg.build()
      sns.edit_spec(self._spec, self._entities)
      self._sensors[sensor_cfg.name] = sns

    for sns in self._spec.sensors:
      if sns.name not in self._sensors:
        self._sensors[sns.name] = BuiltinSensor.from_existing(sns.name)

  def _compute_env_origins(self) -> torch.Tensor:
    """Compute environment origins in a grid layout."""
    num_envs = self._cfg.num_envs
    spacing = self._cfg.env_spacing

    env_origins = np.zeros((num_envs, 3), dtype=np.float32)
    num_rows = int(np.ceil(num_envs / int(np.sqrt(num_envs))))
    num_cols = int(np.ceil(num_envs / num_rows))
    ii, jj = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing="ij")
    env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * spacing
    env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * spacing

    return torch.from_numpy(env_origins).to(self._device, dtype=torch.float)
