from pathlib import Path
import abc
from typing import Sequence

import mujoco
import mujoco_warp as mjwarp

from mjlab.entities.indexing import EntityIndexing
from mjlab.entities.entity_config import EntityCfg
from mjlab.utils.spec_editor.spec_editor import (
  TextureEditor,
  MaterialEditor,
  LightEditor,
  CameraEditor,
  SensorEditor,
)


class Entity(abc.ABC):
  def __init__(self, cfg: EntityCfg):
    self.cfg = cfg
    self._spec = cfg.spec_fn()
    self._configure_spec()
    self._give_names_to_missing_elems()

  # Attributes.

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  # Methods.

  def compile(self) -> mujoco.MjModel:
    return self.spec.compile()

  def write_xml(self, xml_path: Path) -> None:
    with open(xml_path, "w") as f:
      f.write(self.spec.to_xml())

  @abc.abstractmethod
  def initialize(
    self,
    indexing: EntityIndexing,
    model: mujoco.MjModel,
    data: mjwarp.Data,
    device: str,
    wp_model,
  ) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def update(self, dt: float) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def reset(self, env_ids: Sequence[int] | None = None):
    raise NotImplementedError

  @abc.abstractmethod
  def write_data_to_sim(self) -> None:
    raise NotImplementedError

  # Private methods.

  def _configure_spec(self) -> None:
    for light in self.cfg.lights:
      LightEditor(light).edit_spec(self._spec)
    for camera in self.cfg.cameras:
      CameraEditor(camera).edit_spec(self._spec)
    for tex in self.cfg.textures:
      TextureEditor(tex).edit_spec(self._spec)
    for mat in self.cfg.materials:
      MaterialEditor(mat).edit_spec(self._spec)
    for sns in self.cfg.sensors:
      SensorEditor(sns).edit_spec(self._spec)

  def _give_names_to_missing_elems(self) -> None:
    """Ensure all important elements of the spec have names to simplify attachment."""

    def _incremental_rename(
      elem_list: Sequence[mujoco.MjsElement], elem_type: str
    ) -> None:
      counter: int = 0
      for elem in elem_list:
        if not elem.name:  # type: ignore
          elem.name = f"{elem_type}_{counter}"  # type: ignore
          counter += 1

    _incremental_rename(self._spec.bodies, "body")
    _incremental_rename(self._spec.geoms, "geom")
    _incremental_rename(self._spec.sites, "site")
    _incremental_rename(self._spec.sensors, "sensor")
