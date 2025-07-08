import mujoco
from pathlib import Path

from mjlab.core import entity
from mjlab.entities.scene.scene_config import SceneCfg
from mjlab.entities.scene import editors
from mjlab.entities.robots.robot import Robot
from mjlab.entities.terrains.terrain import Terrain
from mjlab.entities.common import editors as common_editors

_HERE = Path(__file__).parent
_XML = _HERE / "scene.xml"


class Scene(entity.Entity):
  def __init__(self, scene_cfg: SceneCfg):
    self._cfg = scene_cfg
    self._spec = mujoco.MjSpec.from_file(str(_XML))

    self._configure_terrain()
    self._configure_robots()
    self._configure_lights()
    self._configure_cameras()
    self._configure_skybox()

  @property
  def spec(self) -> mujoco.MjSpec:
    return self._spec

  # Private methods.

  def _configure_terrain(self) -> None:
    for ter_cfg in self._cfg.terrains:
      ter = Terrain(ter_cfg)
      frame = self._spec.worldbody.add_frame()
      self._spec.attach(ter.spec, prefix="", frame=frame)

  def _configure_robots(self) -> None:
    for rob_cfg in self._cfg.robots:
      rob = Robot(rob_cfg)
      frame = self._spec.worldbody.add_frame()
      self._spec.attach(rob.spec, prefix="", frame=frame)

  def _configure_lights(self) -> None:
    for lig in self._cfg.lights:
      editors.LightEditor(lig).edit_spec(self._spec)

  def _configure_cameras(self) -> None:
    for cam in self._cfg.cameras:
      editors.CameraEditor(cam).edit_spec(self._spec)

  def _configure_skybox(self) -> None:
    if self._cfg.skybox is not None:
      common_editors.TextureEditor(self._cfg.skybox).edit_spec(self._spec)
