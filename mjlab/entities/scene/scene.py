import mujoco
from pathlib import Path
import mujoco_warp as mjwarp

from mjlab.entities import entity
from mjlab.entities.scene.scene_config import SceneCfg
from mjlab.entities.scene import editors
from mjlab.entities.robots.robot import Robot
from mjlab.entities.terrains.terrain import Terrain
from mjlab.entities.common.config import OptionCfg
from mjlab.entities.common import editors as common_editors

_HERE = Path(__file__).parent
_XML = _HERE / "scene.xml"


class Scene(entity.Entity):
  def __init__(self, scene_cfg: SceneCfg):
    self._cfg = scene_cfg
    self._entities: dict[str, entity.Entity] = {}
    spec = mujoco.MjSpec.from_file(str(_XML))
    super().__init__(spec)

    self._configure_terrain()
    self._configure_robots()
    self._configure_lights()
    self._configure_cameras()
    self._configure_skybox()

  # Attributes.

  @property
  def entities(self) -> dict[str, entity.Entity]:
    return self._entities

  # Methods.

  def initialize(self):
    print("SCENE INITIALIZE")

  def reset(self):
    print("SCENE RESET")
    pass

  def update(self, dt: float, data: mjwarp.Data) -> None:
    print("SCENE UPDATE")
    # for en in self._entities.values():
    #   en.update(dt, data)

  # Private methods.

  def _configure_terrain(self) -> None:
    for ter_name, ter_cfg in self._cfg.terrains.items():
      ter = Terrain(ter_cfg)
      self._entities[ter_name] = ter
      frame = self._spec.worldbody.add_frame()
      self._spec.attach(ter.spec, prefix=f"{ter_name}/", frame=frame)

  def _configure_robots(self) -> None:
    for rob_name, rob_cfg in self._cfg.robots.items():
      rob = Robot(rob_cfg)
      self._entities[rob_name] = rob
      frame = self._spec.worldbody.add_frame()
      self._spec.attach(rob.spec, prefix=f"{rob_name}/", frame=frame)

  def _configure_lights(self) -> None:
    for lig in self._cfg.lights:
      editors.LightEditor(lig).edit_spec(self._spec)

  def _configure_cameras(self) -> None:
    for cam in self._cfg.cameras:
      editors.CameraEditor(cam).edit_spec(self._spec)

  def _configure_skybox(self) -> None:
    if self._cfg.skybox is not None:
      common_editors.TextureEditor(self._cfg.skybox).edit_spec(self._spec)

  def configure_sim_options(self, cfg: OptionCfg) -> None:
    common_editors.OptionEditor(cfg).edit_spec(self._spec)
