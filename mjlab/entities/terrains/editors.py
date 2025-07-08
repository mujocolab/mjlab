from dataclasses import dataclass
import mujoco

from mjlab.core.editors import SpecEditor
from mjlab.entities.terrains.terrain_config import GeomCfg


@dataclass
class GeomEditor(SpecEditor):
  cfg: GeomCfg

  TYPE_MAP = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "plane": mujoco.mjtGeom.mjGEOM_PLANE,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "hfield": mujoco.mjtGeom.mjGEOM_HFIELD,
  }

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    if self.cfg.body == "world":
      body = spec.worldbody
    else:
      body = spec.body(self.cfg.body)
    geom = body.add_geom(
      name=self.cfg.name,
      type=self.TYPE_MAP[self.cfg.type],
      rgba=self.cfg.rgba,
      material=self.cfg.material,
      group=self.cfg.group,
    )
    for i in range(len(self.cfg.size)):
      geom.size[i] = self.cfg.size[i]
