from dataclasses import dataclass
import mujoco

from mjlab.core.editors import SpecEditor
from mjlab.entities.scene.scene_config import LightCfg, CameraCfg

CAM_LIGHT_MODE_MAP = {
  "fixed": mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
  "track": mujoco.mjtCamLight.mjCAMLIGHT_TRACK,
  "trackcom": mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
  "targetbody": mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
  "targetbodycom": mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM,
}


@dataclass
class CameraEditor(SpecEditor):
  cfg: CameraCfg

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    if self.cfg.body == "world":
      body = spec.worldbody
    else:
      body = spec.body(self.cfg.body)
    camera = body.add_camera(
      mode=CAM_LIGHT_MODE_MAP[self.cfg.mode],
      fovy=self.cfg.fovy,
      pos=self.cfg.pos,
      quat=self.cfg.quat,
    )
    if self.cfg.name is not None:
      camera.name = self.cfg.name
    if self.cfg.target is not None:
      camera.targetbody = self.cfg.target


@dataclass
class LightEditor(SpecEditor):
  cfg: LightCfg

  TYPE_MAP = {
    "directional": mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    "spot": mujoco.mjtLightType.mjLIGHT_SPOT,
  }

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    if self.cfg.body == "world":
      body = spec.worldbody
    else:
      body = spec.body(self.cfg.body)
    light = body.add_light(
      mode=CAM_LIGHT_MODE_MAP[self.cfg.mode],
      type=self.TYPE_MAP[self.cfg.type],
      castshadow=self.cfg.castshadow,
      pos=self.cfg.pos,
      dir=self.cfg.dir,
      cutoff=self.cfg.cutoff,
      exponent=self.cfg.exponent,
    )
    if self.cfg.name is not None:
      light.name = self.cfg.name
    if self.cfg.target is not None:
      light.targetbody = self.cfg.target
