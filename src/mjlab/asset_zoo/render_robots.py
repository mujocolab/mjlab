from pathlib import Path

import mujoco
from PIL import Image

from mjlab.entity import Entity, EntityCfg
from mjlab.utils.spec_editor.spec_editor_config import TextureCfg

_HERE = Path(__file__).parent
_HEIGHT = 300
_WIDTH = 400


def render_robots(name: str, cfg: EntityCfg):
  cfg.textures = (
    TextureCfg(
      name="skybox",
      type="skybox",
      builtin="gradient",
      rgb1=(1, 1, 1),
      rgb2=(1, 1, 1),
      width=512,
      height=3072,
    ),
  )
  robot = Entity(cfg)
  model = robot.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  data.qpos[:3] = 0.0
  mujoco.mj_forward(model, data)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  cam.elevation = -10
  cam.distance = 1.15 * model.stat.extent
  cam.azimuth = -160
  cam.lookat[2] -= 0.1
  with mujoco.Renderer(model, height=_HEIGHT, width=_WIDTH) as renderer:
    renderer.update_scene(data, cam)
    img = renderer.render()
    img = Image.fromarray(img)
    img.save(_HERE / "img" / f"{name}.png")


if __name__ == "__main__":
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ROBOT_CFG
  from mjlab.asset_zoo.robots.unitree_go1.go1_constants import GO1_ROBOT_CFG

  render_robots("g1", G1_ROBOT_CFG)
  render_robots("go1", GO1_ROBOT_CFG)
