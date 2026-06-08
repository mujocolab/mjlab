"""Visualize material domain-randomization fields with MuJoCo Warp RGB rendering."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from PIL import Image, ImageDraw

from mjlab.entity import EntityCfg
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg

_FIELDS = (
  "baseline",
  "mat_rgba",
  "mat_emission",
  "mat_specular",
  "mat_shininess",
  "mat_texrepeat",
)


@dataclass
class _DemoEnv:
  scene: Scene
  sim: Simulation
  device: str

  @property
  def num_envs(self) -> int:
    return self.scene.num_envs


def _camera_xyaxes(
  pos: tuple[float, float, float],
  target: tuple[float, float, float],
) -> tuple[float, ...]:
  """Return MuJoCo camera xyaxes for a camera looking at ``target``."""
  pos_np = np.array(pos, dtype=np.float64)
  target_np = np.array(target, dtype=np.float64)
  forward = target_np - pos_np
  forward /= np.linalg.norm(forward)
  z_axis = -forward
  up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
  x_axis = np.cross(up, z_axis)
  x_axis /= np.linalg.norm(x_axis)
  y_axis = np.cross(z_axis, x_axis)
  y_axis /= np.linalg.norm(y_axis)
  return (*x_axis, *y_axis)


def _format_floats(values: tuple[float, ...]) -> str:
  return " ".join(f"{v:.8f}" for v in values)


def _demo_xml() -> str:
  camera_pos = (2.0, -3.0, 1.6)
  xyaxes = _format_floats(_camera_xyaxes(camera_pos, (0.0, 0.0, 0.25)))
  return f"""
<mujoco model="material_randomization">
  <visual>
    <headlight active="0"/>
  </visual>
  <asset>
    <texture name="checker" type="2d" builtin="checker" width="64" height="64"
      rgb1="0.15 0.15 0.15" rgb2="0.95 0.95 0.95"/>
    <material name="demo_mat" texture="checker" rgba="0.75 0.75 0.75 1"
      emission="0" specular="1" shininess="0.25" texrepeat="2 2"/>
  </asset>
  <worldbody>
    <light name="key" directional="true" pos="2 -3 4" dir="-0.4 0.6 -1"
      ambient="0.03 0.03 0.03" diffuse="1 1 1" specular="1 1 1"/>
    <camera name="main" pos="{_format_floats(camera_pos)}" xyaxes="{xyaxes}"/>
    <geom name="floor" type="plane" size="1.3 1.3 0.01" material="demo_mat"/>
    <body name="object" pos="0 0 0.35">
      <freejoint name="free"/>
      <geom name="sphere" type="sphere" size="0.35" mass="1"
        material="demo_mat"/>
    </body>
  </worldbody>
</mujoco>
"""


def _make_env(device: str) -> _DemoEnv:
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(_demo_xml()))
  scene = Scene(
    SceneCfg(num_envs=len(_FIELDS), entities={"robot": entity_cfg}),
    device,
  )
  mj_model = scene.compile()
  sim = Simulation(
    num_envs=len(_FIELDS),
    cfg=SimulationCfg(),
    model=mj_model,
    device=device,
  )
  scene.initialize(mj_model, sim.model, sim.data)
  sim.expand_model_fields(
    ("mat_rgba", "mat_emission", "mat_specular", "mat_shininess", "mat_texrepeat")
  )
  return _DemoEnv(scene=scene, sim=sim, device=device)


def _apply_material_randomization(env: _DemoEnv) -> None:
  torch.manual_seed(0)
  material_cfg = SceneEntityCfg("robot", material_names=("demo_mat",))
  material_cfg.resolve(env.scene)

  def env_id(index: int) -> torch.Tensor:
    return torch.tensor([index], device=env.device, dtype=torch.int)

  dr.mat_rgba(
    env,
    env_id(1),
    ranges={0: (0.2, 0.2), 1: (0.9, 0.9), 2: (1.0, 1.0), 3: (1.0, 1.0)},
    asset_cfg=material_cfg,
  )
  dr.mat_emission(
    env,
    env_id(2),
    ranges=(2.0, 2.0),
    asset_cfg=material_cfg,
  )
  dr.mat_specular(
    env,
    env_id(3),
    ranges=(0.0, 0.0),
    asset_cfg=material_cfg,
  )
  dr.mat_shininess(
    env,
    env_id(4),
    ranges=(0.9, 0.9),
    asset_cfg=material_cfg,
  )
  dr.mat_texrepeat(
    env,
    env_id(5),
    ranges=(8.0, 8.0),
    asset_cfg=material_cfg,
  )


def _unpack_rgb(packed: np.ndarray, width: int, height: int) -> np.ndarray:
  r = ((packed >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)
  g = ((packed >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)
  b = (packed & np.uint32(0xFF)).astype(np.uint8)
  rgb = np.stack((r, g, b), axis=-1)
  return rgb.reshape(packed.shape[0], height, width, 3)


def _render(env: _DemoEnv, width: int, height: int) -> np.ndarray:
  env.sim.forward()
  with wp.ScopedDevice(env.sim.wp_device):
    render_context = mjwarp.create_render_context(
      mjm=env.sim.mj_model,
      nworld=env.num_envs,
      cam_res=(width, height),
      render_rgb=True,
      use_textures=True,
      use_shadows=False,
      use_ambient_lighting=True,
      enabled_geom_groups=[0, 1, 2, 3, 4, 5],
      background_color=(0.05, 0.06, 0.07, 1.0),
      enable_specular=True,
      enable_emission=True,
    )
    mjwarp.render(env.sim.wp_model, env.sim.wp_data, render_context)
  return _unpack_rgb(render_context.rgb_data.numpy(), width, height)


def _compose_labeled_grid(images: np.ndarray) -> Image.Image:
  cell_h, cell_w = images.shape[1:3]
  label_h = 30
  output = Image.new("RGB", (cell_w * len(_FIELDS), cell_h + label_h), "white")
  draw = ImageDraw.Draw(output)
  for i, label in enumerate(_FIELDS):
    x = i * cell_w
    output.paste(Image.fromarray(images[i]), (x, 0))
    text_bbox = draw.textbbox((0, 0), label)
    text_w = text_bbox[2] - text_bbox[0]
    draw.text((x + (cell_w - text_w) / 2, cell_h + 8), label, fill=(20, 20, 20))
  return output


def _print_material_values(env: _DemoEnv) -> None:
  mat_id = mujoco.mj_name2id(
    env.sim.mj_model,
    mujoco.mjtObj.mjOBJ_MATERIAL,
    "robot/demo_mat",
  )
  model = env.sim.model
  print("Rendered worlds:")
  for idx, label in enumerate(_FIELDS):
    rgba = model.mat_rgba[idx, mat_id].cpu().numpy().round(3).tolist()
    emission = float(model.mat_emission[idx, mat_id].cpu())
    specular = float(model.mat_specular[idx, mat_id].cpu())
    shininess = float(model.mat_shininess[idx, mat_id].cpu())
    texrepeat = model.mat_texrepeat[idx, mat_id].cpu().numpy().round(3).tolist()
    print(
      f"  {idx}: {label}: rgba={rgba}, emission={emission:.3f}, "
      f"specular={specular:.3f}, shininess={shininess:.3f}, "
      f"texrepeat={texrepeat}"
    )


def main() -> None:
  parser = argparse.ArgumentParser(
    description=(
      "Render a fixed material-domain-randomization comparison image through "
      "MuJoCo Warp's RGB renderer."
    )
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("material_randomization.png"),
    help="Path for the generated PNG.",
  )
  parser.add_argument("--width", type=int, default=160, help="Per-world image width.")
  parser.add_argument("--height", type=int, default=160, help="Per-world image height.")
  parser.add_argument(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Warp/Torch device to render on.",
  )
  args = parser.parse_args()

  wp.config.quiet = True
  env = _make_env(args.device)
  _apply_material_randomization(env)
  images = _render(env, args.width, args.height)
  output = _compose_labeled_grid(images)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  output.save(args.output)
  print(f"Wrote {args.output.resolve()}")
  _print_material_values(env)


if __name__ == "__main__":
  main()
