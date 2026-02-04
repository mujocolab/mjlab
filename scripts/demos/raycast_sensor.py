"""Raycast sensor demo.

Run with:
  uv run mjpython scripts/demos/raycast_sensor.py [--viewer native|viser]  # macOS
  uv run python scripts/demos/raycast_sensor.py [--viewer native|viser]    # Linux

Examples:
  # Grid pattern (default)
  uv run python scripts/demos/raycast_sensor.py --pattern grid

  # Pinhole camera pattern
  uv run python scripts/demos/raycast_sensor.py --pattern pinhole

  # With yaw alignment (ignores pitch/roll)
  uv run python scripts/demos/raycast_sensor.py --alignment yaw

If using the native viewer, you can launch in interactive mode with:
  uv run mjpython scripts/demos/raycast_sensor.py --viewer native --interactive
"""

from __future__ import annotations

import os
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro

import mjlab
import mjlab.terrains as terrain_gen
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastSensorCfg,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.terrains.terrain_importer import TerrainImporterCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def create_scanner_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.modelname = "scanner"

  mat = spec.add_material()
  mat.name = "scanner_mat"
  mat.rgba[:] = (1.0, 0.5, 0.0, 0.9)

  scanner = spec.worldbody.add_body(mocap=True)
  scanner.name = "scanner"
  scanner.pos[:] = (0, 0, 2.0)

  geom = scanner.add_geom()
  geom.name = "scanner_geom"
  geom.type = mujoco.mjtGeom.mjGEOM_BOX
  geom.size[:] = (0.15, 0.15, 0.05)
  geom.mass = 1.0
  geom.material = "scanner_mat"

  scanner.add_camera(name="scanner", fovy=58.0, resolution=(16, 12))

  record_cam = scanner.add_camera(name="record_cam")
  record_cam.pos[:] = (2, 0, 2)
  record_cam.fovy = 40.0
  record_cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
  record_cam.targetbody = "scanner"

  return spec


def create_env_cfg(
  pattern: Literal["grid", "pinhole"],
  alignment: Literal["base", "yaw", "world"],
) -> ManagerBasedRlEnvCfg:
  custom_terrain_cfg = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=0.5,
    num_rows=1,
    num_cols=4,
    curriculum=True,
    sub_terrains={
      "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
        proportion=0.25,
        step_height_range=(0.1, 0.25),
        step_width=0.3,
        platform_width=1.5,
        border_width=0.25,
      ),
      "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.25,
        slope_range=(0.6, 1.5),
        platform_width=1.5,
        border_width=0.25,
        inverted=True,
      ),
      "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.25,
        noise_range=(0.05, 0.15),
        noise_step=0.02,
        border_width=0.25,
      ),
      "wave_terrain": terrain_gen.HfWaveTerrainCfg(
        proportion=0.25,
        amplitude_range=(0.15, 0.25),
        num_waves=3,
        border_width=0.25,
      ),
    },
    add_lights=True,
  )

  terrain_cfg = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=custom_terrain_cfg,
    num_envs=1,
  )

  scanner_entity_cfg = EntityCfg(
    spec_fn=create_scanner_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.65, -0.4, 0.5)),
  )

  if pattern == "grid":
    pattern_cfg = GridPatternCfg(
      size=(0.6, 0.6),
      resolution=0.1,
      direction=(0.0, 0.0, -1.0),
    )
  else:
    assert pattern == "pinhole"
    pattern_cfg = PinholeCameraPatternCfg.from_mujoco_camera("scanner/scanner")

  raycast_cfg = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="scanner", entity="scanner"),
    pattern=pattern_cfg,
    ray_alignment=alignment,
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(
      hit_color=(0.0, 1.0, 0.0, 0.9),
      miss_color=(1.0, 0.0, 0.0, 0.5),
      show_rays=False,
      show_normals=True,
    ),
  )

  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    scene=SceneCfg(
      num_envs=1,
      env_spacing=0.0,
      extent=2.0,
      terrain=terrain_cfg,
      entities={"scanner": scanner_entity_cfg},
      sensors=(raycast_cfg,),
    ),
  )

  cfg.viewer.body_name = "scanner"
  cfg.viewer.distance = 12.0
  cfg.viewer.elevation = -25.0
  cfg.viewer.azimuth = 135.0

  return cfg


def main(
  viewer: str = "auto",
  interactive: bool = False,
  pattern: Literal["grid", "pinhole"] = "grid",
  alignment: Literal["base", "yaw", "world"] = "base",
) -> None:
  configure_torch_backends()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  print("=" * 60)
  print("Raycast Sensor Demo - 4 Terrain Types")
  print(f"  Pattern: {pattern}")
  print(f"  Alignment: {alignment}")
  print("=" * 60)
  print()

  env_cfg = create_env_cfg(pattern, alignment)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = viewer

  use_auto_scan = (resolved_viewer == "viser") or (not interactive)

  if use_auto_scan:

    class AutoScanPolicy:
      def __init__(self):
        self.step_count = 0

      def __call__(self, obs) -> torch.Tensor:
        del obs
        t = self.step_count * 0.005
        y_period = 1000
        y_normalized = (self.step_count % y_period) / y_period
        y = -8.0 + 16.0 * y_normalized
        x = 1.5 * np.sin(2 * np.pi * t * 0.3)
        z = 1.0
        env.unwrapped.sim.data.mocap_pos[0, 0, :] = torch.tensor(
          [x, y, z], device=device, dtype=torch.float32
        )
        env.unwrapped.sim.data.mocap_quat[0, 0, :] = torch.tensor(
          [1, 0, 0, 0], device=device, dtype=torch.float32
        )
        self.step_count += 1
        return torch.zeros(env.unwrapped.action_space.shape, device=device)

    policy = AutoScanPolicy()
  else:

    class PolicyZero:
      def __call__(self, obs) -> torch.Tensor:
        del obs
        return torch.zeros(env.unwrapped.action_space.shape, device=device)

    policy = PolicyZero()

  if resolved_viewer == "native":
    print("Launching native viewer...")
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    print("Launching viser viewer...")
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unknown viewer: {viewer}")

  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
