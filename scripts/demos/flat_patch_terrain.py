"""Flat patch terrain demo.

Spawns a Go1 on rough terrain with flat-patch sampling.
On each reset, the robot lands on a flat patch.

Run with:
  uv run python scripts/demos/flat_patch_terrain.py [--viewer native|viser]

Toggle visualization group 3 to see flat patch locations visualized as box sites.
"""

from __future__ import annotations

import os

import torch
import tyro

import mjlab
import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import events as mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_rough_env_cfg
from mjlab.terrains import FlatPatchSamplingCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def main(viewer: str = "auto") -> None:
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  cfg = unitree_go1_rough_env_cfg(play=True)

  spawn_patch_cfg = FlatPatchSamplingCfg(
    num_patches=100,
    patch_radius=0.3,
    max_height_diff=0.05,
  )

  # Override terrain: 1 row x 2 cols, curriculum mode so each column is deterministic.
  # Column 0 = discrete obstacles, Column 1 = pyramid slope.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    num_rows=1,
    num_cols=2,
    border_width=1.0,
    curriculum=True,
    add_lights=True,
    sub_terrains={
      "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        proportion=0.5,
        obstacle_height_range=(0.05, 0.5),
        obstacle_width_range=(0.4, 1.2),
        num_obstacles=30,
        platform_width=1.5,
        border_width=0.25,
        flat_patch_sampling={"spawn": spawn_patch_cfg},
      ),
      "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.5,
        slope_range=(0.3, 0.8),
        platform_width=1.5,
        border_width=0.25,
        flat_patch_sampling={"spawn": spawn_patch_cfg},
      ),
    },
  )

  # Remove all termination conditions except time limit.
  for key in list(cfg.terminations):
    if key != "time_out":
      del cfg.terminations[key]

  # Reset every 2 seconds to better showcase flat patch spawning.
  cfg.episode_length_s = 2.0

  # Replace reset_base event with flat-patch spawning.
  cfg.events["reset_base"] = EventTermCfg(
    func=mdp.reset_root_state_from_flat_patches,
    mode="reset",
    params={
      "patch_name": "spawn",
      "pose_range": {"z": (0.01, 0.05), "yaw": (-3.14, 3.14)},
    },
  )

  print("=" * 60)
  print("Flat Patch Terrain Demo")
  print("  Toggle group 3 to see flat patch markers (orange spheres)")
  print("  Press Enter in terminal to reset robot onto a flat patch")
  print("=" * 60)

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  class ZeroPolicy:
    def __call__(self, obs) -> torch.Tensor:
      del obs
      return torch.zeros(env.unwrapped.action_space.shape, device=device)

  policy = ZeroPolicy()

  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unknown viewer: {viewer}")

  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
