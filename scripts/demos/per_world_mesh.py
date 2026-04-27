"""Per-world mesh variant demo.

Each world simulates a different object shape: sphere, cone, hemisphere,
or torus. Objects fall onto a ground plane with randomized colors.

Run with:
  uv run mjpython scripts/demos/per_world_mesh.py                  # macOS
  uv run scripts/demos/per_world_mesh.py                           # Linux
  uv run scripts/demos/per_world_mesh.py --viewer native           # Native
  uv run scripts/demos/per_world_mesh.py --viewer viser            # Viser
"""

from __future__ import annotations

import os

import mujoco
import torch
import tyro

import mjlab
from mjlab.entity import EntityCfg, VariantCfg, VariantEntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

# Object variant specs using MuJoCo mesh builtins.


# Shape definitions: (mesh_name, make_fn, make_kwargs, base_scale).
SHAPES: list[tuple[str, str, dict, float]] = [
  ("sphere", "make_sphere", {"subdivision": 3}, 0.10),
  ("cone", "make_cone", {"nedge": 16, "radius": 0.04}, 0.10),
  ("hemi", "make_hemisphere", {"resolution": 8}, 0.10),
  (
    "torus",
    "make_supertorus",
    {"resolution": 10, "radius": 0.4, "s": 1.0, "t": 1.0},
    0.08,
  ),
]

# Three size tiers per shape.
SIZE_MULTIPLIERS = {"small": 0.6, "medium": 1.0, "large": 1.5}


def _make_object_spec(
  mesh_name: str,
  make_fn_name: str,
  make_kwargs: dict,
  scale: float,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  mesh = spec.add_mesh()
  mesh.name = mesh_name
  getattr(mesh, make_fn_name)(**make_kwargs)
  mesh.scale[:] = (scale, scale, scale)

  body = spec.worldbody.add_body()
  body.name = "prop"
  body.add_freejoint()

  g = body.add_geom()
  g.name = "visual"
  g.type = mujoco.mjtGeom.mjGEOM_MESH
  g.meshname = mesh_name
  g.density = 200.0  # Light plastic, easy to drag in viewer.

  return spec


def _build_variants() -> dict[str, VariantCfg]:
  """Generate shape x size variant grid."""
  variants: dict[str, VariantCfg] = {}
  for shape_name, make_fn, make_kwargs, base_scale in SHAPES:
    for size_name, mult in SIZE_MULTIPLIERS.items():
      name = f"{shape_name}_{size_name}"
      scale = base_scale * mult

      def spec_fn(
        _mn=shape_name,
        _fn=make_fn,
        _kw=make_kwargs,
        _s=scale,
      ) -> mujoco.MjSpec:
        return _make_object_spec(_mn, _fn, _kw, _s)

      variants[name] = VariantCfg(spec_fn=spec_fn, weight=1.0)
  return variants


# Environment config.


def create_env_cfg() -> ManagerBasedRlEnvCfg:
  variants = _build_variants()
  object_cfg = VariantEntityCfg(
    variants=variants,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
  )

  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane", textures=(), materials=()),
      num_envs=24,
      env_spacing=1.5,
      extent=5.0,
      entities={"object": object_cfg},
    ),
    events={
      "reset_object": EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
          "pose_range": {
            "roll": (-3.14, 3.14),
            "pitch": (-3.14, 3.14),
            "yaw": (-3.14, 3.14),
          },
          "velocity_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.5, 2.0),
            "roll": (-5.0, 5.0),
            "pitch": (-5.0, 5.0),
            "yaw": (-5.0, 5.0),
          },
          "asset_cfg": SceneEntityCfg("object"),
        },
      ),
      "randomize_color": EventTermCfg(
        func=dr.geom_rgba,
        mode="reset",
        params={
          "ranges": (0.2, 1.0),
          "axes": [0, 1, 2],
          "asset_cfg": SceneEntityCfg("object"),
        },
      ),
      "randomize_friction": EventTermCfg(
        func=dr.geom_friction,
        mode="startup",
        params={
          "ranges": (0.5, 2.0),
          "operation": "scale",
          "axes": [0],
          "asset_cfg": SceneEntityCfg("object"),
        },
      ),
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002)),
    episode_length_s=5.0,
  )

  cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
  cfg.viewer.lookat = (0.0, 0.0, 0.0)
  cfg.viewer.distance = 4.0
  cfg.viewer.elevation = -30.0
  cfg.viewer.azimuth = 135.0

  return cfg


class ZeroPolicy:
  def __call__(self, obs: object) -> torch.Tensor:
    del obs
    return torch.zeros(1, 0)


def main(device: str | None = None, viewer: str = "auto") -> None:
  configure_torch_backends()
  if device is None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = create_env_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  n_variants = len(env_cfg.scene.entities["object"].variants)  # type: ignore[union-attr]
  print("=" * 50)
  print("Per-World Mesh Variant Demo")
  print(f"  Worlds: {env.num_envs}")
  print(f"  Variants: {n_variants} (4 shapes x 3 sizes)")
  print("  + randomized color, friction, velocity")
  print("=" * 50)

  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = viewer

  policy = ZeroPolicy()
  if resolved == "native":
    print("Launching native viewer...")
    NativeMujocoViewer(env, policy).run()
  elif resolved == "viser":
    print("Launching Viser viewer...")
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unknown viewer: {viewer}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
