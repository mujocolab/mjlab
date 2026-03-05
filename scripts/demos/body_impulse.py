"""Body impulse demo with force visualization.

A cylinder hangs from a ball joint like a punching bag. Random impulses
swing it around while magenta arrows show the applied forces. Both native
and Viser viewers render the arrows via ``apply_body_impulse``'s built-in
debug visualization.

Run with:
  uv run mjpython scripts/demos/body_impulse.py                      # macOS
  uv run python scripts/demos/body_impulse.py                        # Linux
  uv run python scripts/demos/body_impulse.py --viewer viser         # Viser
"""

from __future__ import annotations

import math
import os

import mujoco
import torch
import tyro

import mjlab
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import apply_body_impulse, reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

BAG_RADIUS = 0.12  # punching bag radius
BAG_HALF_HEIGHT = 0.25  # punching bag half-height
BAG_DENSITY = 500.0  # ~5.7 kg
ROPE_LEN = 0.3  # rope length in meters
CEILING_Z = 1.0  # pivot height
ROPE_RADIUS = 0.008  # visual rope thickness


def create_pendulum_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.modelname = "impulse_punching_bag"

  # Ground plane.
  ground = spec.worldbody.add_geom()
  ground.type = mujoco.mjtGeom.mjGEOM_PLANE
  ground.size[:] = (5.0, 5.0, 0.1)
  ground.rgba[:] = (0.4, 0.5, 0.6, 1.0)

  # Light.
  light = spec.worldbody.add_light()
  light.pos[:] = (0, 0, 4)
  light.dir[:] = (0, 0, -1)
  light.diffuse[:] = (0.8, 0.8, 0.8)

  # Ceiling anchor (visual only).
  anchor = spec.worldbody.add_geom()
  anchor.type = mujoco.mjtGeom.mjGEOM_BOX
  anchor.size[:] = (0.04, 0.04, 0.02)
  anchor.pos[:] = (0, 0, CEILING_Z)
  anchor.rgba[:] = (0.3, 0.3, 0.3, 1.0)
  anchor.contype = 0
  anchor.conaffinity = 0

  # Pendulum body. Ball joint pivot is at (0, 0, CEILING_Z).
  bag_body = spec.worldbody.add_body()
  bag_body.name = "bag"
  bag_body.pos[:] = (0, 0, CEILING_Z)

  joint = bag_body.add_joint()
  joint.name = "bag_joint"
  joint.type = mujoco.mjtJoint.mjJNT_BALL
  joint.damping = 3.0
  joint.frictionloss = 0.5

  # Rope (visual only capsule from pivot to bag center).
  rope = bag_body.add_geom()
  rope.type = mujoco.mjtGeom.mjGEOM_CAPSULE
  rope.size[:2] = (ROPE_RADIUS, ROPE_LEN / 2)
  rope.pos[:] = (0, 0, -ROPE_LEN / 2)
  rope.rgba[:] = (0.5, 0.4, 0.3, 1.0)
  rope.contype = 0
  rope.conaffinity = 0
  rope.mass = 0.001  # negligible mass

  # Punching bag cylinder hanging at the end of the rope.
  geom = bag_body.add_geom()
  geom.name = "bag_geom"
  geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
  geom.size[:2] = (BAG_RADIUS, BAG_HALF_HEIGHT)
  geom.pos[:] = (0, 0, -ROPE_LEN - BAG_HALF_HEIGHT)
  geom.density = BAG_DENSITY
  geom.rgba[:] = (0.55, 0.15, 0.1, 0.35)

  return spec


def create_env_cfg() -> ManagerBasedRlEnvCfg:
  bag_cfg = EntityCfg(
    spec_fn=create_pendulum_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0),
    ),
  )

  bag_mass = BAG_DENSITY * math.pi * BAG_RADIUS**2 * (2 * BAG_HALF_HEIGHT)
  weight = bag_mass * 9.81
  force_mag = weight * 0.8  # 0.8x body weight

  cfg = ManagerBasedRlEnvCfg(
    decimation=10,
    scene=SceneCfg(
      num_envs=1,
      env_spacing=0.0,
      extent=2.0,
      entities={"bag": bag_cfg},
    ),
    events={
      "reset_scene_to_default": EventTermCfg(
        func=reset_scene_to_default,
        mode="reset",
      ),
      "impulse": EventTermCfg(
        func=apply_body_impulse,
        mode="step",
        params={
          "force_range": (-force_mag, force_mag),
          "torque_range": (-force_mag * 0.3, force_mag * 0.3),
          "duration_s": (0.05, 0.1),
          "cooldown_s": (1.0, 2.5),
          "asset_cfg": SceneEntityCfg("bag", body_names=("bag",)),
        },
      ),
    },
  )

  cfg.viewer.distance = 1.8
  cfg.viewer.elevation = -10.0

  return cfg


class ZeroPolicy:
  def __call__(self, obs: object) -> torch.Tensor:
    del obs
    return torch.zeros(1, 0)


def main(device: str = "cpu", viewer: str = "auto") -> None:
  configure_torch_backends()

  env_cfg = create_env_cfg()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  # Print force scaling info.
  mjm = env.unwrapped.sim.mj_model
  bag_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "bag")
  subtree_mass = mjm.body_subtreemass[bag_id]
  weight = subtree_mass * 9.81
  bag_mass = BAG_DENSITY * math.pi * BAG_RADIUS**2 * (2 * BAG_HALF_HEIGHT)
  force_mag = bag_mass * 9.81 * 0.8
  print("=" * 50)
  print("Body Impulse Demo (punching bag)")
  print(f"  Bag mass         : {bag_mass:.2f} kg")
  print(f"  Bag weight       : {weight:.2f} N")
  print(f"  Rope length      : {ROPE_LEN} m")
  print(f"  Force range      : +/-{force_mag:.0f} N")
  print(f"  Force/weight     : {force_mag / weight:.1f}x")
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

  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
