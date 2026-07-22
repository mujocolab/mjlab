"""MuJoCo football asset configuration."""

import mujoco

from mjlab.entity import EntityCfg

FOOTBALL_RADIUS = 0.1098
FOOTBALL_MASS = 0.43
FOOTBALL_INITIAL_POS = (0.25, 0.0, FOOTBALL_RADIUS)
FOOTBALL_RGBA = (0.95, 0.55, 0.10, 1.0)
FOOTBALL_CONDIM = 6
FOOTBALL_FRICTION = (0.1, 0.005, 0.001)


def get_football_spec() -> mujoco.MjSpec:
  """Create a free-moving spherical football specification."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint(name="ball_freejoint")
  body.add_geom(
    name="ball_collision",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(FOOTBALL_RADIUS, 0.0, 0.0),
    mass=FOOTBALL_MASS,
    rgba=FOOTBALL_RGBA,
    condim=FOOTBALL_CONDIM,
    friction=FOOTBALL_FRICTION,
    # Keep the ball out of terrain-only ray scans without disabling contact.
    group=3,
  )
  return spec


def get_football_cfg() -> EntityCfg:
  """Create the football entity configuration used by the scene."""
  return EntityCfg(
    spec_fn=get_football_spec,
    init_state=EntityCfg.InitialStateCfg(pos=FOOTBALL_INITIAL_POS),
  )
