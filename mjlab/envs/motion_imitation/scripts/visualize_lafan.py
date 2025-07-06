from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter
from mink import SO3

from mjlab.envs.motion_imitation import timeseries
from mjlab.entities.g1 import G1_XML, get_assets
from mjlab.entities.g1 import g1_constants

_ARROW_OFFSETS = [
  np.array([0, 1, 0, 1]),
  np.array([-1, 0, 0, 1]),
  np.array([0, 0, 1, 1]),
]
_ARROW_SO3 = [SO3(x / np.linalg.norm(x)) for x in _ARROW_OFFSETS]
_ARROW_COLORS = [
  np.array([0.6, 0.2, 0.2, 1]),
  np.array([0.2, 0.6, 0.2, 1]),
  np.array([0.2, 0.2, 0.6, 1]),
]
_AXIS_RADIUS = 0.01
_AXIS_LENGTH = 0.3


def visualize_body_axes(scn, xpos, xquat, body_ids):
  scn.ngeom = 0
  for body_idx in body_ids:
    body_xmat = SO3(xquat[body_idx])
    body_xpos = xpos[body_idx]
    for axis in range(3):
      scn.ngeom += 1
      mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom - 1],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=[_AXIS_RADIUS, _AXIS_RADIUS, _AXIS_LENGTH],
        pos=body_xpos,
        mat=(body_xmat @ _ARROW_SO3[axis]).as_matrix().ravel(),
        rgba=_ARROW_COLORS[axis],
      )
      scn.geoms[scn.ngeom - 1].category = mujoco.mjtCatBit.mjCAT_DECOR


@dataclass(frozen=True)
class Args:
  npz_file: str
  playback_speed: float = 1.0


def main(args: Args):
  model = mujoco.MjModel.from_xml_path(str(G1_XML), assets=get_assets())
  data = mujoco.MjData(model)

  body_ids = [model.body(name).id for name in g1_constants.BODY_NAMES]

  ts = timeseries.TimeSeries.from_npz(args.npz_file).resample(
    target_dt=model.opt.timestep
  )

  rate = RateLimiter(
    frequency=(1.0 / model.opt.timestep) * args.playback_speed, warn=False
  )
  with mujoco.viewer.launch_passive(
    model=model, data=data, show_left_ui=False, show_right_ui=False
  ) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = model.camera("tracking").id
    i = 0
    while viewer.is_running():
      data.qpos[:] = ts.qpos[i]
      mujoco.mj_forward(model, data)
      visualize_body_axes(viewer.user_scn, ts.xpos[i], ts.xquat[i], body_ids)
      viewer.sync()
      rate.sleep()
      i = min(i + 1, len(ts) - 1)


if __name__ == "__main__":
  main(tyro.cli(Args))
