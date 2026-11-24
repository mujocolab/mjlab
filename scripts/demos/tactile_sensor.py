"""Tactile sensor demo on the Sharpa Wave hand.

A probe visits each fingertip in turn: it presses into the pad, slides along
it, and lifts off. Every fingertip carries a taxel grid, drawn on the hand as a
heatmap and shown in a bottom strip as live depth and slip images.

Run with:
  uv run python scripts/demos/tactile_sensor.py
  uv run python scripts/demos/tactile_sensor.py --probe edge
  uv run python scripts/demos/tactile_sensor.py --probe ridge --side left
"""

from __future__ import annotations

import base64
import io
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro
import viser
from PIL import Image

import mjlab
from mjlab.asset_zoo.robots.sharpa_wave.sharpa_constants import (
  Side,
  get_fingertip_tactile_cfgs,
  get_sharpa_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.sensor import TactileSensor
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viewer_config import ViewerConfig

Probe = Literal["sphere", "edge", "ridge"]

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
RELAXED = {
  ".*_thumb_CMC_FE": 0.9,
  ".*_thumb_MCP_FE": 0.2,
  ".*_MCP_FE": 0.35,
  ".*_PIP": 0.35,
  ".*_DIP": 0.25,
  ".*": 0.0,
}

# Probe stroke per fingertip, in seconds and meters.
APPROACH, PRESS, SLIDE, LIFT = 0.6, 0.5, 1.6, 0.6
STANDOFF, DEPTH, SLIDE_HALF = 0.012, 0.0025, 0.004
PERIOD = APPROACH + PRESS + SLIDE + LIFT


def probe_spec(probe: Probe) -> mujoco.MjSpec:
  """A mocap probe whose contact feature points along its -z axis."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="probe", mocap=True)
  rgba = (0.95, 0.55, 0.15, 0.9)
  if probe == "sphere":
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.006, 0, 0), rgba=rgba)
  elif probe == "edge":
    # A box turned 45 degrees about y presents an edge across the finger.
    body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.006, 0.012, 0.006),
      quat=(np.cos(np.pi / 8), 0, np.sin(np.pi / 8), 0),
      pos=(0, 0, 0.0085),
      rgba=rgba,
    )
  else:
    # A thin rod lying across the finger.
    body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      size=(0.002, 0.012, 0),
      quat=(np.sqrt(0.5), np.sqrt(0.5), 0, 0),
      pos=(0, 0, 0.002),
      rgba=rgba,
    )
  return spec


def create_env_cfg(side: Side, probe: Probe) -> ManagerBasedRlEnvCfg:
  robot = get_sharpa_robot_cfg(side)
  robot.init_state = EntityCfg.InitialStateCfg(joint_pos=RELAXED)
  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    scene=SceneCfg(
      num_envs=1,
      entities={
        "robot": robot,
        "probe": EntityCfg(spec_fn=lambda: probe_spec(probe)),
      },
      sensors=get_fingertip_tactile_cfgs(side, debug_vis=True),
    ),
  )
  cfg.sim.mujoco.timestep = 0.002
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    lookat=(0.0, 0.0, 0.12),
    distance=0.4,
    elevation=-15.0,
    azimuth=200.0 if side == "right" else 20.0,
  )
  return cfg


def smoothstep(x: float) -> float:
  x = min(max(x, 0.0), 1.0)
  return x * x * (3.0 - 2.0 * x)


def pad_frame(sensor: TactileSensor) -> tuple[np.ndarray, np.ndarray]:
  """Center and orthonormal frame [x along finger, y across, z out] of a pad."""
  rows, cols = sensor.cfg.resolution
  grid = sensor.taxel_pos_w()[0].view(rows, cols, 3).cpu().numpy()
  r, c = rows // 2, cols // 2
  x = grid[r + 1, c] - grid[r - 1, c]
  y = grid[r, c + 1] - grid[r, c - 1]
  z = np.cross(x, y)
  z /= np.linalg.norm(z)
  x /= np.linalg.norm(x)
  return grid.reshape(-1, 3).mean(0), np.stack([x, np.cross(z, x), z], axis=1)


# Light grey (no signal) deepening to blue (full scale).
STOPS = np.array(
  [[245, 245, 247], [221, 236, 255], [110, 170, 240], [0, 113, 227], [0, 58, 122]],
  dtype=np.float32,
)


def colormap(x: np.ndarray) -> np.ndarray:
  """Map values in [0, 1] to RGB in [0, 255]."""
  pos = np.clip(x, 0, 1) * (len(STOPS) - 1)
  i = np.minimum(pos.astype(int), len(STOPS) - 2)
  frac = (pos - i)[..., None]
  return STOPS[i] * (1 - frac) + STOPS[i + 1] * frac


class TactileStrip:
  """A floating bottom panel showing every pad side by side, one tab per signal.

  The layout is HTML, so labels stay crisp. Each pad is a tiny PNG at taxel
  resolution that the browser upscales with sharp pixel edges.
  """

  SIGNALS = {"Depth": (0.001, "1 mm"), "Slip": (0.02, "20 mm/s")}
  STYLE = (
    'font: 12px/1.2 "Helvetica Neue", Helvetica, Arial, sans-serif; color: #787a82'
  )

  def __init__(
    self,
    server: viser.ViserServer,
    fingers: list[str],
    resolution: tuple[int, int],
  ):
    self.fingers = fingers
    self.aspect = f"{resolution[1]} / {resolution[0]}"
    stops = ", ".join(f"rgb({r:.0f},{g:.0f},{b:.0f})" for r, g, b in STOPS)
    self.bar = f"background: linear-gradient(to top, {stops})"
    panel = server.gui.add_panel()
    self.html = {}
    for name in self.SIGNALS:
      with panel.add_tab(name):
        self.html[name] = server.gui.add_html("")
    panel.float(x=16, y=-16, width=560, height=250)

  def _pad(self, label: str, pad: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(colormap(pad[::-1]).astype(np.uint8)).save(buf, "PNG")
    src = base64.b64encode(buf.getvalue()).decode()
    return (
      f'<div style="flex: 1; text-align: center">{label}'
      f'<img src="data:image/png;base64,{src}" style="display: block; width: 100%;'
      f" aspect-ratio: {self.aspect}; margin-top: 4px; border-radius: 6px;"
      " border: 1px solid #e5e5ea;"
      f' image-rendering: pixelated"></div>'
    )

  def render(self, name: str, pads: np.ndarray) -> str:
    """HTML for pads [F, R, C] already scaled to [0, 1], fingertip up."""
    unit = self.SIGNALS[name][1]
    tiles = "".join(self._pad(f, p) for f, p in zip(self.fingers, pads, strict=True))
    legend = (
      '<div style="display: flex; gap: 6px; padding-top: 18px">'
      f'<div style="width: 8px; border-radius: 4px; {self.bar}"></div>'
      '<div style="display: flex; flex-direction: column;'
      f' justify-content: space-between">{unit}<span>0</span></div></div>'
    )
    return (
      f'<div style="display: flex; gap: 10px; padding: 4px 2px; {self.STYLE}">'
      f"{tiles}{legend}</div>"
    )

  def update(self, sensors: list[TactileSensor]) -> None:
    depth = torch.stack([s.data.depth[0] for s in sensors])
    slip = torch.stack([s.data.slip[0].norm(dim=-1) for s in sensors])
    for name, values in (("Depth", depth), ("Slip", slip)):
      pads = (values / self.SIGNALS[name][0]).clamp(0, 1).cpu().numpy()
      content = self.render(name, pads)
      if content != self.html[name].content:  # Skip uploads while idle.
        self.html[name].content = content


class ProbePolicy:
  """Holds the hand still and drives the probe over each fingertip."""

  STRIP_HZ = 30.0

  def __init__(self, env: ManagerBasedRlEnv, server: viser.ViserServer | None):
    self.env = env
    self.robot = env.scene["robot"]
    # Order the fingers left to right as seen from the starting camera, so the
    # probe sweeps across the hand in the same order the strip lists the pads.
    az = np.radians(env.cfg.viewer.azimuth)
    camera_right = np.array([np.sin(az), -np.cos(az), 0.0])
    sensors = {f: env.scene[f"robot/{f}_tactile"] for f in FINGERS}
    self.fingers = sorted(
      FINGERS,
      key=lambda f: sensors[f].taxel_pos_w()[0].mean(0).cpu().numpy() @ camera_right,
    )
    self.sensors = [sensors[f] for f in self.fingers]
    model = env.sim.mj_model
    self.mocap = int(model.body_mocapid[model.body("probe/probe").id])
    self.dt = env.step_dt
    self.strip = None
    if server is not None:
      self.strip = TactileStrip(server, self.fingers, self.sensors[0].cfg.resolution)
    self.reset()

  def reset(self, dones: torch.Tensor | None = None) -> None:
    """Start over at the first fingertip when the viewer resets the env."""
    if dones is None or bool(dones.any()):
      self.t = self.next_strip_t = 0.0
      self.visit = -1

  def __call__(self, obs) -> torch.Tensor:
    del obs
    self.robot.set_joint_position_target(self.robot.data.default_joint_pos)

    visit, phase = divmod(self.t, PERIOD)
    if visit != self.visit:  # Aim at the next fingertip from where it is now.
      self.visit = visit
      self.center, self.frame = pad_frame(self.sensors[int(visit) % len(FINGERS)])
    t1, t2, t3 = APPROACH, APPROACH + PRESS, APPROACH + PRESS + SLIDE
    if phase < t2:
      height = STANDOFF * (1 - smoothstep(phase / t1)) - DEPTH * smoothstep(
        (phase - t1) / PRESS
      )
      slide = 0.0
    elif phase < t3:
      height = -DEPTH
      slide = SLIDE_HALF * np.sin(2 * np.pi * (phase - t2) / SLIDE)
    else:
      height = -DEPTH + (DEPTH + 2 * STANDOFF) * smoothstep((phase - t3) / LIFT)
      slide = 0.0
    pos = self.center + height * self.frame[:, 2] + slide * self.frame[:, 0]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, self.frame.ravel())
    data = self.env.sim.data
    data.mocap_pos[0, self.mocap] = torch.as_tensor(pos, dtype=torch.float32)
    data.mocap_quat[0, self.mocap] = torch.as_tensor(quat, dtype=torch.float32)

    if self.strip is not None and self.t >= self.next_strip_t:
      self.strip.update(self.sensors)
      self.next_strip_t = self.t + 1.0 / self.STRIP_HZ
    self.t += self.dt
    return torch.zeros(self.env.action_space.shape, device=self.env.device)


def main(
  viewer: Literal["viser", "native"] = "viser",
  side: Side = "right",
  probe: Probe = "sphere",
) -> None:
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=create_env_cfg(side, probe), device=device)
  env.reset()
  wrapped = RslRlVecEnvWrapper(env)
  if viewer == "viser":
    server = viser.ViserServer(label="mjlab")
    ViserPlayViewer(wrapped, ProbePolicy(env, server), viser_server=server).run()
  else:
    NativeMujocoViewer(wrapped, ProbePolicy(env, None)).run()
  wrapped.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
