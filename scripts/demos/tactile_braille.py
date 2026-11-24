"""Braille reading demo on the Sharpa Wave hand.

A museum exhibit label: a clear acrylic plaque, tilted toward the visitor,
with an engraved title and braille lines below. The hand rests its index
fingertip on the first line and slides it along, like a reader would, then
moves on to the next line. Every reading of the fingertip's taxel grid is
written into a scan of the plaque, and each braille cell is decoded from that
scan once the finger has slid past it.

Run with:
  uv run python scripts/demos/tactile_braille.py
  uv run python scripts/demos/tactile_braille.py --lines "hello world"
"""

from __future__ import annotations

import base64
import io
from dataclasses import replace
from typing import Literal, NamedTuple

import mujoco
import numpy as np
import torch
import tyro
import viser
from PIL import Image, ImageDraw, ImageFont

import mjlab
from mjlab.actuator import BuiltinPositionActuatorCfg
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
from mjlab.utils.spec_config import CollisionCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viewer_config import ViewerConfig

# Braille letters as raised dots: 1-3 down the left column, 4-6 down the right.
# A space is a cell with no dots.
BRAILLE = {
  " ": (),
  "a": (1,), "b": (1, 2), "c": (1, 4), "d": (1, 4, 5), "e": (1, 5),
  "f": (1, 2, 4), "g": (1, 2, 4, 5), "h": (1, 2, 5), "i": (2, 4),
  "j": (2, 4, 5), "k": (1, 3), "l": (1, 2, 3), "m": (1, 3, 4),
  "n": (1, 3, 4, 5), "o": (1, 3, 5), "p": (1, 2, 3, 4), "q": (1, 2, 3, 4, 5),
  "r": (1, 2, 3, 5), "s": (2, 3, 4), "t": (2, 3, 4, 5), "u": (1, 3, 6),
  "v": (1, 2, 3, 6), "w": (2, 4, 5, 6), "x": (1, 3, 4, 6), "y": (1, 3, 4, 5, 6),
  "z": (1, 3, 5, 6),
}  # fmt: skip
LETTERS = {dots: letter for letter, dots in BRAILLE.items()}

# Standard braille geometry (m): dots 2.5 mm apart, cells 6 mm apart, lines
# 10 mm apart, each dot a spherical cap 1.5 mm across and 0.5 mm tall.
DOT_PITCH, CELL_PITCH, LINE_PITCH = 0.0025, 0.006, 0.010
DOT_RADIUS, DOT_HEIGHT = 0.00075, 0.0005
DOT_SPHERE = (DOT_RADIUS**2 + DOT_HEIGHT**2) / (2 * DOT_HEIGHT)

# The exhibit (m). The plaque frame has x along the braille lines, y up the
# plaque, z out of its face, and its origin at the first cell of the last line.
# The plaque tilts back so its face looks up toward the visitor.
TILT = np.radians(35.0)
LINE_HEIGHT = 0.3  # Last braille line above the world origin.
EDGE = 0.008  # Plaque's lower edge below the last braille line.
SIDE_MARGIN, TITLE_BAND = 0.016, 0.03  # Plaque margins beside and above the text.
THICKNESS = 0.005
# Clear acrylic, faintly blue; the raised dots catch a little more light.
ACRYLIC, ACRYLIC_DOT = (0.86, 0.93, 0.97, 0.25), (0.75, 0.88, 0.95, 0.6)

# Reading pose: index finger out, the others curled away, thumb tucked. With
# the pad flat on the plaque, everything below its face stays more than 10 mm
# below the line being read, so only the fingertip touches the plaque.
READING = {
  ".*_index_MCP_FE": 0.25,
  ".*_index_PIP": 0.15,
  ".*_index_DIP": 0.1,
  ".*_(middle|ring|pinky)_MCP_FE": 1.4,
  ".*_(middle|ring|pinky)_PIP": 1.5,
  ".*_(middle|ring|pinky)_DIP": 1.0,
  ".*_thumb_CMC_FE": 0.9,
  ".*_thumb_MCP_FE": 0.6,
  ".*_thumb_IP": 0.8,
  ".*": 0.0,
}
HAND_START = (0.0, -0.3, 0.5)

# The stroke: the pad set flat on the plaque, pressed into the dot tops, and
# slid along a line. The pad curves along the finger, so rows 2.5 mm from its
# lowest point sit ~0.3 mm higher; this press reaches all three rows.
PRESS = 0.00045
BRACE = 100.0
HOVER = 0.01
SPEED = 0.009  # m/s, a slow reader.
LOWER, LIFT, RETURN, HOLD = 1.0, 0.6, 1.4, 3.0  # s
MARGIN = 0.007  # Start and end each slide this far beyond the text.

# Scan of the braille lines, in plaque coordinates (m).
SCAN_DX, SCAN_DY = 0.0005, 0.001


def line_y(lines: tuple[str, ...], i: int) -> float:
  """Plaque y of line i's middle dot row; the last line sits at y = 0."""
  return (len(lines) - 1 - i) * LINE_PITCH


def line_length(line: str) -> float:
  return (len(line) - 1) * CELL_PITCH + DOT_PITCH


def plaque_extent(lines: tuple[str, ...]) -> tuple[float, float, float, float]:
  """Plaque (x min, x max, y min, y max) in the plaque frame."""
  width = max(line_length(line) for line in lines)
  return (
    -SIDE_MARGIN,
    width + SIDE_MARGIN,
    -EDGE,
    line_y(lines, 0) + DOT_PITCH + TITLE_BAND,
  )


def cells(lines: tuple[str, ...]) -> list[tuple[int, str, np.ndarray]]:
  """(line, character, six dot sites [6, 2]) for every braille cell."""
  out = []
  for i, line in enumerate(lines):
    for k, char in enumerate(line):
      sites = np.array(
        [
          (
            k * CELL_PITCH + (d // 3) * DOT_PITCH,
            line_y(lines, i) + DOT_PITCH * (1 - d % 3),
          )
          for d in range(6)
        ]
      )
      out.append((i, char, sites))
  return out


def add_engraving(
  spec: mujoco.MjSpec, body: mujoco.MjsBody, title: str, lines: tuple[str, ...]
) -> None:
  """The title and a border, frosted into the plaque's face: a textured decal a
  hair above the face, so the acrylic around it stays clear."""
  x0, x1, y0, y1 = plaque_extent(lines)
  width = 2048
  height = round(width * (y1 - y0) / (x1 - x0))
  image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
  draw = ImageDraw.Draw(image)
  frost = (70, 82, 92, 210)
  inset = round(0.02 * width)
  draw.rectangle((inset, inset, width - inset, height - inset), outline=frost, width=5)
  band = TITLE_BAND / (y1 - y0) * height  # The title sits above the braille.
  font = ImageFont.load_default(size=round(0.55 * band))
  draw.text((width / 2, 0.55 * band), title, fill=frost, font=font, anchor="mm")
  texture = spec.add_texture(
    name="engraving",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    width=width,
    height=height,
    nchannel=4,
  )
  texture.data = np.flipud(np.asarray(image)).tobytes()  # Rows bottom-up.
  # Translucent, so the native viewer blends the texture's alpha too.
  material = spec.add_material(name="engraving", rgba=(1, 1, 1, 0.99))
  material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "engraving"
  faces = [0, 1, 2, 0, 2, 3]
  spec.add_mesh(
    name="engraving",
    uservert=[x0, y0, 1e-5, x1, y0, 1e-5, x1, y1, 1e-5, x0, y1, 1e-5],
    userface=faces,
    usertexcoord=[0, 0, 1, 0, 1, 1, 0, 1],
    userfacetexcoord=faces,
    inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
  )
  body.add_geom(
    name="engraving",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="engraving",
    material="engraving",
    contype=0,
    conaffinity=0,
    density=0,
  )


def exhibit_spec(title: str, lines: tuple[str, ...]) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  exhibit = spec.worldbody.add_body(name="exhibit")
  x0, x1, y0, y1 = plaque_extent(lines)
  center = ((x0 + x1) / 2, (y0 + y1) / 2)
  plaque = exhibit.add_body(
    name="plaque",
    pos=(0.0, 0.0, LINE_HEIGHT),
    quat=(np.cos(TILT / 2), np.sin(TILT / 2), 0.0, 0.0),
  )
  plaque.add_geom(
    name="plaque",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=((x1 - x0) / 2, (y1 - y0) / 2, THICKNESS / 2),
    pos=(*center, -THICKNESS / 2),
    rgba=ACRYLIC,
  )
  add_engraving(spec, plaque, title, lines)
  for n, (_, char, sites) in enumerate(cells(lines)):
    for d in BRAILLE[char]:
      x, y = sites[d - 1]
      plaque.add_geom(
        name=f"dot_{n}_{d}",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(DOT_SPHERE, 0, 0),
        pos=(x, y, DOT_HEIGHT - DOT_SPHERE),
        rgba=ACRYLIC_DOT,
      )
  return spec


def create_env_cfg(
  side: Side, title: str, lines: tuple[str, ...]
) -> ManagerBasedRlEnvCfg:
  robot = get_sharpa_robot_cfg(side)
  robot.init_state = EntityCfg.InitialStateCfg(pos=HAND_START, joint_pos=READING)
  # Brace the hand like a reader does: stiffer joints keep the pad where the
  # stroke puts it. Torque limits stay at the hardware values.
  assert robot.articulation is not None
  robot.articulation = replace(
    robot.articulation,
    actuators=tuple(
      replace(a, stiffness=BRACE * a.stiffness, damping=BRACE * a.damping)
      for a in robot.articulation.actuators
      if isinstance(a, BuiltinPositionActuatorCfg)
    ),
  )
  exhibit = EntityCfg(
    spec_fn=lambda: exhibit_spec(title, lines),
    # The plaque outranks the hand, so its contact parameters govern: smooth
    # metal so the pad glides, and a soft contact so the fingertip sinks around
    # the dots, pressing with ~1 N like a human reader.
    collisions=(
      CollisionCfg(
        geom_names_expr=("plaque", "dot_.*"),
        contype=1,
        conaffinity=1,
        condim=3,
        priority=1,
        friction=(0.3,),
        solref=(0.15, 0.9),
      ),
    ),
  )
  # The fingertip's taxels, drawn on the pad with full heat at the dot height.
  sensors = get_fingertip_tactile_cfgs(
    side, fingers=("index",), resolution=(16, 10), debug_vis=True
  )
  for sensor in sensors:
    sensor.viz.depth_range = DEPTH_SCALE
  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    scene=SceneCfg(
      num_envs=1,
      entities={"robot": robot, "exhibit": exhibit},
      sensors=sensors,
    ),
  )
  cfg.sim.mujoco.timestep = 0.002
  cfg.sim.nconmax = 64
  cfg.sim.njmax = 256
  x0, x1, _, _ = plaque_extent(lines)
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    # From behind and below, through the clear plaque at the reading fingertip.
    lookat=((x0 + x1) / 2 - 0.01, 0.0, LINE_HEIGHT),
    distance=0.3,
    elevation=40.0,
    azimuth=-70.0,
  )
  return cfg


def pad_frame(points: np.ndarray, rows: int, cols: int):
  """Center and frame [x along finger, y across, z out] of a taxel grid."""
  grid = points.reshape(rows, cols, 3)
  r, c = rows // 2, cols // 2
  x = grid[r + 1, c] - grid[r - 1, c]
  y = grid[r, c + 1] - grid[r, c - 1]
  z = np.cross(x, y)
  z /= np.linalg.norm(z)
  x /= np.linalg.norm(x)
  return points.mean(0), np.stack([x, np.cross(z, x), z], axis=1)


# Panel theme: Helvetica, ink and grey text on light grey, Apple system blue.
FONT = '"Helvetica Neue", Helvetica, Arial, sans-serif'
TEXT, DIM, FAINT, LINE, ACCENT = "#1c1c20", "#787a82", "#d2d2d7", "#e5e5ea", "#0071e3"
# No contact is light grey; deeper contact deepens to blue.
INK = np.array(
  [[245, 245, 247], [221, 236, 255], [110, 170, 240], [0, 113, 227], [0, 58, 122]],
  dtype=np.float32,
)
DEPTH_SCALE = 0.8 * DOT_HEIGHT  # Depth shown at full ink.


def colormap(x: np.ndarray) -> np.ndarray:
  pos = np.clip(x, 0, 1) * (len(INK) - 1)
  i = np.minimum(pos.astype(int), len(INK) - 2)
  frac = (pos - i)[..., None]
  return INK[i] * (1 - frac) + INK[i + 1] * frac


def png(values: np.ndarray) -> str:
  """Inline PNG of values in [0, 1], first row at the top."""
  buf = io.BytesIO()
  Image.fromarray(colormap(values).astype(np.uint8)).save(buf, "PNG")
  return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class BrailleReader:
  """Accumulates tactile readings into a scan of the plaque and decodes cells."""

  def __init__(self, lines: tuple[str, ...]):
    self.cells = cells(lines)
    sites = np.stack([s for _, _, s in self.cells])  # [cells, 6, 2]
    lo = sites.reshape(-1, 2).min(0) - 0.004
    hi = sites.reshape(-1, 2).max(0) + 0.004
    self.origin = lo
    self.shape = (
      int(np.ceil((hi[1] - lo[1]) / SCAN_DY)) + 1,
      int(np.ceil((hi[0] - lo[0]) / SCAN_DX)) + 1,
    )
    # Scan bins within 1 mm of each dot site.
    iy, ix = np.indices(self.shape)
    centers = np.stack([lo[0] + ix * SCAN_DX, lo[1] + iy * SCAN_DY], -1)
    self.windows = (
      np.linalg.norm(centers[None, None] - sites[:, :, None, None], axis=-1) < 0.001
    )  # [cells, 6, H, W]
    self.reset()

  def reset(self) -> None:
    self.scan = np.zeros(self.shape)
    self.felt = np.zeros(self.shape, dtype=bool)
    self.decoded: list[str | None] = [None] * len(self.cells)

  def feel(self, points: np.ndarray, depth: np.ndarray) -> None:
    """Record taxel readings at plaque xy positions."""
    ix = np.round((points[:, 0] - self.origin[0]) / SCAN_DX).astype(int)
    iy = np.round((points[:, 1] - self.origin[1]) / SCAN_DY).astype(int)
    ok = (ix >= 0) & (ix < self.shape[1]) & (iy >= 0) & (iy < self.shape[0])
    np.maximum.at(self.scan, (iy[ok], ix[ok]), depth[ok])
    self.felt[iy[ok], ix[ok]] = True

  def decode(self, line: int, trailing_x: float) -> None:
    """Decode every cell of the line that the pad has slid completely past."""
    if not self.felt.any():
      return
    # Dots stand out above the flat plaque level by up to their height.
    threshold = np.median(self.scan[self.felt]) + 0.1 * DOT_HEIGHT
    for k, (i, _, sites) in enumerate(self.cells):
      if i != line or self.decoded[k] is not None or sites[:, 0].max() > trailing_x:
        continue
      dots = tuple(
        d + 1 for d, w in enumerate(self.windows[k]) if self.scan[w].max() > threshold
      )
      self.decoded[k] = LETTERS.get(dots, "?")


class Segment(NamedTuple):
  duration: float
  start: np.ndarray
  end: np.ndarray
  line: int
  reading: bool


class ReadingPolicy:
  """Moves the hand so its index pad strokes each braille line, and reads it."""

  STRIP_HZ = 30.0

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    lines: tuple[str, ...],
    server: viser.ViserServer | None,
  ):
    self.env = env
    self.lines = lines
    self.robot = env.scene["robot"]
    self.sensor: TactileSensor = env.scene["robot/index_tactile"]
    model = env.sim.mj_model
    self.mocap = next(
      int(model.body_mocapid[b])
      for b in range(model.nbody)
      if model.body_mocapid[b] >= 0 and model.body(b).name.startswith("robot/")
    )
    self.rows, self.cols = self.sensor.cfg.resolution

    # The plaque frame in the world; the stroke is planned in that frame.
    plaque = model.body("exhibit/plaque").id
    self.plaque_pos = env.sim.data.xpos[0, plaque].cpu().numpy().astype(float)
    self.plaque_rot = env.sim.data.xmat[0, plaque].view(3, 3).cpu().numpy()

    # Pad geometry in the hand frame, at the reading pose.
    taxels = self.sensor.taxel_pos_w()[0].cpu().numpy() - np.array(HAND_START)
    self.pad_center, pad = pad_frame(taxels, self.rows, self.cols)
    # Target pad frame in the plaque frame: finger pointing up the plaque, pad
    # facing it. Its across-finger axis then runs along the lines (+x).
    x, z = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    self.rot = np.stack([x, np.cross(z, x), z], axis=1) @ pad.T
    self.quat = np.zeros(4)
    mujoco.mju_mat2Quat(self.quat, (self.plaque_rot @ self.rot).ravel())
    # The pad's palmar face is nearly flat; center the part that touches first
    # (within 0.1 mm of its lowest point) on the line of text.
    offsets = (taxels - self.pad_center) @ self.rot.T
    touching = offsets[:, 2] < offsets[:, 2].min() + 1e-4
    self.contact = offsets[touching].mean(0)
    self.contact[2] = offsets[:, 2].min()
    # Half the pad's width along a line, to tell when it has left a cell.
    self.half_width = np.ptp(offsets[touching, 0]) / 2

    # The stroke as straight moves between waypoints (contact point x, y, and
    # height above the dot tops): lower onto each line, slide along it, lift,
    # and return to the next line's start; then pause.
    self.segments: list[Segment] = []
    end = None
    for i, line in enumerate(lines):
      x0, x1, y = -MARGIN, line_length(line) + MARGIN, line_y(lines, i)
      above, on = np.array([x0, y, HOVER]), np.array([x0, y, -PRESS])
      done, after = np.array([x1, y, -PRESS]), np.array([x1, y, HOVER])
      if end is not None:
        self.segments.append(Segment(RETURN, end, above, i, False))
      self.segments += [
        Segment(LOWER, above, on, i, False),
        Segment((x1 - x0) / SPEED, on, done, i, True),
        Segment(LIFT, done, after, i, False),
      ]
      end = after
    assert end is not None
    self.segments.append(Segment(HOLD, end, end, len(lines) - 1, False))
    self.period = sum(segment.duration for segment in self.segments)

    self.reader = BrailleReader(lines)
    self.t = 0.0
    self.next_strip_t = 0.0
    self.server = server
    self.html = None
    if server is not None:
      panel = server.gui.add_panel()
      with panel.add_tab("Reading"):
        self.html = server.gui.add_html("")
      panel.float(x=16, y=-16, width=520, height=310)

  def reset(self, dones: torch.Tensor | None = None) -> None:
    """Start reading over when the viewer resets the environment."""
    if dones is None or bool(dones.any()):
      self.t = self.next_strip_t = 0.0
      self.reader.reset()

  def stroke(self, phase: float) -> tuple[Segment, np.ndarray]:
    """The current segment and contact point waypoint (x, y, height)."""
    for segment in self.segments:
      if phase < segment.duration:
        break
      phase -= segment.duration
    u = min(phase / segment.duration, 1.0)
    if not segment.reading:  # Ease in and out, except at reading speed.
      u = u * u * (3 - 2 * u)
    return segment, segment.start + (segment.end - segment.start) * u

  def __call__(self, obs) -> torch.Tensor:
    del obs
    self.robot.set_joint_position_target(self.robot.data.default_joint_pos)
    phase = self.t % self.period
    if phase < self.env.step_dt:
      self.reader.reset()
    segment, (x, y, height) = self.stroke(phase)

    contact = np.array([x, y, DOT_HEIGHT + height])
    root = contact - self.contact - self.rot @ self.pad_center
    root = self.plaque_pos + self.plaque_rot @ root
    data = self.env.sim.data
    data.mocap_pos[0, self.mocap] = torch.as_tensor(root, dtype=torch.float32)
    data.mocap_quat[0, self.mocap] = torch.as_tensor(self.quat, dtype=torch.float32)

    if segment.reading:
      points = self.sensor.taxel_pos_w()[0].cpu().numpy()
      depth = self.sensor.data.depth[0].flatten().cpu().numpy()
      self.reader.feel((points - self.plaque_pos) @ self.plaque_rot, depth)
      self.reader.decode(segment.line, x - self.half_width)
    if self.server is not None and self.t == 0.0:
      self.style_forces()
    if self.html is not None and self.t >= self.next_strip_t:
      content = self.render()
      if content != self.html.content:
        self.html.content = content
      self.next_strip_t = self.t + 1.0 / self.STRIP_HZ
    self.t += self.env.step_dt
    return torch.zeros(self.env.action_space.shape, device=self.env.device)

  def style_forces(self) -> None:
    """Contact force arrows: ~2 cm per newton, thin, and an orange that stands
    out against acrylic and silver. Runs once the viewer is up, after its
    defaults."""
    vis = self.env.sim.mj_model.vis
    vis.map.force = 0.001  # Arrow length is force * map.force / meanmass.
    vis.scale.forcewidth = 0.03
    vis.rgba.contactforce[:] = (1.0, 0.584, 0.0, 0.95)

  def render(self) -> str:
    """The reading panel: the fingertip's taxels, the scan, and the text."""
    r = self.reader
    pad = self.sensor.data.depth[0].flip(0).cpu().numpy() / DEPTH_SCALE
    scan = r.scan[::-1] / DEPTH_SCALE
    # Scan extent in plaque coordinates, to place a box around each cell.
    x0, width = r.origin[0] - SCAN_DX / 2, r.shape[1] * SCAN_DX
    top, height = r.origin[1] + (r.shape[0] - 0.5) * SCAN_DY, r.shape[0] * SCAN_DY
    faint = f'<span style="color: {FAINT}">{{}}</span>'
    boxes = ""
    braille, printed = "", ""
    for k, ((i, char, sites), seen) in enumerate(zip(r.cells, r.decoded, strict=True)):
      if k > 0 and r.cells[k - 1][0] != i:  # Lines read on as one sentence.
        braille, printed = braille + "&nbsp;", printed + "&nbsp;"
      if char != " ":
        lo, hi = sites.min(0) - DOT_PITCH / 2, sites.max(0) + DOT_PITCH / 2
        left, w = (lo[0] - x0) / width * 100, (hi[0] - lo[0]) / width * 100
        t, h = (top - hi[1]) / height * 100, (hi[1] - lo[1]) / height * 100
        edge = f"1.5px solid {ACCENT}" if seen else f"1px dashed {FAINT}"
        boxes += (
          f'<div style="position: absolute; left: {left}%; top: {t}%; width: {w}%;'
          f' height: {h}%; border: {edge}; border-radius: 4px"></div>'
        )
      if seen is None:
        braille += faint.format("⠿" if char != " " else "&nbsp;")
        printed += faint.format("_" if char != " " else "&nbsp;")
      else:
        dots = sum(1 << (d - 1) for d in BRAILLE.get(seen, ()))
        braille += chr(0x2800 + dots) if seen != " " else "&nbsp;"
        printed += seen if seen != " " else "&nbsp;"
    image = "display: block; image-rendering: pixelated; border-radius: 6px;"
    image += f" border: 1px solid {LINE}"
    label = (
      "font-size: 10.5px; font-weight: 600; letter-spacing: 0.08em;"
      f" text-transform: uppercase; color: {DIM}; margin-bottom: 8px"
    )
    return (
      '<div style="display: flex; flex-direction: column; gap: 18px;'
      f' padding: 14px 18px 18px; font: 12px/1.4 {FONT}; color: {DIM}">'
      f'<div><div style="{label}">Read by touch</div>'
      f'<div style="font-size: 13px; color: {ACCENT}; letter-spacing: 1px;'
      f' white-space: nowrap">{braille}</div>'
      f'<div style="font: 500 20px/1.3 {FONT}; color: {TEXT};'
      f' white-space: nowrap; margin-top: 2px">{printed}</div></div>'
      '<div style="display: flex; gap: 18px; align-items: flex-start">'
      f'<div><div style="{label}">Fingertip</div><img src="{png(pad)}"'
      f' style="{image}; height: 104px; aspect-ratio: {self.cols} / {self.rows}">'
      "</div>"
      f'<div style="flex: 1"><div style="{label}">Scan of the plaque</div>'
      '<div style="position: relative">'
      f'<img src="{png(scan)}" style="{image}; width: 100%;'
      f' aspect-ratio: {width} / {height}">{boxes}</div></div></div></div>'
    )


def main(
  viewer: Literal["viser", "native"] = "viser",
  side: Side = "right",
  title: str = "mjlab",
  lines: tuple[str, ...] = ("mjlab now supports", "tactile sensors"),
) -> None:
  if not all(ch in BRAILLE for line in lines for ch in line):
    raise ValueError(f"Lines must use lowercase letters and spaces, got {lines!r}")
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=create_env_cfg(side, title, lines), device=device)
  env.reset()
  wrapped = RslRlVecEnvWrapper(env)
  if viewer == "viser":
    server = viser.ViserServer(label="mjlab")
    policy = ReadingPolicy(env, lines, server)
    ViserPlayViewer(wrapped, policy, viser_server=server).run()
  else:
    NativeMujocoViewer(wrapped, ReadingPolicy(env, lines, None)).run()
  wrapped.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
