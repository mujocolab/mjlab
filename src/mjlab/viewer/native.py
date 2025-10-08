"""Environment viewer built on MuJoCo's passive viewer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Callable, Optional

import mujoco
import mujoco.viewer
import numpy as np
import torch

from mjlab.viewer.base import (
  BaseViewer,
  EnvProtocol,
  PolicyProtocol,
  VerbosityLevel,
  ViewerAction,
)
from mjlab.viewer.mujoco_native_visualizer import MujocoNativeDebugVisualizer

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(frozen=True)
class PlotCfg:
  """Reward plot configuration."""

  history: int = 300  # Points kept per series.
  p_lo: float = 2.0  # Percentile low.
  p_hi: float = 98.0  # Percentile high.
  pad: float = 0.25  # Pad % of span on both sides.
  min_span: float = 1e-6  # Minimum vertical span.
  init_yrange: tuple[float, float] = (-0.01, 0.01)  # Initial y-range.
  grid_size: tuple[int, int] = (3, 4)  # Grid size (rows, columns).
  max_viewports: int = 12  # Cap number of plots shown.
  max_rows_per_col: int = 6  # Stack up to this many per column.
  plot_strip_fraction: float = 1 / 3  # Right-side width reserved for plots.
  background_alpha: float = 0.5  # Background alpha for plots.


class NativeMujocoViewer(BaseViewer):
  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 60.0,
    render_all_envs: bool = True,
    key_callback: Optional[Callable[[int], None]] = None,
    plot_cfg: PlotCfg | None = None,
    enable_perturbations: bool = True,
    disable_shadows: bool = False,
    verbosity: VerbosityLevel = VerbosityLevel.SILENT,
  ):
    super().__init__(env, policy, frame_rate, render_all_envs, verbosity)
    self.user_key_callback = key_callback
    self.enable_perturbations = enable_perturbations
    self.disable_shadows = disable_shadows

    self.mjm: Optional[mujoco.MjModel] = None
    self.mjd: Optional[mujoco.MjData] = None
    self.viewer: Optional[mujoco.viewer.Handle] = None
    self.vd: Optional[mujoco.MjData] = None
    self.vopt: Optional[mujoco.MjvOption] = None
    self.pert: Optional[mujoco.MjvPerturb] = None
    self.catmask: int = mujoco.mjtCatBit.mjCAT_DYNAMIC.value

    self._term_names: list[str] = []
    self._figures: dict[str, mujoco.MjvFigure] = {}  # Per-term figure.
    self._histories: dict[str, deque[float]] = {}  # Per-term ring buffer.
    self._yrange: dict[str, tuple[float, float]] = {}  # Per-term y-range.
    self._show_plots: bool = True
    self._plot_cfg = plot_cfg or PlotCfg()

    self.env_idx = 0
    self._mj_lock = Lock()

  def setup(self) -> None:
    """Setup MuJoCo viewer resources."""
    sim = self.env.unwrapped.sim
    self.mjm = sim.mj_model
    self.mjd = sim.mj_data

    if self.render_all_envs and self.env.unwrapped.num_envs > 1:
      assert self.mjm is not None
      self.vd = mujoco.MjData(self.mjm)

    self.pert = mujoco.MjvPerturb() if self.enable_perturbations else None
    self.vopt = mujoco.MjvOption()

    # self._term_names = [
    #   name
    #   for name, _ in self.env.unwrapped.reward_manager.get_active_iterable_terms(
    #     self.env_idx
    #   )
    # ]
    # self._init_reward_plots(self._term_names)

    assert self.mjm is not None
    assert self.mjd is not None
    self.viewer = mujoco.viewer.launch_passive(
      self.mjm,
      self.mjd,
      key_callback=self._safe_key_callback,
      show_left_ui=False,
      show_right_ui=False,
    )
    if self.viewer is None:
      raise RuntimeError("Failed to launch MuJoCo viewer")

    if self.disable_shadows:
      self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    self._setup_camera()

    if self.enable_perturbations:
      self.log("[INFO] Interactive perturbations enabled", VerbosityLevel.INFO)

  def is_running(self) -> bool:
    return bool(self.viewer and self.viewer.is_running())

  def sync_env_to_viewer(self) -> None:
    """Copy env state to viewer; update reward figures; render other envs."""
    v = self.viewer
    assert v is not None
    assert self.mjm is not None and self.mjd is not None and self.vopt is not None

    with self._mj_lock:
      sim_data = self.env.unwrapped.sim.data
      self.mjd.qpos[:] = sim_data.qpos[self.env_idx].cpu().numpy()
      self.mjd.qvel[:] = sim_data.qvel[self.env_idx].cpu().numpy()
      mujoco.mj_forward(self.mjm, self.mjd)

      # text_1 = "Env\nStep\nStatus\nSpeed\nFPS"
      # text_2 = (
      #   f"{self.env_idx + 1}/{self.env.num_envs}\n"
      #   f"{self._step_count}\n"
      #   f"{'PAUSED' if self._is_paused else 'RUNNING'}\n"
      #   f"{self._time_multiplier * 100:.1f}%\n"
      #   f"{self._smoothed_fps:.1f}"
      # )
      # overlay = (
      #   mujoco.mjtFontScale.mjFONTSCALE_150.value,
      #   mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
      #   text_1,
      #   text_2,
      # )
      # v.set_texts(overlay)

      # if self._show_plots and self._term_names:
      #   terms = list(
      #     self.env.unwrapped.reward_manager.get_active_iterable_terms(self.env_idx)
      #   )
      #   if not self._is_paused:
      #     for name, arr in terms:
      #       if name in self._histories:
      #         self._append_point(name, float(arr[0]))
      #         self._write_history_to_figure(name)

      #   viewports = compute_viewports(len(self._term_names), v.viewport, self._plot_cfg)
      #   viewport_figs = [
      #     (viewports[i], self._figures[self._term_names[i]])
      #     for i in range(
      #       min(len(viewports), len(self._term_names), self._plot_cfg.max_viewports)
      #     )
      #   ]
      #   v.set_figures(viewport_figs)
      # else:
      #   v.set_figures([])

      v.user_scn.ngeom = 0
      if hasattr(self.env.unwrapped, "update_visualizers"):
        visualizer = MujocoNativeDebugVisualizer(v.user_scn, self.mjm)
        self.env.unwrapped.update_visualizers(visualizer)

      if self.render_all_envs and self.vd is not None:
        for i in range(self.env.unwrapped.num_envs):
          if i == self.env_idx:
            continue
          self.vd.qpos[:] = sim_data.qpos[i].cpu().numpy()
          self.vd.qvel[:] = sim_data.qvel[i].cpu().numpy()
          mujoco.mj_forward(self.mjm, self.vd)
          assert self.pert is not None
          mujoco.mjv_addGeoms(
            self.mjm, self.vd, self.vopt, self.pert, self.catmask, v.user_scn
          )

      v.sync(state_only=True)

  def sync_viewer_to_env(self) -> None:
    """Copy perturbation forces from viewer to env (when not paused)."""
    if not (self.enable_perturbations and not self._is_paused and self.mjd):
      return
    with self._mj_lock:
      xfrc = torch.as_tensor(
        self.mjd.xfrc_applied, dtype=torch.float, device=self.env.device
      )
    self.env.unwrapped.sim.data.xfrc_applied[:] = xfrc[None]

  def close(self) -> None:
    """Close viewer and cleanup."""
    v = self.viewer
    self.viewer = None
    if v:
      try:
        if v.is_running():
          v.close()
      except Exception as e:
        self.log(f"[WARN] Error while closing viewer: {e}", VerbosityLevel.INFO)
    self.log("[INFO] MuJoCo viewer closed", VerbosityLevel.INFO)

  def reset_environment(self) -> None:
    """Extend BaseViewer.reset_environment to clear reward histories."""
    super().reset_environment()
    self._clear_histories()

  def _safe_key_callback(self, key: int) -> None:
    """Runs on MuJoCo viewer thread; must not touch env/sim directly."""
    from mjlab.viewer.keys import (
      KEY_COMMA,
      KEY_ENTER,
      KEY_EQUAL,
      KEY_MINUS,
      KEY_P,
      KEY_PERIOD,
      KEY_SPACE,
    )

    if key == KEY_ENTER:
      self.request_reset()
    elif key == KEY_SPACE:
      self.request_toggle_pause()
    elif key == KEY_MINUS:
      self.request_speed_down()
    elif key == KEY_EQUAL:
      self.request_speed_up()
    elif key == KEY_COMMA:
      self.request_action("PREV_ENV")
    elif key == KEY_PERIOD:
      self.request_action("NEXT_ENV")
    elif key == KEY_P:
      self.request_action("TOGGLE_PLOTS")

    if self.user_key_callback:
      try:
        self.user_key_callback(key)
      except Exception as e:
        self.log(f"[WARN] user key_callback raised: {e}", VerbosityLevel.INFO)

  def _handle_custom_action(self, action, payload) -> bool:
    if action == ViewerAction.PREV_ENV and self.env.unwrapped.num_envs > 1:
      self.env_idx = (self.env_idx - 1) % self.env.unwrapped.num_envs
      self._clear_histories()
      self.log(f"[INFO] Switched to environment {self.env_idx}", VerbosityLevel.INFO)
      return True
    elif action == ViewerAction.NEXT_ENV and self.env.unwrapped.num_envs > 1:
      self.env_idx = (self.env_idx + 1) % self.env.unwrapped.num_envs
      self._clear_histories()
      self.log(f"[INFO] Switched to environment {self.env_idx}", VerbosityLevel.INFO)
      return True
    else:
      if hasattr(action, "value") and action.value == "custom":
        if payload == "TOGGLE_PLOTS" or payload is None:
          self._show_plots = not self._show_plots
          self.log(
            f"[INFO] Reward plots {'shown' if self._show_plots else 'hidden'}",
            VerbosityLevel.INFO,
          )
          return True
    return False

  def _setup_camera(self) -> None:
    # TODO(kevin): This function is gross and has lots of redundant code. Clean it up.
    assert self.viewer is not None
    self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD.value

    if self.cfg and hasattr(self.cfg, "origin_type"):
      if self.cfg.origin_type == self.cfg.OriginType.WORLD:
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
        self.viewer.cam.fixedcamid = -1
        self.viewer.cam.trackbodyid = -1

      elif self.cfg.origin_type == self.cfg.OriginType.ASSET_ROOT:
        if not self.cfg.asset_name:
          raise ValueError("Asset name must be specified for ASSET_ROOT origin type")
        robot: Entity = self.env.unwrapped.scene[self.cfg.asset_name]
        body_id = robot.indexing.root_body_id
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
        self.viewer.cam.trackbodyid = body_id
        self.viewer.cam.fixedcamid = -1

      else:  # ASSET_BODY
        if not self.cfg.asset_name or not self.cfg.body_name:
          raise ValueError("asset_name/body_name required for ASSET_BODY origin type")
        robot: Entity = self.env.unwrapped.scene[self.cfg.asset_name]
        if self.cfg.body_name not in robot.body_names:
          raise ValueError(
            f"Body '{self.cfg.body_name}' not found in asset '{self.cfg.asset_name}'"
          )
        body_id_list, _ = robot.find_bodies(self.cfg.body_name)
        body_id = robot.indexing.bodies[body_id_list[0]].id

        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
        self.viewer.cam.trackbodyid = body_id
        self.viewer.cam.fixedcamid = -1

      self.viewer.cam.lookat = getattr(self.cfg, "lookat", self.viewer.cam.lookat)
      self.viewer.cam.elevation = getattr(
        self.cfg, "elevation", self.viewer.cam.elevation
      )
      self.viewer.cam.azimuth = getattr(self.cfg, "azimuth", self.viewer.cam.azimuth)
      self.viewer.cam.distance = getattr(self.cfg, "distance", self.viewer.cam.distance)
    else:
      self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
      self.viewer.cam.fixedcamid = -1
      self.viewer.cam.trackbodyid = -1

  # Reward plotting helpers.

  def _init_reward_plots(self, term_names: list[str]) -> None:
    """Create per-term figures and histories."""
    self._figures.clear()
    self._histories.clear()
    self._yrange.clear()
    for name in term_names:
      self._figures[name] = make_empty_figure(
        name,
        self._plot_cfg.grid_size,
        self._plot_cfg.init_yrange,
        self._plot_cfg.history,
        self._plot_cfg.background_alpha,
      )
      self._histories[name] = deque(maxlen=self._plot_cfg.history)
      self._yrange[name] = self._plot_cfg.init_yrange

  def _clear_histories(self) -> None:
    """Clear histories and reset figures."""
    for name in self._term_names:
      self._histories[name].clear()
      self._yrange[name] = self._plot_cfg.init_yrange
      fig = self._figures[name]
      fig.linepnt[0] = 0
      fig.range[1][0] = float(self._plot_cfg.init_yrange[0])
      fig.range[1][1] = float(self._plot_cfg.init_yrange[1])

  def _append_point(self, name: str, value: float) -> None:
    """Append a new point to the ring buffer."""
    if not np.isfinite(value):
      return
    self._histories[name].append(float(value))

  def _write_history_to_figure(self, name: str) -> None:
    """Copy history into figure and autoscale y-axis."""
    fig = self._figures[name]
    hist = self._histories[name]
    n = min(len(hist), self._plot_cfg.history)

    fig.linepnt[0] = n
    for i in range(n):
      fig.linedata[0][2 * i] = float(-i)
      fig.linedata[0][2 * i + 1] = float(hist[-1 - i])

    # Autoscale y-axis.
    if n >= 5:
      data = np.fromiter(hist, dtype=float, count=n)
      lo = float(np.percentile(data, self._plot_cfg.p_lo))
      hi = float(np.percentile(data, self._plot_cfg.p_hi))
      span = max(hi - lo, self._plot_cfg.min_span)
      lo -= self._plot_cfg.pad * span
      hi += self._plot_cfg.pad * span
    elif n >= 1:
      v = float(hist[-1])
      span = max(abs(v), 1e-3)
      lo, hi = v - span, v + span
    else:
      lo, hi = self._plot_cfg.init_yrange

    fig.range[1][0] = float(lo)
    fig.range[1][1] = float(hi)


def compute_viewports(
  num_plots: int,
  rect: mujoco.MjrRect,
  cfg: PlotCfg,
) -> list[mujoco.MjrRect]:
  """Lay plots in a strip on the right."""
  if num_plots <= 0:
    return []
  cols = 1 if num_plots <= cfg.max_rows_per_col else 2
  rows = min(cfg.max_rows_per_col, (num_plots + cols - 1) // cols)

  strip_w = int(rect.width * cfg.plot_strip_fraction)
  vp_w = strip_w // cols
  vp_h = rect.height // rows

  left0 = rect.left + rect.width - strip_w
  vps: list[mujoco.MjrRect] = []
  for idx in range(min(num_plots, cfg.max_viewports)):
    c = idx // rows
    r = idx % rows
    left = left0 + c * vp_w
    bottom = rect.bottom + rect.height - (r + 1) * vp_h
    vps.append(mujoco.MjrRect(left=left, bottom=bottom, width=vp_w, height=vp_h))
  return vps


def make_empty_figure(
  title: str,
  grid_size: tuple[int, int],
  yrange: tuple[float, float],
  history: int,
  alpha: float,
) -> mujoco.MjvFigure:
  fig = mujoco.MjvFigure()
  mujoco.mjv_defaultFigure(fig)
  fig.flg_extend = 1
  fig.gridsize[0] = grid_size[0]
  fig.gridsize[1] = grid_size[1]
  fig.range[1][0] = float(yrange[0])
  fig.range[1][1] = float(yrange[1])
  fig.figurergba[3] = alpha
  fig.title = title
  # Pre-fill x coordinates; y's will be written on update.
  for i in range(history):
    fig.linedata[0][2 * i] = -float(i)
    fig.linedata[0][2 * i + 1] = 0.0
  fig.linepnt[0] = 0
  return fig
