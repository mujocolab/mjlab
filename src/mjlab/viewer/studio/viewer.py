"""Environment viewer built on MuJoCo Studio (Filament PBR & Web WASM).

MuJoCo 4.0.1 introduces ``mujoco.studio`` (distributed via ``mujoco-studio``),
providing physically based rendering (PBR) powered by Google Filament in both a
desktop GUI window and a browser-streamed WebAssembly client (``gfx="web"``).
"""

from __future__ import annotations

import importlib
from threading import Lock
from typing import Any, Literal

import mujoco
import numpy as np
import torch

from mjlab.viewer.base import (
  BaseViewer,
  EnvProtocol,
  PolicyProtocol,
  VerbosityLevel,
  ViewerAction,
)
from mjlab.viewer.model_sync import (
  VIEWER_MODEL_FIELDS,
  disable_model_sameframe_shortcuts,
  sync_model_fields,
)

StudioGfxMode = Literal[
  "",
  "opengl",
  "opengl_headless",
  "opengl_software",
  "vulkan",
  "vulkan_software",
  "web",
  "webgl",
]

_STUDIO_GFX_MODES: frozenset[str] = frozenset(
  {
    "",
    "opengl",
    "opengl_headless",
    "opengl_software",
    "vulkan",
    "vulkan_software",
    "web",
    "webgl",
  }
)

_STUDIO_STATE_SIG: int = int(
  mujoco.mjtState.mjSTATE_INTEGRATION.value
  | mujoco.mjtState.mjSTATE_CTRL.value
  | mujoco.mjtState.mjSTATE_MOCAP_POS.value
  | mujoco.mjtState.mjSTATE_MOCAP_QUAT.value
  | mujoco.mjtState.mjSTATE_XFRC_APPLIED.value
)


def _load_studio_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
  """Import optional ``mujoco.studio`` modules on demand."""
  try:
    studio_app = importlib.import_module("mujoco.studio.app")
    launch_passive = importlib.import_module("mujoco.studio.launch_passive")
    messages = importlib.import_module("mujoco.studio.messages")
    studio_plugin = importlib.import_module("mujoco.studio.plugin")
    studio_sim = importlib.import_module("mujoco.studio.sim")
    viewer_protocol = importlib.import_module("mujoco.studio.viewer_protocol")
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "MuJoCo Studio viewer requires the optional 'studio' extra. "
      "Install it with `uv sync --extra studio` or `pip install 'mjlab[studio]'`."
    ) from exc
  return (
    studio_app,
    launch_passive,
    messages,
    studio_plugin,
    studio_sim,
    viewer_protocol,
  )


def _make_sim_bridge(
  owner: StudioMujocoViewer,
  studio_plugin: Any,
  messages: Any,
  studio_sim: Any,
) -> Any:
  """Create a Studio SimPlugin forwarding UI reset/pause/step events to mjlab."""

  def _on_reset(_self: Any, _event: Any) -> None:
    owner.request_reset()

  _on_reset.__annotations__ = {"_event": messages.ResetEvent, "return": None}

  def _on_pause(_self: Any, event: Any) -> None:
    if event.pause_state == studio_sim.PauseState.UNPAUSED:
      if owner._is_paused:
        owner.request_toggle_pause()
    else:
      if not owner._is_paused:
        owner.request_toggle_pause()

  _on_pause.__annotations__ = {
    "event": messages.RequestPauseEvent,
    "return": None,
  }

  def _on_single_step(_self: Any, _event: Any) -> None:
    owner.request_single_step()

  _on_single_step.__annotations__ = {
    "_event": messages.SingleStepEvent,
    "return": None,
  }

  bridge_cls = type(
    "_MjlabStudioSimBridge",
    (studio_plugin.SimPlugin,),
    {
      "on_reset": messages.handler(_on_reset),
      "on_pause": messages.handler(_on_pause),
      "on_single_step": messages.handler(_on_single_step),
    },
  )
  return bridge_cls()


class StudioMujocoViewer(BaseViewer):
  """Interactive viewer backed by ``mujoco.studio`` (Filament PBR & Web WASM)."""

  def __init__(
    self,
    env: EnvProtocol,
    policy: PolicyProtocol,
    frame_rate: float = 60.0,
    gfx: StudioGfxMode = "",
    http_port: int = 0,
    open_browser: bool = False,
    title: str = "mjlab Studio",
    enable_perturbations: bool = True,
    verbosity: VerbosityLevel = VerbosityLevel.SILENT,
  ):
    if gfx not in _STUDIO_GFX_MODES:
      raise ValueError(
        f"Unsupported Studio gfx mode {gfx!r}; expected one of "
        f"{sorted(_STUDIO_GFX_MODES)}."
      )
    super().__init__(env, policy, frame_rate, verbosity)
    self.gfx: StudioGfxMode = gfx
    self.http_port = http_port
    self.open_browser = open_browser
    self.title = title
    self.enable_perturbations = enable_perturbations

    self.mjm: mujoco.MjModel | None = None
    self.mjd: mujoco.MjData | None = None
    self.handle: Any = None
    self._messages: Any = None

    self.env_idx = self.cfg.env_idx
    self._mj_lock = Lock()
    self._synced_env_idx: int | None = None
    self._model_snapshot: dict[str, np.ndarray] = {}
    self._state_buf: np.ndarray | None = None

  def setup(self) -> None:
    """Initialize host MuJoCo buffers and launch MuJoCo Studio."""
    (
      studio_app,
      launch_passive,
      messages,
      studio_plugin,
      studio_sim,
      viewer_protocol,
    ) = _load_studio_modules()
    self._messages = messages

    sim = self.env.unwrapped.sim
    self.mjm = sim.mj_model
    self.mjd = sim.mj_data
    assert self.mjm is not None and self.mjd is not None

    disable_model_sameframe_shortcuts(self.mjm)
    if self.cfg.fovy is not None:
      self.mjm.vis.global_.fovy = self.cfg.fovy

    self._sync_env_state_to_mjdata(self.mjd, sim.data, self.env_idx)
    self._sync_model_fields_if_changed(sim, self.env_idx, send_event=False)
    mujoco.mj_forward(self.mjm, self.mjd)

    studio_cfg = viewer_protocol.ViewerConfig(
      title=self.title,
      width=self.cfg.width or 1200,
      height=self.cfg.height or 800,
      gfx=self.gfx,
      http_port=self.http_port,
      open_browser=self.open_browser,
    )
    bridge = _make_sim_bridge(self, studio_plugin, messages, studio_sim)
    self.handle = launch_passive.launch(
      studio_cfg,
      plugins=[studio_app.App(), bridge],
    )
    self.handle.set_model(self.mjm, data=self.mjd)
    self._send_full_state_snapshot()

  def is_running(self) -> bool:
    return bool(self.handle is not None and self.handle.is_running())

  def sync_env_to_viewer(self) -> None:
    """Push current GPU environment state and DR model updates to Studio."""
    handle = self.handle
    if handle is None or not handle.is_running():
      return
    assert self.mjm is not None and self.mjd is not None

    with self._mj_lock:
      sim = self.env.unwrapped.sim
      self._sync_env_state_to_mjdata(self.mjd, sim.data, self.env_idx)
      self._sync_model_fields_if_changed(sim, self.env_idx, send_event=True)
      mujoco.mj_forward(self.mjm, self.mjd)

      handle.sync()
      if handle.is_running():
        self._send_full_state_snapshot()

  def sync_viewer_to_env(self) -> None:
    """Transfer Studio mouse perturbation generalized forces into GPU sim."""
    if self.mjm is None or self.mjd is None or self.mjm.nv == 0:
      return
    with self._mj_lock:
      sim_data = self.env.unwrapped.sim.data
      if not self.enable_perturbations:
        sim_data.qfrc_applied[self.env_idx] = 0.0
        return
      sim_data.qfrc_applied[self.env_idx] = torch.as_tensor(
        self.mjd.qfrc_applied,
        dtype=torch.float32,
        device=self.env.device,
      )
      self.mjd.qfrc_applied[:] = 0.0

  def _forward_paused(self) -> None:
    self.env.unwrapped.sim.forward()

  def _handle_custom_action(self, action: ViewerAction, payload: Any | None) -> bool:
    del payload
    num_envs = self.env.unwrapped.num_envs
    if action == ViewerAction.PREV_ENV and num_envs > 1:
      self.env_idx = (self.env_idx - 1) % num_envs
      self.log(f"[INFO] Switched to environment {self.env_idx}", VerbosityLevel.INFO)
      return True
    if action == ViewerAction.NEXT_ENV and num_envs > 1:
      self.env_idx = (self.env_idx + 1) % num_envs
      self.log(f"[INFO] Switched to environment {self.env_idx}", VerbosityLevel.INFO)
      return True
    return False

  def close(self) -> None:
    """Close Studio viewer handle and release resources."""
    handle = self.handle
    self.handle = None
    if handle is not None:
      handle.close()

  def _sync_env_state_to_mjdata(
    self, target_data: mujoco.MjData, sim_data: Any, env_idx: int
  ) -> None:
    assert self.mjm is not None
    if self.mjm.nq > 0:
      target_data.qpos[:] = sim_data.qpos[env_idx].cpu().numpy()
    if self.mjm.nv > 0:
      target_data.qvel[:] = sim_data.qvel[env_idx].cpu().numpy()
    if self.mjm.nu > 0:
      target_data.ctrl[:] = sim_data.ctrl[env_idx].cpu().numpy()
    if self.mjm.nmocap > 0:
      target_data.mocap_pos[:] = sim_data.mocap_pos[env_idx].cpu().numpy()
      target_data.mocap_quat[:] = sim_data.mocap_quat[env_idx].cpu().numpy()
    target_data.xfrc_applied[:] = sim_data.xfrc_applied[env_idx].cpu().numpy()

  def _sync_model_fields_if_changed(
    self, sim: Any, env_idx: int, *, send_event: bool
  ) -> bool:
    """Sync visual DR model fields and notify Studio only when values change."""
    assert self.mjm is not None
    fields = sim.expanded_fields & VIEWER_MODEL_FIELDS
    if not fields:
      self._synced_env_idx = env_idx
      return False

    changed = self._synced_env_idx != env_idx
    for field_name in fields:
      dst = getattr(self.mjm, field_name)
      if dst.size == 0:
        continue
      src = getattr(sim.model, field_name)[env_idx].cpu().numpy().reshape(dst.shape)
      prev = self._model_snapshot.get(field_name)
      if prev is None or not np.array_equal(prev, src):
        changed = True
        self._model_snapshot[field_name] = src.copy()

    if changed:
      sync_model_fields(self.mjm, sim.model, fields, env_idx)
      self._synced_env_idx = env_idx
      if send_event and self.handle is not None and self._messages is not None:
        self.handle.send_to_viewer(self._messages.ModelEvent(model=self.mjm))
    return changed

  def _send_full_state_snapshot(self) -> None:
    """Send full render-relevant MjData state (including mocap/ctrl/xfrc)."""
    if self.handle is None or self.mjm is None or self.mjd is None:
      return
    assert self._messages is not None
    size = mujoco.mj_stateSize(self.mjm, _STUDIO_STATE_SIG)
    if self._state_buf is None or self._state_buf.shape != (size,):
      self._state_buf = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(self.mjm, self.mjd, self._state_buf, _STUDIO_STATE_SIG)
    self.handle.send_to_viewer(
      self._messages.StateSnapshot(
        state=self._state_buf.copy(), state_sig=_STUDIO_STATE_SIG
      )
    )
