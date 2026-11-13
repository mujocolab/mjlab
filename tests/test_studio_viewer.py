"""Tests for the MuJoCo Studio viewer integration."""

import sys
from dataclasses import dataclass
from enum import IntEnum
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import torch

from mjlab.viewer.base import ViewerAction
from mjlab.viewer.studio import StudioMujocoViewer
from mjlab.viewer.viewer_config import ViewerConfig

_FREE_BODY_XML = """
<mujoco>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="0.1" rgba="1 0 0 1"/>
    </body>
    <body name="target" mocap="true" pos="0 0 0">
      <geom name="target_geom" type="sphere" size="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="push" joint="ball_free" gear="1 0 0 0 0 0"/>
  </actuator>
</mujoco>
"""


class _FakePauseState(IntEnum):
  UNPAUSED = 0
  NORMAL_PAUSED = 1


class _FakeMessage:
  pass


@dataclass(frozen=True)
class _FakeResetEvent(_FakeMessage):
  key: int | None = None


@dataclass(frozen=True)
class _FakeRequestPauseEvent(_FakeMessage):
  pause_state: _FakePauseState


@dataclass(frozen=True)
class _FakeSingleStepEvent(_FakeMessage):
  pass


@dataclass(frozen=True)
class _FakeModelEvent(_FakeMessage):
  model: mujoco.MjModel
  path: str = ""


@dataclass(frozen=True)
class _FakeStateSnapshot(_FakeMessage):
  state: np.ndarray
  state_sig: int


class _FakeSimPlugin:
  pass


class _FakeApp:
  pass


@dataclass
class _FakeStudioConfig:
  title: str = ""
  width: int = 1200
  height: int = 800
  gfx: str = ""
  http_port: int = 0
  open_browser: bool = False
  hot_reload: str = ""


class _FakeHandle:
  def __init__(self, config: _FakeStudioConfig, plugins: list[Any]):
    self.config = config
    self.plugins = plugins
    self.running = True
    self.sent_messages: list[Any] = []
    self.sync_calls = 0
    self.closed = False
    self.model: mujoco.MjModel | None = None
    self.data: mujoco.MjData | None = None

  def set_model(self, model: mujoco.MjModel, data: mujoco.MjData | None = None) -> None:
    self.model = model
    self.data = data

  def is_running(self) -> bool:
    return self.running

  def sync(self) -> tuple[mujoco.MjModel | None, mujoco.MjData | None]:
    self.sync_calls += 1
    return self.model, self.data

  def send_to_viewer(self, msg: Any) -> None:
    self.sent_messages.append(msg)

  def close(self) -> None:
    self.running = False
    self.closed = True


def _install_fake_studio(monkeypatch: pytest.MonkeyPatch) -> list[_FakeHandle]:
  handles: list[_FakeHandle] = []

  studio_pkg = ModuleType("mujoco.studio")
  app_mod = ModuleType("mujoco.studio.app")
  app_mod.App = _FakeApp  # ty: ignore[missing-attribute]

  launch_mod = ModuleType("mujoco.studio.launch_passive")

  def _launch(config: _FakeStudioConfig, *, plugins: Any = None) -> _FakeHandle:
    handle = _FakeHandle(config, list(plugins or []))
    handles.append(handle)
    return handle

  launch_mod.launch = _launch  # ty: ignore[missing-attribute]

  messages_mod = ModuleType("mujoco.studio.messages")
  messages_mod.Message = _FakeMessage  # ty: ignore[missing-attribute]
  messages_mod.ResetEvent = _FakeResetEvent  # ty: ignore[missing-attribute]
  messages_mod.RequestPauseEvent = _FakeRequestPauseEvent  # ty: ignore[missing-attribute]
  messages_mod.SingleStepEvent = _FakeSingleStepEvent  # ty: ignore[missing-attribute]
  messages_mod.ModelEvent = _FakeModelEvent  # ty: ignore[missing-attribute]
  messages_mod.StateSnapshot = _FakeStateSnapshot  # ty: ignore[missing-attribute]
  messages_mod.handler = lambda fn=None, **_kw: fn if fn is not None else (lambda f: f)  # ty: ignore[missing-attribute]

  plugin_mod = ModuleType("mujoco.studio.plugin")
  plugin_mod.SimPlugin = _FakeSimPlugin  # ty: ignore[missing-attribute]

  sim_mod = ModuleType("mujoco.studio.sim")
  sim_mod.PauseState = _FakePauseState  # ty: ignore[missing-attribute]

  vp_mod = ModuleType("mujoco.studio.viewer_protocol")
  vp_mod.ViewerConfig = _FakeStudioConfig  # ty: ignore[missing-attribute]

  monkeypatch.setitem(sys.modules, "mujoco.studio", studio_pkg)
  monkeypatch.setitem(sys.modules, "mujoco.studio.app", app_mod)
  monkeypatch.setitem(sys.modules, "mujoco.studio.launch_passive", launch_mod)
  monkeypatch.setitem(sys.modules, "mujoco.studio.messages", messages_mod)
  monkeypatch.setitem(sys.modules, "mujoco.studio.plugin", plugin_mod)
  monkeypatch.setitem(sys.modules, "mujoco.studio.sim", sim_mod)
  monkeypatch.setitem(sys.modules, "mujoco.studio.viewer_protocol", vp_mod)
  return handles


def _make_mock_env(num_envs: int = 2, expand_rgba: bool = False) -> Mock:
  mjm = mujoco.MjSpec.from_string(_FREE_BODY_XML).compile()
  mjd = mujoco.MjData(mjm)

  qpos = torch.zeros((num_envs, mjm.nq), dtype=torch.float32)
  qpos[:, 3] = 1.0  # Identity quaternion w.
  qpos[0, 2] = 1.5
  qpos[1, 2] = 2.5
  qvel = torch.zeros((num_envs, mjm.nv), dtype=torch.float32)
  ctrl = torch.full((num_envs, mjm.nu), 3.25, dtype=torch.float32)
  mocap_pos = torch.zeros((num_envs, mjm.nmocap, 3), dtype=torch.float32)
  mocap_pos[0, 0] = torch.tensor([0.5, -0.25, 0.75])
  mocap_quat = torch.zeros((num_envs, mjm.nmocap, 4), dtype=torch.float32)
  mocap_quat[:, :, 0] = 1.0
  xfrc_applied = torch.zeros((num_envs, mjm.nbody, 6), dtype=torch.float32)
  qfrc_applied = torch.zeros((num_envs, mjm.nv), dtype=torch.float32)

  geom_rgba = torch.tile(
    torch.as_tensor(mjm.geom_rgba, dtype=torch.float32).unsqueeze(0),
    (num_envs, 1, 1),
  )
  geom_rgba[0, 0] = torch.tensor([0.1, 0.8, 0.2, 1.0])
  geom_rgba[1, 0] = torch.tensor([0.9, 0.2, 0.7, 1.0])

  sim = SimpleNamespace(
    mj_model=mjm,
    mj_data=mjd,
    data=SimpleNamespace(
      qpos=qpos,
      qvel=qvel,
      ctrl=ctrl,
      mocap_pos=mocap_pos,
      mocap_quat=mocap_quat,
      xfrc_applied=xfrc_applied,
      qfrc_applied=qfrc_applied,
    ),
    model=SimpleNamespace(geom_rgba=geom_rgba),
    expanded_fields={"geom_rgba"} if expand_rgba else set(),
    forward=Mock(),
  )

  env = Mock()
  env.num_envs = num_envs
  env.device = "cpu"
  env.cfg = SimpleNamespace(viewer=ViewerConfig(width=1024, height=768))
  env.unwrapped = SimpleNamespace(num_envs=num_envs, step_dt=0.02, sim=sim)
  env.get_observations = Mock(return_value=torch.zeros((num_envs, 4)))
  env.step = Mock(
    return_value=(
      torch.zeros((num_envs, 4)),
      torch.zeros(num_envs),
      torch.zeros(num_envs, dtype=torch.bool),
      {},
    )
  )
  return env


def test_studio_viewer_missing_extra_raises_helpful_error(
  monkeypatch: pytest.MonkeyPatch,
):
  """Missing mujoco.studio raises ModuleNotFoundError with install command."""
  for name in list(sys.modules):
    if name == "mujoco.studio" or name.startswith("mujoco.studio."):
      monkeypatch.setitem(sys.modules, name, None)
  monkeypatch.setitem(sys.modules, "mujoco.studio.app", None)

  env = _make_mock_env()
  viewer = StudioMujocoViewer(env, lambda obs: torch.zeros((obs.shape[0], 1)))
  with pytest.raises(ModuleNotFoundError, match="uv sync --extra studio"):
    viewer.setup()


def test_studio_viewer_rejects_unknown_gfx_mode():
  env = _make_mock_env()
  with pytest.raises(ValueError, match="Unsupported Studio gfx mode"):
    StudioMujocoViewer(
      env,
      lambda obs: torch.zeros((obs.shape[0], 1)),
      gfx="directx9",  # ty: ignore[invalid-argument-type]
    )


def test_studio_viewer_desktop_and_web_launch_and_state_sync(
  monkeypatch: pytest.MonkeyPatch,
):
  """Desktop and web modes pass config, sync mocap/ctrl/qpos, and transfer forces."""
  handles = _install_fake_studio(monkeypatch)
  env = _make_mock_env(num_envs=2, expand_rgba=True)
  policy = lambda obs: torch.zeros((obs.shape[0], 1))  # noqa: E731

  viewer = StudioMujocoViewer(env, policy, gfx="web", http_port=8123, open_browser=True)
  viewer.setup()
  assert viewer.is_running()
  assert len(handles) == 1
  handle = handles[0]
  assert handle.config.gfx == "web"
  assert handle.config.http_port == 8123
  assert handle.config.open_browser is True
  assert handle.config.width == 1024
  assert handle.config.height == 768

  # Initial setup synced env 0 qpos, ctrl, mocap_pos, and DR geom_rgba.
  assert viewer.mjd is not None and viewer.mjm is not None
  assert viewer.mjd.qpos[2] == pytest.approx(1.5)
  assert viewer.mjd.ctrl[0] == pytest.approx(3.25)
  assert np.allclose(viewer.mjd.mocap_pos[0], [0.5, -0.25, 0.75])
  assert np.allclose(viewer.mjm.geom_rgba[0], [0.1, 0.8, 0.2, 1.0])

  # Unchanged DR fields do not emit redundant ModelEvent on sync_env_to_viewer.
  handle.sent_messages.clear()
  viewer.sync_env_to_viewer()
  assert handle.sync_calls == 1
  assert not any(isinstance(m, _FakeModelEvent) for m in handle.sent_messages)
  assert any(isinstance(m, _FakeStateSnapshot) for m in handle.sent_messages)

  # Switching environment emits ModelEvent with env 1's DR geom_rgba.
  handle.sent_messages.clear()
  viewer._handle_custom_action(ViewerAction.NEXT_ENV, None)
  viewer.sync_env_to_viewer()
  assert viewer.env_idx == 1
  assert np.allclose(viewer.mjm.geom_rgba[0], [0.9, 0.2, 0.7, 1.0])
  assert any(isinstance(m, _FakeModelEvent) for m in handle.sent_messages)

  # Perturbation force in mjd.qfrc_applied transfers to GPU and clears host buffer.
  viewer.mjd.qfrc_applied[2] = 9.5
  viewer.sync_viewer_to_env()
  assert env.unwrapped.sim.data.qfrc_applied[1, 2].item() == pytest.approx(9.5)
  assert viewer.mjd.qfrc_applied[2] == 0.0

  # SimPlugin forwards Studio UI reset/pause/single-step events into BaseViewer queue.
  bridge = next(p for p in handle.plugins if isinstance(p, _FakeSimPlugin))
  bridge.on_pause(_FakeRequestPauseEvent(_FakePauseState.NORMAL_PAUSED))
  bridge.on_single_step(_FakeSingleStepEvent())
  bridge.on_reset(_FakeResetEvent())
  queued = [a for a, _ in viewer._actions]
  assert queued == [
    ViewerAction.TOGGLE_PAUSE,
    ViewerAction.SINGLE_STEP,
    ViewerAction.RESET,
  ]

  viewer.close()
  assert handle.closed
  assert not viewer.is_running()
