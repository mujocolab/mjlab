"""Tests for cross-platform GL setup and viewer display detection."""

from pathlib import Path

import pytest

from mjlab import _configure_linux_gl_env, _is_wsl
from mjlab.scripts.play import _has_display


def test_is_wsl_detects_env_vars(tmp_path: Path):
  missing = tmp_path / "missing_osrelease"
  assert _is_wsl(env={"WSL_DISTRO_NAME": "Ubuntu"}, osrelease_path=missing)
  assert _is_wsl(env={"WSL_INTEROP": "/run/WSL/1_interop"}, osrelease_path=missing)
  assert not _is_wsl(env={}, osrelease_path=missing)


def test_is_wsl_detects_proc_osrelease(tmp_path: Path):
  wsl_osrelease = tmp_path / "osrelease_wsl"
  wsl_osrelease.write_text("5.15.167.4-microsoft-standard-WSL2\n", encoding="utf-8")
  linux_osrelease = tmp_path / "osrelease_linux"
  linux_osrelease.write_text("6.8.0-45-generic\n", encoding="utf-8")

  assert _is_wsl(env={}, osrelease_path=wsl_osrelease)
  assert not _is_wsl(env={}, osrelease_path=linux_osrelease)


def test_configure_linux_gl_env_sets_egl_and_wsl_d3d12(tmp_path: Path):
  wsl_osrelease = tmp_path / "osrelease_wsl"
  wsl_osrelease.write_text("6.6.87.2-microsoft-standard-WSL2\n", encoding="utf-8")
  linux_osrelease = tmp_path / "osrelease_linux"
  linux_osrelease.write_text("6.8.0-45-generic\n", encoding="utf-8")

  # Native Linux: sets MUJOCO_GL=egl, leaves GALLIUM_DRIVER unset.
  linux_env: dict[str, str] = {}
  _configure_linux_gl_env(
    platform="linux", env=linux_env, osrelease_path=linux_osrelease
  )
  assert linux_env == {"MUJOCO_GL": "egl"}

  # WSL2: sets both MUJOCO_GL=egl and GALLIUM_DRIVER=d3d12.
  wsl_env: dict[str, str] = {}
  _configure_linux_gl_env(platform="linux", env=wsl_env, osrelease_path=wsl_osrelease)
  assert wsl_env == {"MUJOCO_GL": "egl", "GALLIUM_DRIVER": "d3d12"}

  # Explicit user overrides are preserved.
  override_env = {"MUJOCO_GL": "osmesa", "GALLIUM_DRIVER": "zink"}
  _configure_linux_gl_env(
    platform="linux", env=override_env, osrelease_path=wsl_osrelease
  )
  assert override_env == {"MUJOCO_GL": "osmesa", "GALLIUM_DRIVER": "zink"}

  # Non-Linux platforms are untouched.
  for plat in ("win32", "darwin"):
    non_linux_env: dict[str, str] = {}
    _configure_linux_gl_env(
      platform=plat, env=non_linux_env, osrelease_path=wsl_osrelease
    )
    assert non_linux_env == {}


@pytest.mark.parametrize(
  "platform,env,expected",
  [
    ("win32", {}, True),
    ("win32", {"SSH_CONNECTION": "10.0.0.1 1234 10.0.0.2 22"}, False),
    ("darwin", {}, True),
    ("darwin", {"SSH_TTY": "/dev/ttys001"}, False),
    ("linux", {}, False),
    ("linux", {"DISPLAY": ":0"}, True),
    ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, True),
  ],
)
def test_has_display_across_platforms(
  platform: str, env: dict[str, str], expected: bool
):
  assert _has_display(platform=platform, env=env) is expected
