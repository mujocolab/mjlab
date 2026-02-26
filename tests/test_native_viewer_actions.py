"""Tests for native viewer custom action dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock

from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native.viewer import NativeMujocoViewer


def _make_viewer(num_envs: int = 3) -> NativeMujocoViewer:
  env = MagicMock()
  env.cfg.viewer.env_idx = 0
  env.unwrapped.num_envs = num_envs
  return NativeMujocoViewer(env, MagicMock())


def test_toggle_actions_dispatch_by_action_enum():
  v = _make_viewer()

  assert not v._show_plots
  assert v._handle_custom_action(ViewerAction.TOGGLE_PLOTS, None)
  assert v._show_plots

  assert v._show_debug_vis
  assert v._handle_custom_action(ViewerAction.TOGGLE_DEBUG_VIS, None)
  assert not v._show_debug_vis

  assert not v._show_all_envs
  assert v._handle_custom_action(ViewerAction.TOGGLE_SHOW_ALL_ENVS, None)
  assert v._show_all_envs


def test_prev_next_env_actions_wrap_and_succeed():
  v = _make_viewer(num_envs=3)
  v.env_idx = 0
  assert v._handle_custom_action(ViewerAction.PREV_ENV, None)
  assert v.env_idx == 2
  assert v._handle_custom_action(ViewerAction.NEXT_ENV, None)
  assert v.env_idx == 0
