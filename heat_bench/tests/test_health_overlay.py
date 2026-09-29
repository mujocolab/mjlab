"""Tests for RobotHealthOverlay and NodeBarPanel (no live Viser server needed).

Follows tests/test_viser_update_policy.py's convention: a MagicMock stands
in for the Viser server, since every widget-creation call just becomes a
MagicMock method call returning a MagicMock handle that tolerates
arbitrary attribute assignment (``.data = ...``, ``.value = ...``).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from heat_bench.viewer.health_overlay import RobotHealthOverlay
from heat_bench.viewer.node_bar_panel import NodeBarPanel, _severity

JOINT_NAMES = [
  f"{leg}_{part}_joint"
  for leg in ("FR", "FL", "RR", "RL")
  for part in ("hip", "thigh", "calf")
]


def _fake_term(num_envs: int = 2, initial_soc: float = 1.0) -> SimpleNamespace:
  return SimpleNamespace(
    joint_names=list(JOINT_NAMES),
    thermal=SimpleNamespace(T=torch.full((num_envs, 14), 25.0)),
    last_current=torch.zeros(num_envs, 12),
    last_torque=torch.zeros(num_envs, 12),
    battery=SimpleNamespace(
      soc=torch.full((num_envs,), initial_soc),
      bus_voltage=torch.full((num_envs,), 24.0),
      cum_wh=torch.zeros(num_envs),
      capacity_loss_pct=torch.zeros(num_envs),
    ),
  )


def test_setup_tab_creates_all_widgets():
  server = MagicMock()
  overlay = RobotHealthOverlay(server, _fake_term())
  overlay.setup_tab()

  assert server.gui.add_uplot.call_count == 4  # thermal, current, torque, bus voltage
  assert server.gui.add_html.call_count == 3  # thermal, current, torque bar panels
  assert server.gui.add_progress_bar.call_count == 1
  assert (
    server.gui.add_number.call_count == 3
  )  # SoC %, cumulative energy, capacity loss


def test_update_grows_history_and_pushes_values():
  server = MagicMock()
  term = _fake_term()
  overlay = RobotHealthOverlay(server, term)
  overlay.setup_tab()

  term.thermal.T[0, 0] = 40.0
  overlay.update(paused=False, env_idx=0)
  overlay.update(paused=False, env_idx=0)

  assert len(overlay._thermal_chart.history[0]) == 2
  assert overlay._thermal_chart.history[0][-1] == 40.0
  assert len(overlay._current_chart.history[0]) == 2
  assert len(overlay._torque_chart.history[0]) == 2


def test_update_skips_when_paused():
  server = MagicMock()
  overlay = RobotHealthOverlay(server, _fake_term())
  overlay.setup_tab()

  overlay.update(paused=True, env_idx=0)
  assert len(overlay._thermal_chart.history[0]) == 0


def test_on_env_switch_clears_histories():
  server = MagicMock()
  overlay = RobotHealthOverlay(server, _fake_term())
  overlay.setup_tab()

  overlay.update(paused=False, env_idx=0)
  assert len(overlay._thermal_chart.history[0]) == 1

  overlay.on_env_switch()
  assert len(overlay._thermal_chart.history[0]) == 0


def test_node_names_use_real_joint_names_not_placeholders():
  server = MagicMock()
  overlay = RobotHealthOverlay(server, _fake_term())

  assert overlay.joint_names == JOINT_NAMES
  assert overlay.thermal_node_names == JOINT_NAMES + ["chassis", "ambient"]
  assert "joint_0" not in overlay.thermal_node_names


def test_soc_bar_initializes_from_live_soc_not_hardcoded_100():
  server = MagicMock()
  overlay = RobotHealthOverlay(server, _fake_term(initial_soc=0.6))
  overlay.setup_tab()

  _, kwargs = server.gui.add_progress_bar.call_args
  assert kwargs["value"] == pytest.approx(60.0, abs=1e-3)

  _, kwargs = server.gui.add_number.call_args_list[0]
  assert kwargs["initial_value"] == pytest.approx(60.0, abs=1e-3)


def test_soc_readout_tracks_soc_bar_each_frame():
  server = MagicMock()
  # add_number is called twice (SoC readout, cumulative energy); give each
  # call its own handle so asserting on one doesn't read the other's value.
  server.gui.add_number.side_effect = lambda *a, **k: MagicMock()
  term = _fake_term(initial_soc=1.0)
  overlay = RobotHealthOverlay(server, term)
  overlay.setup_tab()

  term.battery.soc[:] = 0.42
  overlay.update(paused=False, env_idx=0)

  assert overlay._soc_bar is not None
  assert overlay._soc_readout is not None
  assert overlay._soc_bar.value == pytest.approx(42.0, abs=1e-3)
  assert overlay._soc_readout.value == pytest.approx(42.0, abs=1e-3)


def test_node_bar_panel_severity_and_update():
  server = MagicMock()
  panel = NodeBarPanel(server, ["a", "b"], low=0.0, high=10.0, unit="C")

  assert _severity(5.0, 0.0, 10.0) == 0.0  # Within range.
  assert _severity(15.0, 0.0, 10.0) == 0.5  # One-half range-width over.
  assert _severity(20.0, 0.0, 10.0) == 1.0  # Clamped at one full range-width over.

  panel.update(np.array([5.0, 15.0]))
  assert panel._values == [5.0, 15.0]

  panel.clear()
  assert panel._values == [0.0, 0.0]
