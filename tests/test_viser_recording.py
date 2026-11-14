"""Tests for ViserPlayViewer .viser / .html rollout recording."""

from __future__ import annotations

from pathlib import Path

import msgspec
import mujoco
import pytest
import torch
import tyro
import zstandard
from conftest import get_test_device, load_fixture_xml

import mjlab
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import SceneCfg
from mjlab.scripts.play import PlayConfig
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer.viser.viewer import ViserPlayViewer


def _decode_viser_payload(data: bytes) -> dict:
  """Decompress and decode a binary .viser recording for inspection."""
  inner_size = int.from_bytes(data[:8], "little")
  decompressed = zstandard.ZstdDecompressor().decompress(
    data[8:], max_output_size=inner_size
  )
  assert len(decompressed) == inner_size
  msgpack_len = int.from_bytes(decompressed[:8], "little")
  return msgspec.msgpack.decode(decompressed[8 : 8 + msgpack_len])


def _make_wrapped_env(device: str) -> RslRlVecEnvWrapper:
  xml = load_fixture_xml("fixed_base_articulated")
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=("joint.*",), stiffness=20.0, damping=2.0
        ),
      ),
    ),
  )
  cfg = ManagerBasedRlEnvCfg(
    decimation=2,
    scene=SceneCfg(
      num_envs=1,
      terrain=TerrainEntityCfg(),
      entities={"robot": robot_cfg},
    ),
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=("joint.*",)
      )
    },
    observations={
      "actor": ObservationGroupCfg(
        terms={"joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel)}
      )
    },
  )
  raw_env = ManagerBasedRlEnv(cfg=cfg, device=device)
  return RslRlVecEnvWrapper(raw_env)


def test_viser_record_rollout_binary_and_html(tmp_path: Path) -> None:
  """record_rollout exports valid .viser binary and standalone .html recordings."""
  device = get_test_device()
  env = _make_wrapped_env(device)
  action_shape = env.unwrapped.action_space.shape

  def zero_policy(obs) -> torch.Tensor:
    del obs
    return torch.zeros(action_shape, device=device)

  viewer = ViserPlayViewer(env, zero_policy)
  try:
    viser_path = tmp_path / "rollout.viser"
    payload = viewer.record_rollout(viser_path, num_steps=4)
    assert viser_path.is_file()
    assert viser_path.read_bytes() == payload

    decoded = _decode_viser_payload(payload)
    assert decoded["durationSeconds"] == pytest.approx(4 * env.unwrapped.step_dt)
    assert len(decoded["messages"]) > 0
    assert not viewer.is_recording

    html_path = tmp_path / "rollout.html"
    html_bytes = viewer.record_rollout(html_path, num_steps=2)
    assert html_path.is_file()
    html_text = html_bytes.decode("utf-8")
    assert "window.__VISER_EMBED_DATA__" in html_text
  finally:
    viewer.close()
    env.close()


def test_viser_interactive_start_stop_recording() -> None:
  """start_recording and stop_recording track live viewer ticks cleanly."""
  device = get_test_device()
  env = _make_wrapped_env(device)
  action_shape = env.unwrapped.action_space.shape

  def step_policy(obs) -> torch.Tensor:
    del obs
    return torch.full(action_shape, 0.2, device=device)

  viewer = ViserPlayViewer(env, step_policy)
  try:
    viewer.setup()
    viewer.reset_environment()
    assert not viewer.is_recording

    viewer.start_recording()
    assert viewer.is_recording
    assert viewer._record_button is not None
    assert viewer._record_button.label == "Stop & Download"

    for _ in range(3):
      assert viewer._execute_step()
      # Advance twice so counter % 2 == 0 triggers a 30Hz scene submit.
      viewer.sync_env_to_viewer()
      viewer.sync_env_to_viewer()

    # Wait for queued background scene updates before stopping.
    viewer._threadpool.submit(lambda: None).result(timeout=5.0)
    data = viewer.stop_recording()
    assert not viewer.is_recording
    assert viewer._record_button.label == "Start Recording"

    decoded = _decode_viser_payload(data)
    assert decoded["durationSeconds"] == pytest.approx(3 * env.unwrapped.step_dt)
  finally:
    viewer.close()
    env.close()


def test_play_config_viser_recording_cli_flags() -> None:
  """PlayConfig exposes --viser-record-file and --viser-record-steps via tyro."""
  parsed = tyro.cli(
    PlayConfig,
    default=PlayConfig(),
    args=["--viser-record-file", "demo.viser", "--viser-record-steps", "75"],
    config=mjlab.TYRO_FLAGS,
  )
  assert parsed.viser_record_file == "demo.viser"
  assert parsed.viser_record_steps == 75
