"""Tests for NaN guard functionality."""

import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.nan_guard import NanGuardCfg

wp.config.quiet = True


def get_test_device():
  """Get device for testing, preferring CUDA if available."""
  if torch.cuda.is_available():
    return "cuda:0"
  return "cpu"


@pytest.fixture
def simple_model():
  """Create a simple MuJoCo model for testing."""
  xml = """
  <mujoco>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>
  </mujoco>
  """
  spec = mujoco.MjSpec.from_string(xml)
  return spec.compile()


def test_nan_guard_disabled_by_default(simple_model):
  """NaN guard should be disabled by default with no overhead."""
  cfg = SimulationCfg()
  sim = Simulation(num_envs=2, cfg=cfg, model=simple_model, device=get_test_device())

  assert not sim.nan_guard.enabled
  sim.step()  # Should not trigger any capture.
  sim.close()


def test_nan_guard_captures_and_dumps_on_nan(simple_model):
  """NaN guard should capture states and dump when NaN detected."""
  with tempfile.TemporaryDirectory() as tmpdir:
    cfg = SimulationCfg(
      nan_guard=NanGuardCfg(
        enabled=True,
        buffer_size=5,
        output_dir=tmpdir,
        max_envs_to_capture=2,
      )
    )
    sim = Simulation(num_envs=4, cfg=cfg, model=simple_model, device=get_test_device())

    # Run a few steps to populate buffer.
    for _ in range(3):
      sim.step()

    # Inject NaN into environment 1.
    sim.data.qpos[1, 0] = float("nan")

    # Next step should trigger dump.
    sim.step()

    # Check that dump file was created.
    dump_files = list(Path(tmpdir).glob("nan_dump_*.npz"))
    assert len(dump_files) == 1

    # Load and inspect the dump.
    dump = np.load(dump_files[0], allow_pickle=True)
    metadata = dump["_metadata"].item()

    assert metadata["num_envs_total"] == 4
    assert metadata["num_envs_captured"] == 2
    assert 1 in metadata["nan_env_ids"]
    assert metadata["buffer_size"] == 4  # 3 clean steps + 1 with NaN injected.

    # Check that states were captured.
    assert "states_step_000000" in dump
    assert "states_step_000001" in dump
    assert "states_step_000002" in dump
    assert "states_step_000003" in dump  # State with NaN injected

    # Verify state shape: (num_envs_captured, state_size).
    state = dump["states_step_000000"]
    assert state.shape[0] == 2  # Only captured 2 envs.

    sim.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_nan_guard_detects_correct_env_ids(simple_model):
  """NaN guard should correctly identify which environments have NaN/Inf."""
  with tempfile.TemporaryDirectory() as tmpdir:
    cfg = SimulationCfg(
      nan_guard=NanGuardCfg(enabled=True, buffer_size=5, output_dir=tmpdir)
    )
    sim = Simulation(num_envs=10, cfg=cfg, model=simple_model, device=get_test_device())

    # Run a few steps to populate buffer.
    for _ in range(3):
      sim.step()

    # Inject NaN/Inf into environments 2, 5, and 7 (in state variables).
    sim.data.qpos[2, 0] = float("nan")
    sim.data.qvel[5, 1] = float("nan")
    sim.data.qvel[7, 2] = float("inf")

    # Next step should trigger dump.
    sim.step()

    # Load and inspect the dump.
    dump_files = list(Path(tmpdir).glob("nan_dump_*.npz"))
    assert len(dump_files) == 1

    dump = np.load(dump_files[0], allow_pickle=True)
    metadata = dump["_metadata"].item()

    # Should detect exactly the environments with NaN/Inf.
    nan_env_ids = set(metadata["nan_env_ids"])
    assert nan_env_ids == {2, 5, 7}, f"Expected {{2, 5, 7}}, got {nan_env_ids}"

    sim.close()


def test_nan_guard_saves_model(simple_model):
  """NaN guard should save model file alongside state dump."""
  with tempfile.TemporaryDirectory() as tmpdir:
    cfg = SimulationCfg(
      nan_guard=NanGuardCfg(enabled=True, buffer_size=5, output_dir=tmpdir)
    )
    sim = Simulation(num_envs=2, cfg=cfg, model=simple_model, device=get_test_device())

    # Inject NaN and trigger dump.
    sim.data.qpos[0, 0] = float("nan")
    sim.step()

    # Check that both dump and model files were created.
    dump_files = list(Path(tmpdir).glob("nan_dump_*.npz"))
    model_files = list(Path(tmpdir).glob("model_*.mjb"))
    assert len(dump_files) == 1
    assert len(model_files) == 1

    # Verify model file can be loaded.
    dump = np.load(dump_files[0], allow_pickle=True)
    metadata = dump["_metadata"].item()
    model_path = Path(tmpdir) / metadata["model_file"]
    assert model_path.exists()

    # Load model and verify it's valid.
    loaded_model = mujoco.MjModel.from_binary_path(str(model_path))
    assert loaded_model.nq == simple_model.nq
    assert loaded_model.nv == simple_model.nv

    sim.close()


def test_nan_guard_with_complex_model():
  """NaN guard should work with complex robot model."""
  from mjlab.scene import Scene
  from mjlab.tasks.velocity.config.go1.rough_env_cfg import UnitreeGo1RoughEnvCfg

  env_cfg = UnitreeGo1RoughEnvCfg()
  scene = Scene(env_cfg.scene, device=get_test_device())
  model = scene.compile()

  with tempfile.TemporaryDirectory() as tmpdir:
    cfg = SimulationCfg(
      nan_guard=NanGuardCfg(enabled=True, buffer_size=3, output_dir=tmpdir)
    )
    sim = Simulation(num_envs=2, cfg=cfg, model=model, device=get_test_device())

    # Run a few steps.
    for _ in range(2):
      sim.step()

    # Inject NaN and trigger dump.
    sim.data.qpos[0, 0] = float("nan")
    sim.step()

    # Verify dump and model files were created.
    dump_files = list(Path(tmpdir).glob("nan_dump_*.npz"))
    model_files = list(Path(tmpdir).glob("model_*.mjb"))
    assert len(dump_files) == 1
    assert len(model_files) == 1

    # Load the saved model and verify it matches.
    dump = np.load(dump_files[0], allow_pickle=True)
    metadata = dump["_metadata"].item()
    model_path = Path(tmpdir) / metadata["model_file"]

    loaded_model = mujoco.MjModel.from_binary_path(str(model_path))
    assert loaded_model.nq == model.nq
    assert loaded_model.nv == model.nv
    assert loaded_model.nu == model.nu  # Check actuators too

    # Verify we can create MjData and restore a state.
    loaded_data = mujoco.MjData(loaded_model)
    state = dump["states_step_000000"][0]
    mujoco.mj_setState(
      loaded_model, loaded_data, state, mujoco.mjtState.mjSTATE_PHYSICS
    )
    mujoco.mj_forward(loaded_model, loaded_data)

    # Data should be valid (no NaN in derived quantities after forward).
    assert not np.isnan(loaded_data.qpos).any()

    sim.close()


def test_nan_guard_only_dumps_once(simple_model):
  """NaN guard should only dump once per training run."""
  with tempfile.TemporaryDirectory() as tmpdir:
    cfg = SimulationCfg(
      nan_guard=NanGuardCfg(enabled=True, buffer_size=5, output_dir=tmpdir)
    )
    sim = Simulation(num_envs=2, cfg=cfg, model=simple_model, device=get_test_device())

    # Inject NaN.
    sim.data.qpos[0, 0] = float("nan")
    sim.step()

    # Should have exactly one dump.
    dump_files = list(Path(tmpdir).glob("nan_dump_*.npz"))
    assert len(dump_files) == 1

    # Inject another NaN.
    sim.data.qpos[1, 0] = float("nan")
    sim.step()

    # Should still have only one dump.
    dump_files = list(Path(tmpdir).glob("nan_dump_*.npz"))
    assert len(dump_files) == 1

    sim.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_nan_guard_respects_buffer_size(simple_model):
  """NaN guard should only keep last K states in buffer."""
  with tempfile.TemporaryDirectory() as tmpdir:
    cfg = SimulationCfg(
      nan_guard=NanGuardCfg(enabled=True, buffer_size=3, output_dir=tmpdir)
    )
    sim = Simulation(num_envs=2, cfg=cfg, model=simple_model, device=get_test_device())

    # Run 10 steps.
    for _ in range(10):
      sim.step()

    # Inject NaN.
    sim.data.qpos[0, 0] = float("nan")
    sim.step()

    # Load dump.
    dump_files = list(Path(tmpdir).glob("nan_dump_*.npz"))
    dump = np.load(dump_files[0], allow_pickle=True)

    # Should only have 3 states (buffer size).
    state_keys = [k for k in dump.keys() if k.startswith("states_step_")]
    assert len(state_keys) == 3

    # Should be the last 3 steps (steps 8, 9, 10 where step 10 has NaN).
    assert "states_step_000008" in dump
    assert "states_step_000009" in dump
    assert "states_step_000010" in dump

    sim.close()
