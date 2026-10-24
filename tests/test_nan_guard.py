"""Tests for NaN guard functionality."""

import errno
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.nan_guard import NanGuardCfg


def _load_dump(path: Path) -> dict[str, np.ndarray]:
  """Close NPZ handles before Windows temporary-directory cleanup."""
  with np.load(path, allow_pickle=True) as dump:
    return {key: dump[key] for key in dump.files}


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


@pytest.fixture
def mocap_model():
  """Create a model with a mocap body and a freejoint body."""
  xml = """
  <mujoco>
    <worldbody>
      <body name="mocap_body" mocap="true">
        <geom type="sphere" size="0.05"/>
      </body>
      <body name="free_body">
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_nan_guard_captures_and_dumps_on_nan(simple_model):
  """NaN guard should capture states and dump when NaN detected."""
  with tempfile.TemporaryDirectory() as tmpdir:
    cfg = SimulationCfg(
      nan_guard=NanGuardCfg(
        enabled=True,
        buffer_size=5,
        output_dir=tmpdir,
        max_envs_to_dump=2,
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

    # Check that timestamped dump file was created.
    dump_files = [
      f for f in Path(tmpdir).glob("nan_dump_*.npz") if "latest" not in f.name
    ]
    assert len(dump_files) == 1

    # Load and inspect the dump.
    dump = _load_dump(dump_files[0])
    metadata = dump["_metadata"].item()

    assert metadata["num_envs_total"] == 4
    assert metadata["num_envs_dumped"] == 1
    assert 1 in metadata["nan_env_ids"]
    assert metadata["buffer_size"] == 4

    # Check that states were captured.
    assert "states_step_000000" in dump
    assert "states_step_000001" in dump
    assert "states_step_000002" in dump
    assert "states_step_000003" in dump

    # Verify state shape: (num_envs_dumped, state_size).
    state = dump["states_step_000000"]
    assert state.shape[0] == 1


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
    dump_files = [
      f for f in Path(tmpdir).glob("nan_dump_*.npz") if "latest" not in f.name
    ]
    assert len(dump_files) == 1

    dump = _load_dump(dump_files[0])
    metadata = dump["_metadata"].item()

    # Should detect exactly the environments with NaN/Inf.
    nan_env_ids = set(metadata["nan_env_ids"])
    assert nan_env_ids == {2, 5, 7}, f"Expected {{2, 5, 7}}, got {nan_env_ids}"


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
    dump_files = [
      f for f in Path(tmpdir).glob("nan_dump_*.npz") if "latest" not in f.name
    ]
    model_files = [
      f for f in Path(tmpdir).glob("model_*.mjb") if "latest" not in f.name
    ]
    assert len(dump_files) == 1
    assert len(model_files) == 1

    # Verify model file can be loaded.
    dump = _load_dump(dump_files[0])
    metadata = dump["_metadata"].item()
    model_path = Path(tmpdir) / metadata["model_file"]
    assert model_path.exists()

    # Load model and verify it's valid.
    loaded_model = mujoco.MjModel.from_binary_path(str(model_path))
    assert loaded_model.nq == simple_model.nq
    assert loaded_model.nv == simple_model.nv


@pytest.mark.slow
def test_nan_guard_with_complex_model():
  """NaN guard should work with complex robot model."""
  from mjlab.scene import Scene
  from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_rough_env_cfg

  scene = Scene(unitree_go1_rough_env_cfg().scene, device=get_test_device())
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
    dump_files = [
      f for f in Path(tmpdir).glob("nan_dump_*.npz") if "latest" not in f.name
    ]
    model_files = [
      f for f in Path(tmpdir).glob("model_*.mjb") if "latest" not in f.name
    ]
    assert len(dump_files) == 1
    assert len(model_files) == 1

    # Load the saved model and verify it matches.
    dump = _load_dump(dump_files[0])
    metadata = dump["_metadata"].item()
    model_path = Path(tmpdir) / metadata["model_file"]

    loaded_model = mujoco.MjModel.from_binary_path(str(model_path))
    assert loaded_model.nq == model.nq
    assert loaded_model.nv == model.nv
    assert loaded_model.nu == model.nu  # Check actuators too

    # Verify we can create MjData and restore a state.
    loaded_data = mujoco.MjData(loaded_model)
    state = dump["states_step_000000"][0]
    state_spec = metadata.get("state_spec", mujoco.mjtState.mjSTATE_PHYSICS.value)
    mujoco.mj_setState(loaded_model, loaded_data, state, state_spec)
    mujoco.mj_forward(loaded_model, loaded_data)

    # Data should be valid (no NaN in derived quantities after forward).
    assert not np.isnan(loaded_data.qpos).any()


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

    # Should have exactly one timestamped dump.
    dump_files = [
      f for f in Path(tmpdir).glob("nan_dump_*.npz") if "latest" not in f.name
    ]
    assert len(dump_files) == 1

    # Inject another NaN.
    sim.data.qpos[1, 0] = float("nan")
    sim.step()

    # Should still have only one timestamped dump.
    dump_files = [
      f for f in Path(tmpdir).glob("nan_dump_*.npz") if "latest" not in f.name
    ]
    assert len(dump_files) == 1


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
    dump_files = [
      f for f in Path(tmpdir).glob("nan_dump_*.npz") if "latest" not in f.name
    ]
    dump = _load_dump(dump_files[0])

    # Should only have 3 states (buffer size).
    state_keys = [k for k in dump.keys() if k.startswith("states_step_")]
    assert len(state_keys) == 3

    # Should be the last 3 steps (steps 8, 9, 10 where step 10 has NaN).
    assert "states_step_000008" in dump
    assert "states_step_000009" in dump
    assert "states_step_000010" in dump


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_nan_guard_captures_high_indexed_envs(simple_model):
  """NaN guard should capture NaN in high-indexed environments beyond max_envs_to_dump."""
  with tempfile.TemporaryDirectory() as tmpdir:
    cfg = SimulationCfg(
      nan_guard=NanGuardCfg(
        enabled=True, buffer_size=5, output_dir=tmpdir, max_envs_to_dump=3
      )
    )
    sim = Simulation(num_envs=10, cfg=cfg, model=simple_model, device=get_test_device())

    for _ in range(3):
      sim.step()

    sim.data.qpos[7, 0] = float("nan")
    sim.step()

    dump_files = [
      f for f in Path(tmpdir).glob("nan_dump_*.npz") if "latest" not in f.name
    ]
    assert len(dump_files) == 1

    dump = _load_dump(dump_files[0])
    metadata = dump["_metadata"].item()

    assert 7 in metadata["nan_env_ids"]
    assert 7 in metadata["dumped_env_ids"]
    assert metadata["num_envs_total"] == 10
    assert metadata["num_envs_dumped"] == 1

    state = dump["states_step_000000"]
    assert state.shape[0] == 1


def test_nan_guard_creates_retrievable_latest_artifacts(simple_model):
  """Latest artifacts remain usable, with symlinks on capable filesystems."""
  with tempfile.TemporaryDirectory() as tmpdir:
    probe_source = Path(tmpdir) / "probe_source"
    probe_source.touch()
    probe_link = Path(tmpdir) / "probe_link"
    try:
      probe_link.symlink_to(probe_source.name)
      supports_symlinks = True
    except (NotImplementedError, OSError):
      supports_symlinks = False

    cfg = SimulationCfg(
      nan_guard=NanGuardCfg(enabled=True, buffer_size=5, output_dir=tmpdir)
    )
    sim = Simulation(num_envs=2, cfg=cfg, model=simple_model, device=get_test_device())

    sim.data.qpos[0, 0] = float("nan")
    sim.step()

    latest_dump = Path(tmpdir) / "nan_dump_latest.npz"
    latest_model = Path(tmpdir) / "model_latest.mjb"

    # Verify both retrieval paths, and retain the native symlink behavior where
    # supported rather than silently replacing it everywhere with copies.
    assert latest_dump.exists()
    assert latest_model.exists()
    assert latest_dump.is_symlink() == supports_symlinks
    assert latest_model.is_symlink() == supports_symlinks

    # Load via symlink and verify it works.
    dump = _load_dump(latest_dump)
    metadata = dump["_metadata"].item()

    # Metadata should reference the timestamped model file.
    assert metadata["model_file"].startswith("model_")
    assert metadata["model_file"].endswith(".mjb")
    assert (Path(tmpdir) / metadata["model_file"]).exists()

    # Loading via symlink should work.
    loaded_model = mujoco.MjModel.from_binary_path(str(latest_model))
    assert loaded_model.nq == simple_model.nq
    assert (
      latest_model.read_bytes() == (Path(tmpdir) / metadata["model_file"]).read_bytes()
    )


@pytest.mark.parametrize("reason", ["privilege", "filesystem", "unimplemented"])
def test_nan_guard_copies_latest_when_symlinks_are_unavailable(
  simple_model, tmp_path, monkeypatch, reason
):
  error: OSError | NotImplementedError
  if reason == "privilege":
    error = OSError("symlink privilege unavailable")
    monkeypatch.setattr(error, "winerror", 1314, raising=False)
  elif reason == "filesystem":
    error = OSError(errno.ENOTSUP, "symlinks unsupported")
  else:
    error = NotImplementedError("symlinks unavailable")
  attempts = []

  def deny_symlink(path, target, target_is_directory=False):
    attempts.append(path)
    raise error

  monkeypatch.setattr(Path, "symlink_to", deny_symlink)
  cfg = SimulationCfg(nan_guard=NanGuardCfg(enabled=True, output_dir=str(tmp_path)))
  sim = Simulation(num_envs=2, cfg=cfg, model=simple_model, device=get_test_device())
  sim.nan_guard.capture(sim.data)
  sim.data.qpos[0, 0] = float("nan")
  assert sim.nan_guard.check_and_dump(sim.data)
  assert sim.nan_guard._dumped
  assert not sim.nan_guard.check_and_dump(sim.data)

  latest_dump = tmp_path / "nan_dump_latest.npz"
  latest_model = tmp_path / "model_latest.mjb"
  # Latest aliases are staged privately, then atomically replace the old path.
  assert len(attempts) == 2
  assert attempts[0].name.startswith(".model_latest.mjb.")
  assert attempts[1].name.startswith(".nan_dump_latest.npz.")
  assert all(not path.exists() for path in attempts)
  assert not latest_dump.is_symlink() and not latest_model.is_symlink()
  with np.load(latest_dump, allow_pickle=True) as dump:
    metadata = dump["_metadata"].item()
    state = dump["states_step_000000"][0]
  assert metadata["dumped_env_ids"] == [0]
  timestamped_model = tmp_path / metadata["model_file"]
  timestamped_dump = tmp_path / f"nan_dump_{metadata['timestamp']}.npz"
  assert latest_model.read_bytes() == timestamped_model.read_bytes()
  assert latest_dump.read_bytes() == timestamped_dump.read_bytes()
  loaded_model = mujoco.MjModel.from_binary_path(str(latest_model))
  data = mujoco.MjData(loaded_model)
  mujoco.mj_setState(loaded_model, data, state, metadata["state_spec"])
  assert np.isfinite(data.qpos).all()


@pytest.mark.parametrize("error_number", [errno.EIO, errno.EACCES])
def test_nan_guard_propagates_unrelated_latest_artifact_errors(
  simple_model, tmp_path, monkeypatch, error_number
):
  def fail_symlink(*args, **kwargs):
    raise OSError(error_number, "unrelated filesystem failure")

  monkeypatch.setattr(Path, "symlink_to", fail_symlink)
  cfg = SimulationCfg(nan_guard=NanGuardCfg(enabled=True, output_dir=str(tmp_path)))
  sim = Simulation(num_envs=2, cfg=cfg, model=simple_model, device=get_test_device())
  sim.nan_guard.capture(sim.data)
  sim.data.qpos[0, 0] = float("nan")
  with pytest.raises(OSError, match="unrelated filesystem failure") as caught:
    sim.nan_guard.check_and_dump(sim.data)
  assert caught.value.errno == error_number
  assert not sim.nan_guard._dumped


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_nan_guard_captures_mocap_state(mocap_model):
  """NaN guard should capture and restore mocap body poses."""
  with tempfile.TemporaryDirectory() as tmpdir:
    cfg = SimulationCfg(
      nan_guard=NanGuardCfg(enabled=True, buffer_size=5, output_dir=tmpdir)
    )
    sim = Simulation(num_envs=2, cfg=cfg, model=mocap_model, device=get_test_device())

    # Set a known mocap position before stepping.
    sim.data.mocap_pos[0, 0] = torch.tensor([1.0, 2.0, 3.0], device=get_test_device())

    # Run a few steps.
    for _ in range(3):
      sim.step()

    # Inject NaN to trigger dump.
    sim.data.qpos[0, 0] = float("nan")
    sim.step()

    # Load dump.
    dump_files = [
      f for f in Path(tmpdir).glob("nan_dump_*.npz") if "latest" not in f.name
    ]
    assert len(dump_files) == 1

    dump = _load_dump(dump_files[0])
    metadata = dump["_metadata"].item()

    # Metadata should include state_spec with mocap flags.
    assert "state_spec" in metadata
    state_spec = metadata["state_spec"]
    assert state_spec & mujoco.mjtState.mjSTATE_MOCAP_POS.value
    assert state_spec & mujoco.mjtState.mjSTATE_MOCAP_QUAT.value

    # State size should be larger than physics-only.
    physics_size = mujoco.mj_stateSize(mocap_model, mujoco.mjtState.mjSTATE_PHYSICS)
    assert metadata["state_size"] > physics_size

    # Restore a state and verify mocap data round-trips.
    loaded_data = mujoco.MjData(mocap_model)
    state = dump["states_step_000000"][0]
    mujoco.mj_setState(mocap_model, loaded_data, state, state_spec)
    assert loaded_data.mocap_pos.shape == (1, 3)
    np.testing.assert_allclose(loaded_data.mocap_pos[0], [1.0, 2.0, 3.0], atol=1e-5)
