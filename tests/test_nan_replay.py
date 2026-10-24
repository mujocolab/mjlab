"""Replay diagnostics against native MuJoCo, including model/input history."""

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import mujoco
import numpy as np
import pytest
import torch
import warp as wp
from filelock import Timeout
from PIL import Image
from test_variants import (
  _build_scene_with_variants,
  _simple_cone_spec,
  _simple_sphere_spec,
)

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scripts import nan_viz
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.utils import nan_guard
from mjlab.utils.nan_guard import NanGuardCfg
from mjlab.utils.nan_replay import NanDump

XML = """<mujoco><size nuserdata="2"/><worldbody>
  <body name="free"><freejoint/><geom size="0.1" mass="1" contype="0" conaffinity="0"/>
    <body><joint name="slider" type="slide"/><geom size="0.05" mass="0.1" contype="0" conaffinity="0"/></body>
  </body>
  <body mocap="true"><geom size="0.02" contype="0" conaffinity="0"/></body>
</worldbody><equality><weld body1="free" active="false"/></equality>
<actuator><general joint="slider" dyntype="filter" dynprm="0.1"/></actuator>
</mujoco>"""


@pytest.fixture
def simulation(tmp_path):
  sim = Simulation(
    2,
    SimulationCfg(
      mujoco=MujocoCfg(gravity=(0, 0, 0)),
      nan_guard=NanGuardCfg(enabled=True, output_dir=str(tmp_path), buffer_size=3),
    ),
    model=mujoco.MjModel.from_xml_string(XML),
    device="cpu",
  )
  fields = tuple(getattr(dr.body_mass, "model_fields"))  # noqa: B009
  sim.expand_model_fields(fields + ("geom_size",))
  return sim


def _trigger(sim):
  sim.data.qacc[:, 0] = float("nan")
  assert sim.nan_guard.check_and_dump(sim.data)
  return Path(sim.nan_guard.output_dir) / "nan_dump_latest.npz"


def test_native_integration_state_fields_cover_the_official_api():
  model = mujoco.MjModel.from_xml_string(XML)
  data = mujoco.MjData(model)
  spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
  assert sum(np.size(getattr(data, name)) for name in nan_guard._STATE_FIELDS) == (
    mujoco.mj_stateSize(model, spec)
  )
  before = np.empty(mujoco.mj_stateSize(model, spec))
  after = np.empty_like(before)
  mujoco.mj_getState(model, data, before, spec)
  data.tree_asleep[:] = 0
  mujoco.mj_getState(model, data, after, spec)
  # Sleeping-tree counters are outside MuJoCo's integration-state API.
  np.testing.assert_array_equal(before, after)


def test_randomized_model_without_equalities_skips_empty_expanded_eq_data(tmp_path):
  model = mujoco.MjModel.from_xml_string(
    XML.replace('<equality><weld body1="free" active="false"/></equality>', "")
  )
  sim = Simulation(
    2,
    SimulationCfg(nan_guard=NanGuardCfg(enabled=True, output_dir=str(tmp_path))),
    model=model,
    device="cpu",
  )
  sim.expand_model_fields(tuple(getattr(dr.body_mass, "model_fields")))  # noqa: B009
  sim.model.body_mass[1, 1] = 3
  sim.recompute_constants(RecomputeLevel.set_const)
  sim.nan_guard.capture(sim.data)
  assert "eq_data" not in sim.nan_guard.buffer[0]["model_fields"]
  restored, _ = NanDump(_trigger(sim)).restore(0, 1)
  assert restored.body_mass[1] == 3
  assert restored.eq_data.size == 0


def test_replay_keeps_each_frames_world_model_and_full_integration_inputs(simulation):
  sim = simulation
  expected = []
  for frame in range(2):
    sim.model.body_mass[:, 1] = torch.tensor([1.0 + frame, 3.0 + frame])
    sim.model.geom_size[:, 0] = 0.1 + 0.05 * frame
    sim.recompute_constants(RecomputeLevel.set_const)
    sim.model.opt.gravity[0] = torch.tensor([0.0, 0.0, -float(frame)])
    sim.data.time[:] = 0.25 + frame
    sim.data.qvel[:] = frame + 0.2
    sim.data.act[:] = frame + 0.3
    sim.data.ctrl[:] = frame + 0.4
    sim.data.qfrc_applied[:] = frame + 0.5
    sim.data.xfrc_applied[:] = frame + 0.6
    sim.data.qacc_warmstart[:] = frame + 0.7
    sim.data.mocap_pos[:] = frame + 0.8
    sim.data.mocap_quat[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    sim.data.userdata[:] = frame + 0.9
    sim.data.eq_active[:] = bool(frame)
    sim.nan_guard.capture(sim.data)
    expected.append(
      {
        name: getattr(sim.data, name)[:].numpy().copy()
        for name in (
          "time",
          "qpos",
          "qvel",
          "act",
          "ctrl",
          "qfrc_applied",
          "xfrc_applied",
          "qacc_warmstart",
          "mocap_pos",
          "mocap_quat",
          "userdata",
          "eq_active",
        )
      }
    )
  # A later reset/DR change cannot rewrite the history awaiting detection.
  sim.model.body_mass[:, 1] = 99
  sim.data.ctrl[:] = 99
  archive = NanDump(_trigger(sim))
  for frame in range(2):
    for world in range(2):
      model, data = archive.restore(frame, world)
      assert model.body_mass[1] == [1 + frame, 3 + frame][world]
      np.testing.assert_allclose(model.geom_size[0], 0.1 + 0.05 * frame)
      np.testing.assert_allclose(model.opt.gravity, [0, 0, -frame])
      for name, values in expected[frame].items():
        np.testing.assert_array_equal(getattr(data, name), values[world])
      model, data = archive.restore(frame, world, forward=True)
      np.testing.assert_array_equal(
        data.qacc_warmstart, expected[frame]["qacc_warmstart"][world]
      )
  assert len(set(sum(archive.metadata["model_files"].values(), []))) == 4
  # The simulator's public host template was not modified by materialization.
  assert sim.mj_model.body_mass[1] == 1


def test_replayed_native_step_uses_randomized_mass_and_recorded_applied_force(
  simulation,
):
  sim = simulation
  sim.model.body_mass[:, 1] = torch.tensor([1.0, 3.0])
  sim.recompute_constants(RecomputeLevel.set_const)
  sim.data.qfrc_applied[:, 0] = 3
  sim.data.ctrl[:] = 0.25
  sim.nan_guard.capture(sim.data)
  model, data = NanDump(_trigger(sim)).restore(0, 1)
  reference = copy(sim.mj_model)
  reference.body_mass[:] = sim.model.body_mass[1].numpy()
  reference.body_inertia[:] = sim.model.body_inertia[1].numpy()
  mujoco.mj_setConst(reference, mujoco.MjData(reference))
  reference_data = mujoco.MjData(reference)
  reference_data.qfrc_applied[0] = 3
  reference_data.ctrl[:] = 0.25
  mujoco.mj_step(model, data)
  mujoco.mj_step(reference, reference_data)
  np.testing.assert_allclose(data.qvel, reference_data.qvel, atol=1e-8)
  np.testing.assert_allclose(data.qpos, reference_data.qpos, atol=1e-8)


def test_variant_replay_materializes_geometry_assignments_and_native_constants(
  tmp_path,
):
  spec, variants = _build_scene_with_variants(_simple_sphere_spec, _simple_cone_spec)
  sim = Simulation(
    2,
    SimulationCfg(nan_guard=NanGuardCfg(enabled=True, output_dir=str(tmp_path))),
    spec=spec,
    variant_info=variants,
    device="cpu",
  )
  assignments = sim.model.geom_dataid[:].numpy().copy()
  masses = sim.model.body_mass[:].numpy().copy()
  sim.nan_guard.capture(sim.data)
  archive = NanDump(_trigger(sim))
  for world in range(2):
    model, data = archive.restore(0, world, forward=True)
    np.testing.assert_array_equal(model.geom_dataid, assignments[world])
    np.testing.assert_allclose(model.body_mass, masses[world])
    reference = (_simple_sphere_spec, _simple_cone_spec)[world]().compile()
    # Native queries receive the actual captured float32 local transforms.
    for field in ("geom_pos", "geom_quat", "geom_size"):
      getattr(reference, field)[:] = getattr(sim.model, field)[world].numpy()
    reference_data = mujoco.MjData(reference)
    mujoco.mj_forward(reference, reference_data)
    # Avoid a ray exactly through a shared vertex: float32 model rounding can
    # put it on opposite sides of a triangle boundary in the two compiles.
    origin, direction = np.array([0.01, 0.02, 3.0]), np.array([0.0, 0.0, -1.0])
    geom_id = np.empty(1, dtype=np.int32)
    actual = mujoco.mj_ray(model, data, origin, direction, None, True, -1, geom_id)
    expected = mujoco.mj_ray(
      reference, reference_data, origin, direction, None, True, -1, geom_id
    )
    assert actual == pytest.approx(expected, abs=1e-6), (
      world,
      model.geom_rgba,
      model.geom_size,
      model.geom_rbound,
      data.qpos,
      data.geom_xpos,
      data.geom_xmat,
    )
  assert not np.array_equal(assignments[0], assignments[1])


def test_old_dump_without_model_revisions_remains_readable(tmp_path):
  model = mujoco.MjModel.from_xml_string(XML)
  model.body_mass[1] = 5
  data = mujoco.MjData(model)
  data.qpos[0] = 0.75
  spec = int(mujoco.mjtState.mjSTATE_PHYSICS)
  state = np.empty(mujoco.mj_stateSize(model, spec))
  mujoco.mj_getState(model, data, state, spec)
  mujoco.mj_saveModel(model, str(tmp_path / "old.mjb"), None)
  path = tmp_path / "old.npz"
  np.savez(
    path,
    states_step_000000=state[None],
    _metadata=np.array(
      {
        "model_file": "old.mjb",
        "state_spec": spec,
      },
      dtype=object,
    ),
  )
  restored_model, restored_data = NanDump(path).restore(0, 0, forward=True)
  assert restored_model.body_mass[1] == 5
  assert restored_data.qpos[0] == 0.75


def test_concurrent_guards_publish_distinct_complete_archives(simulation, monkeypatch):
  sim = simulation
  monkeypatch.setattr(
    nan_guard, "datetime", SimpleNamespace(now=lambda: datetime(2020, 1, 1))
  )
  second = nan_guard.NanGuard(
    NanGuardCfg(enabled=True, output_dir=str(sim.nan_guard.output_dir)),
    sim.num_envs,
    sim.mj_model,
  )
  sim.nan_guard.capture(sim.data)
  second.capture(sim.data)
  with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
      executor.submit(guard._dump_buffer, [0]) for guard in (sim.nan_guard, second)
    ]
    for future in futures:
      future.result()
  paths = [
    path
    for path in Path(sim.nan_guard.output_dir).glob("nan_dump_*.npz")
    if "latest" not in path.name
  ]
  assert len(paths) == 2
  for path in paths + [Path(sim.nan_guard.output_dir) / "nan_dump_latest.npz"]:
    assert NanDump(path).restore(0, 0)[0].nq == sim.mj_model.nq
  assert not list(Path(sim.nan_guard.output_dir).glob(".*.tmp"))


def test_failed_npz_write_preserves_previous_publication(simulation, monkeypatch):
  sim = simulation
  sim.nan_guard.capture(sim.data)
  sim.nan_guard._dump_buffer([0])
  latest = Path(sim.nan_guard.output_dir) / "nan_dump_latest.npz"
  original = latest.read_bytes()

  def fail(stream, **_data):
    stream.write(b"partial archive")
    raise OSError("write interrupted")

  monkeypatch.setattr(np, "savez_compressed", fail)
  with pytest.raises(OSError, match="write interrupted"):
    sim.nan_guard._dump_buffer([0])
  assert latest.read_bytes() == original
  assert NanDump(latest).restore(0, 0)[0].nq == sim.mj_model.nq
  assert not list(latest.parent.glob(".*.tmp"))


def test_viewer_rebuilds_scene_for_selected_model_revision(simulation, monkeypatch):
  sim = simulation
  sim.nan_guard.capture(sim.data)
  sim.model.body_mass[1, 1] = 3
  sim.nan_guard.capture(sim.data)
  path = _trigger(sim)
  server = MagicMock()
  monkeypatch.setattr(nan_viz.viser, "ViserServer", MagicMock(return_value=server))
  scene = MagicMock()
  monkeypatch.setattr(nan_viz, "ViserMujocoScene", scene)
  viewer = nan_viz.NanDumpViewer(path)
  viewer.setup()
  viewer.current_step, viewer.current_env = 1, 1
  viewer._update_state()
  assert viewer.model.body_mass[1] == 3
  assert scene.call_count == 2
  server.scene.reset.assert_called_once()


def test_model_corruption_is_reported_instead_of_replaying_wrong_constants(simulation):
  sim = simulation
  sim.nan_guard.capture(sim.data)
  archive = NanDump(_trigger(sim))
  model_path = archive.path.parent / archive.metadata["model_file"]
  with model_path.open("ab") as stream:
    stream.write(b"corrupt")
  with pytest.raises(ValueError, match="checksum mismatch"):
    archive.restore(0, 0)


def test_delayed_actuator_history_is_restored(tmp_path):
  model = mujoco.MjModel.from_xml_string(
    XML.replace('dynprm="0.1"', 'dynprm="0.1" delay="0.01" nsample="3"')
  )
  sim = Simulation(
    2,
    SimulationCfg(nan_guard=NanGuardCfg(enabled=True, output_dir=str(tmp_path))),
    model=model,
    device="cpu",
  )
  assert model.nhistory > 0
  values = torch.arange(model.nhistory, dtype=torch.float32)
  sim.data.history[:] = values
  sim.nan_guard.capture(sim.data)
  _, data = NanDump(_trigger(sim)).restore(0, 1)
  np.testing.assert_array_equal(data.history, values.numpy())


def test_alias_lock_timeout_keeps_complete_unique_archive(
  simulation, monkeypatch, capsys
):
  sim = simulation
  sim.nan_guard.capture(sim.data)
  lock = MagicMock()
  lock.__enter__.side_effect = Timeout("publication lock")
  monkeypatch.setattr(nan_guard, "FileLock", MagicMock(return_value=lock))
  with pytest.raises(Timeout):
    _trigger(sim)
  files = list(Path(sim.nan_guard.output_dir).glob("nan_dump_*.npz"))
  assert len(files) == 1
  assert NanDump(files[0]).restore(0, 1)[0].nq == sim.mj_model.nq
  assert not sim.nan_guard._dumped
  assert str(files[0]) in capsys.readouterr().out


def test_texture_assets_remain_embedded_after_source_file_is_removed(
  tmp_path, monkeypatch
):
  if not hasattr(mujoco, "mj_embedAssets"):
    pytest.skip("By-reference textures were introduced in MuJoCo 3.17")
  texture = tmp_path / "texture.png"
  Image.new("RGB", (2, 2), (255, 0, 0)).save(texture)
  xml = f'''<mujoco><asset>
    <texture name="t" type="2d" file="{texture.as_posix()}" embed="false"/>
    <material name="m" texture="t"/></asset><worldbody>
    <body><freejoint/><geom type="box" size=".1 .1 .1" material="m"/></body>
  </worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  assert model.tex_adr[0] == -1
  sim = Simulation(
    2,
    SimulationCfg(nan_guard=NanGuardCfg(enabled=True, output_dir=str(tmp_path))),
    model=model,
    device="cpu",
  )
  texture.unlink()
  sim.nan_guard.capture(sim.data)
  restored, _ = NanDump(_trigger(sim)).restore(0, 0)
  assert restored.tex_adr[0] >= 0
  assert restored.ntexdata > 0
  np.testing.assert_array_equal(restored.tex_data.reshape(-1, 3), [[255, 0, 0]] * 4)
  monkeypatch.delattr(mujoco, "mj_embedAssets")
  with pytest.raises(ValueError, match="by-reference textures"):
    nan_guard.NanGuard(NanGuardCfg(enabled=True), 2, model)


def test_embedded_models_work_without_new_embed_api(simulation, monkeypatch):
  sim = simulation
  monkeypatch.delattr(mujoco, "mj_embedAssets", raising=False)
  guard = nan_guard.NanGuard(NanGuardCfg(enabled=True), 2, sim.mj_model)
  assert guard.mj_model is not sim.mj_model
  np.testing.assert_array_equal(guard.mj_model.body_mass, sim.mj_model.body_mass)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA streams")
def test_capture_on_custom_warp_stream_keeps_pre_mutation_inputs(tmp_path):
  wp.init()
  stream = wp.Stream("cuda:0")
  with wp.ScopedStream(stream):
    sim = Simulation(
      2,
      SimulationCfg(nan_guard=NanGuardCfg(enabled=True, output_dir=str(tmp_path))),
      model=mujoco.MjModel.from_xml_string(XML),
      device="cuda:0",
    )
    sim.data.ctrl[:] = 0.25
    sim.data.qfrc_applied[:] = 0.5
    sim.nan_guard.capture(sim.data)
    sim.data.ctrl[:] = 99
    sim.data.qfrc_applied[:] = 99
    archive = NanDump(_trigger(sim))
    _, data = archive.restore(0, 1)
    np.testing.assert_allclose(data.ctrl, 0.25)
    np.testing.assert_allclose(data.qfrc_applied, 0.5)
