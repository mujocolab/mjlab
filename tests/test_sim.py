"""Tests for sim.py."""

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import pytest
import torch
import warp as wp
from conftest import get_test_device

from mjlab.sim import MujocoCfg, Simulation, SimulationCfg


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def robot_xml():
  """Simple robot with geoms and joints."""
  return """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="1.0"
            friction="0.5 0.01 0.005"/>
          <body name="foot1" pos="0.2 0 0">
            <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="foot1_geom" type="box" size="0.05 0.05 0.05" mass="0.1"
              friction="0.5 0.01 0.005"/>
          </body>
          <body name="foot2" pos="-0.2 0 0">
            <joint name="joint2" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="foot2_geom" type="box" size="0.05 0.05 0.05" mass="0.1"
              friction="0.5 0.01 0.005"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """


def test_simulation_config_is_piped(robot_xml, device):
  """Test that SimulationCfg values are applied to both mj_model and wp_model."""
  model = mujoco.MjModel.from_xml_string(robot_xml)

  cfg = SimulationCfg(
    contact_sensor_maxmatch=128,
    broadphase="sap_tile",
    broadphase_filter=("plane", "aabb"),
    mujoco=MujocoCfg(
      timestep=0.02,
      integrator="euler",
      solver="cg",
      iterations=7,
      ls_iterations=14,
      ccd_iterations=20,
      gravity=(0, 0, 7.5),
      enableflags=("energy",),
    ),
  )

  sim = Simulation(num_envs=1, cfg=cfg, model=model, device=device)

  # MujocoCfg should be applied to mj_model.
  assert sim.mj_model.opt.timestep == cfg.mujoco.timestep
  assert sim.mj_model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
  assert sim.mj_model.opt.solver == mujoco.mjtSolver.mjSOL_CG
  assert sim.mj_model.opt.iterations == cfg.mujoco.iterations
  assert sim.mj_model.opt.ls_iterations == cfg.mujoco.ls_iterations
  assert sim.mj_model.opt.ccd_iterations == cfg.mujoco.ccd_iterations
  assert tuple(sim.mj_model.opt.gravity) == cfg.mujoco.gravity
  assert sim.mj_model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_ENERGY

  # MujocoCfg should be inherited by wp_model via put_model.
  np.testing.assert_almost_equal(
    sim.model.opt.timestep[0].cpu().numpy(), cfg.mujoco.timestep
  )
  np.testing.assert_almost_equal(
    sim.model.opt.gravity[0].cpu().numpy(), cfg.mujoco.gravity
  )
  assert sim.model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
  assert sim.model.opt.solver == mujoco.mjtSolver.mjSOL_CG
  assert sim.model.opt.iterations == cfg.mujoco.iterations
  assert sim.model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_ENERGY

  # SimulationCfg's warp-only settings should be applied to wp_model.opt.
  assert sim.wp_model.opt.contact_sensor_maxmatch == cfg.contact_sensor_maxmatch
  assert sim.wp_model.opt.broadphase == mjwarp.BroadphaseType.SAP_TILE
  assert sim.wp_model.opt.broadphase_filter == (
    mjwarp.BroadphaseFilter.PLANE | mjwarp.BroadphaseFilter.AABB
  )


def test_default_broadphase_keeps_put_model_heuristic(robot_xml, device):
  """Unset broadphase settings should not override put_model's own heuristic."""
  model = mujoco.MjModel.from_xml_string(robot_xml)
  heuristic_opt = mjwarp.put_model(model).opt

  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)

  assert sim.wp_model.opt.broadphase == heuristic_opt.broadphase
  assert sim.wp_model.opt.broadphase_filter == heuristic_opt.broadphase_filter


def test_ls_parallel_is_deprecated():
  """Setting the removed ls_parallel option warns instead of erroring."""
  with pytest.warns(DeprecationWarning, match="ls_parallel"):
    SimulationCfg(ls_parallel=True)


def test_sim_reset_restores_initial_state(robot_xml, device):
  """Test that sim.reset() restores qpos/qvel to initial values."""
  model = mujoco.MjModel.from_xml_string(robot_xml)
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)

  qpos0 = sim.data.qpos.clone()
  qvel0 = sim.data.qvel.clone()

  # Run simulation to modify state.
  for _ in range(10):
    sim.step()

  assert not torch.allclose(sim.data.qpos, qpos0)
  assert not torch.allclose(sim.data.qvel, qvel0)

  # Reset should restore initial state.
  sim.reset()

  torch.testing.assert_close(sim.data.qpos[:], qpos0)
  torch.testing.assert_close(sim.data.qvel[:], qvel0)
  # qacc_warmstart should be zeroed.
  assert (sim.data.qacc_warmstart == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_sim_reset_selective(robot_xml, device):
  """Test that sim.reset() only affects specified environments."""
  model = mujoco.MjModel.from_xml_string(robot_xml)
  sim = Simulation(num_envs=4, cfg=SimulationCfg(), model=model, device=device)

  qpos0 = sim.data.qpos.clone()

  # Run simulation to modify state.
  for _ in range(10):
    sim.step()

  qpos_after_sim = sim.data.qpos.clone()

  # Reset only env 1 and 3.
  sim.reset(torch.tensor([1, 3], device=device))

  # Envs 1 and 3 should be reset.
  torch.testing.assert_close(sim.data.qpos[1], qpos0[1])
  torch.testing.assert_close(sim.data.qpos[3], qpos0[3])
  # Envs 0 and 2 should be unchanged.
  torch.testing.assert_close(sim.data.qpos[0], qpos_after_sim[0])
  torch.testing.assert_close(sim.data.qpos[2], qpos_after_sim[2])


def test_xpos_matches_qpos_after_forward(robot_xml, device):
  """sim.step() leaves xpos stale; sim.forward() makes it match qpos.

  In MuJoCo, mj_step = mj_step1 (forward kinematics + forces) + mj_step2
  (integration). After mj_step, qpos/qvel are post-integration but xpos is
  from the pre-integration forward pass. sim.forward() recomputes xpos from
  the current qpos.
  """
  model = mujoco.MjModel.from_xml_string(robot_xml)
  cfg = SimulationCfg(mujoco=MujocoCfg(timestep=0.01))  # Large dt for clear signal
  sim = Simulation(num_envs=2, cfg=cfg, model=model, device=device)

  # Step enough for significant velocity -> large staleness gap.
  for _ in range(50):
    sim.step()

  # xpos is stale: reflects pre-integration state of last step.
  # For the freejoint body (body 1), qpos[:3] is the true position.
  xpos_stale = sim.data.xpos[:, 1].clone()
  qpos_pos = sim.data.qpos[:, :3].clone()
  assert not torch.allclose(xpos_stale, qpos_pos, atol=1e-4)

  # forward() refreshes derived quantities from current qpos.
  sim.forward()
  xpos_fresh = sim.data.xpos[:, 1].clone()
  torch.testing.assert_close(xpos_fresh, qpos_pos, atol=1e-5, rtol=0)


def test_expand_model_fields_skips_unchanged_graph_rebuild(robot_xml, device):
  """Repeated expand_model_fields with already-expanded fields preserves graphs."""
  model = mujoco.MjModel.from_xml_string(robot_xml)
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  sim.expand_model_fields(("body_mass", "body_subtreemass"))
  step_graph_before = sim.step_graph

  # Re-requesting already expanded fields is a no-op and keeps graphs intact.
  sim.expand_model_fields(("body_mass",))
  assert sim.step_graph is step_graph_before

  # Adding a new field replaces storage and invalidates cached recompute graphs.
  sim._recompute_graphs[int(mjwarp.SetConstGroup.FIXED)] = object()
  sim.expand_model_fields(("geom_friction",))
  assert sim._recompute_graphs == {}
  if sim.use_cuda_graph:
    assert sim.step_graph is not step_graph_before


def test_recompute_constants_none_is_noop_and_preserves_state(
  robot_xml, device, monkeypatch
):
  """SetConstGroup.NONE is a no-op; active groups update constants and keep qpos."""
  model = mujoco.MjModel.from_xml_string(robot_xml)
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  sim.expand_model_fields(
    ("body_mass", "body_subtreemass", "body_invweight0", "dof_invweight0")
  )

  called = False
  orig_set_const = mjwarp.set_const

  def spy_set_const(*args, **kwargs):
    nonlocal called
    called = True
    return orig_set_const(*args, **kwargs)

  monkeypatch.setattr(mjwarp, "set_const", spy_set_const)
  sim.recompute_constants(mjwarp.SetConstGroup.NONE)
  assert not called

  sim.step()
  qpos_before = sim.data.qpos.clone()
  sim.model.body_mass[:, 1] *= torch.tensor([1.5, 2.5], device=device)
  sim.recompute_constants(mjwarp.SetConstGroup.ALL)
  assert called
  torch.testing.assert_close(sim.data.qpos[:], qpos_before)
  assert sim.model.body_subtreemass[1, 1] > sim.model.body_subtreemass[0, 1]


@pytest.mark.skipif(not wp.is_cuda_available(), reason="Warp CUDA device required")
def test_cuda_set_const_graph_replay_matches_eager(robot_xml):
  """CUDA graph capture and replay of mjwarp.set_const matches eager execution."""
  model = mujoco.MjModel.from_xml_string(robot_xml)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  with wp.ScopedDevice("cuda:0"):
    wp_m_graph = mjwarp.put_model(model)
    wp_d_graph = mjwarp.put_data(model, data, nworld=2)
    wp_m_eager = mjwarp.put_model(model)
    wp_d_eager = mjwarp.put_data(model, data, nworld=2)

    for wm in (wp_m_graph, wp_m_eager):
      for fname in (
        "body_mass",
        "body_subtreemass",
        "body_invweight0",
        "dof_invweight0",
      ):
        arr = getattr(wm, fname).numpy()
        setattr(
          wm, fname, wp.array(np.repeat(arr, 2, axis=0), dtype=getattr(wm, fname).dtype)
        )

    with wp.ScopedCapture() as cap:
      mjwarp.set_const(wp_m_graph, wp_d_graph, groups=mjwarp.SetConstGroup.ALL)

    for scale in ((1.2, 2.4), (0.7, 3.1)):
      for wm in (wp_m_graph, wp_m_eager):
        masses = wm.body_mass.numpy()
        masses[0, 1] = model.body_mass[1] * scale[0]
        masses[1, 1] = model.body_mass[1] * scale[1]
        wm.body_mass.assign(masses)

      wp.capture_launch(cap.graph)
      mjwarp.set_const(wp_m_eager, wp_d_eager, groups=mjwarp.SetConstGroup.ALL)
      wp.synchronize()

      np.testing.assert_allclose(
        wp_m_graph.body_subtreemass.numpy(),
        wp_m_eager.body_subtreemass.numpy(),
        rtol=1e-5,
      )
      np.testing.assert_allclose(
        wp_m_graph.dof_invweight0.numpy(),
        wp_m_eager.dof_invweight0.numpy(),
        rtol=1e-5,
      )
