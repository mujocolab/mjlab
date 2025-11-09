"""Tests for sim.py."""

import mujoco
import numpy as np
import pytest
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
    ls_parallel=False,
    mujoco=MujocoCfg(
      timestep=0.02,
      integrator="euler",
      solver="cg",
      iterations=7,
      ls_iterations=14,
      gravity=(0, 0, 7.5),
    ),
  )

  sim = Simulation(num_envs=1, cfg=cfg, model=model, device=device)

  # MujocoCfg should be applied to mj_model.
  assert sim.mj_model.opt.timestep == cfg.mujoco.timestep
  assert sim.mj_model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
  assert sim.mj_model.opt.solver == mujoco.mjtSolver.mjSOL_CG
  assert sim.mj_model.opt.iterations == cfg.mujoco.iterations
  assert tuple(sim.mj_model.opt.gravity) == cfg.mujoco.gravity

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

  # SimulationCfg should be applied to wp_model.
  assert sim.wp_model.opt.contact_sensor_maxmatch == cfg.contact_sensor_maxmatch
  assert sim.wp_model.opt.ls_parallel == cfg.ls_parallel
