"""Tests for the C MuJoCo backend, mostly as parity with the MJWarp one."""

import mujoco
import pytest
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim import MujocoSimulation, Simulation, SimulationCfg
from mjlab.tasks.registry import load_env_cfg

NUM_ENVS = 4

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="target" mocap="true" pos="1 0 1"/>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom name="torso" type="box" size=".1 .1 .1" mass="2"/>
      <site name="imu"/>
      <body name="arm" pos="0 0 .2">
        <joint name="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="link" type="capsule" size=".03" fromto="0 0 0 .3 0 0" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <contact><pair name="pair" geom1="floor" geom2="torso"/></contact>
  <tendon><fixed name="tendon"><joint joint="hinge" coef="1"/></fixed></tendon>
  <actuator><position name="servo" joint="hinge" kp="20" kv="1"/></actuator>
  <sensor><jointpos joint="hinge"/><framepos objtype="site" objname="imu"/></sensor>
</mujoco>
"""

DATA_FIELDS = (
  "time",
  "qpos",
  "qvel",
  "qacc",
  "qacc_warmstart",
  "ctrl",
  "act",
  "sensordata",
  "xpos",
  "xquat",
  "xmat",
  "xipos",
  "subtree_com",
  "cvel",
  "site_xpos",
  "site_xmat",
  "geom_xpos",
  "geom_xmat",
  "mocap_pos",
  "mocap_quat",
  "xfrc_applied",
  "qfrc_applied",
  "actuator_force",
  "ten_length",
  "ten_velocity",
)
EXPANDED = (
  "body_mass",
  "body_ipos",
  "body_inertia",
  "body_subtreemass",
  "dof_armature",
  "dof_invweight0",
  "geom_size",
  "geom_aabb",
  "geom_friction",
  "geom_rgba",
  "geom_matid",
  "actuator_gainprm",
  "actuator_biasprm",
  "actuator_acc0",
  "pair_friction",
)
SHARED = ("body_iquat", "jnt_range", "geom_type", "geom_bodyid", "site_bodyid", "nq")


@pytest.fixture
def sims() -> tuple[MujocoSimulation, Simulation]:
  def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(XML)

  cfg = SimulationCfg()
  cpu = MujocoSimulation(NUM_ENVS, cfg, model())
  warp = Simulation(NUM_ENVS, cfg, model(), device="cpu")
  return cpu, warp


def test_fields_are_laid_out_like_mjwarp(sims):
  cpu, warp = sims
  for sim in sims:
    sim.expand_model_fields(EXPANDED)
  for bridge, fields in (("data", DATA_FIELDS), ("model", EXPANDED + SHARED)):
    for name in fields:
      ours = getattr(getattr(cpu, bridge), name)
      theirs = getattr(getattr(warp, bridge), name)
      if isinstance(theirs, int):
        assert ours == theirs, name
        continue
      assert ours.shape == theirs.shape, name
      assert ours.dtype == theirs.dtype, name
  torch.testing.assert_close(
    cpu.get_default_field("body_mass"), warp.get_default_field("body_mass")
  )


def test_trajectory_matches_mjwarp(sims):
  torch.manual_seed(0)
  qvel = 0.5 * torch.randn(NUM_ENVS, 7)
  ctrl = torch.randn(NUM_ENVS, 1)
  for sim in sims:
    sim.reset()
    sim.data.qvel[:] = qvel
    sim.data.ctrl[:] = ctrl
    for _ in range(100):
      sim.step()
    sim.forward()
  cpu, warp = sims
  for name in ("qpos", "qvel", "sensordata", "xpos", "site_xmat", "actuator_force"):
    torch.testing.assert_close(
      getattr(cpu.data, name), getattr(warp.data, name)[:], atol=1e-3, rtol=0, msg=name
    )


def test_first_read_of_a_derived_field_is_current():
  sim = MujocoSimulation(NUM_ENVS, SimulationCfg(), mujoco.MjModel.from_xml_string(XML))
  sim.data.qpos[:, 2] = 3.0
  assert torch.all(sim.data.xpos[:, 2, 2] == 3.0)


def test_reset_restores_defaults_and_discards_pending_writes(sims):
  cpu, warp = sims
  ids = torch.tensor([1, 2])
  for sim in sims:
    sim.step()
    sim.data.qvel[:] = 1.0
    sim.data.xfrc_applied[:] = 5.0
    sim.data.mocap_pos[:] = 7.0
    sim.reset(ids)
    sim.forward()
  for name in ("time", "qpos", "qvel", "xfrc_applied", "mocap_pos", "ctrl"):
    torch.testing.assert_close(
      getattr(cpu.data, name), getattr(warp.data, name)[:], atol=1e-5, rtol=0, msg=name
    )
  assert torch.all(cpu.data.xfrc_applied[ids] == 0.0)
  assert torch.all(cpu.data.xfrc_applied[0] == 5.0)


def test_randomized_mass_reaches_physics_and_derived_constants(sims):
  mass = torch.tensor([1.0, 2.0, 4.0, 8.0])
  for sim in sims:
    sim.expand_model_fields(("body_mass", "body_subtreemass", "dof_invweight0"))
    sim.model.body_mass[:, 2] = mass
    sim.recompute_constants(RecomputeLevel.set_const)
    sim.data.ctrl[:] = 1.0
    for _ in range(20):
      sim.step()
  cpu, warp = sims
  assert torch.all(cpu.model.body_subtreemass[:, 2] == mass + 0.5)
  for name in ("body_subtreemass", "dof_invweight0"):
    torch.testing.assert_close(
      getattr(cpu.model, name),
      getattr(warp.model, name)[:],
      rtol=1e-4,
      atol=0,
      msg=name,
    )
  torch.testing.assert_close(cpu.data.qpos, warp.data.qpos[:], atol=1e-3, rtol=0)
  assert not torch.allclose(cpu.data.qpos[0], cpu.data.qpos[3], atol=1e-4)


def test_diverged_env_keeps_its_nans():
  sim = MujocoSimulation(NUM_ENVS, SimulationCfg(), mujoco.MjModel.from_xml_string(XML))
  sim.data.qvel[1] = torch.nan
  sim.step()
  assert sim.data.qpos[1].isnan().any()
  assert sim.data.qpos[0].isfinite().all()


def test_env_matches_mjwarp():
  """Same seed and actions through events, randomization, and resets."""
  qpos = {}
  for backend in ("mujoco", "mjwarp"):
    cfg = load_env_cfg("Mjlab-Lift-Cube-Yam")
    cfg.sim.backend = backend
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    for _ in range(5):
      env.step(torch.zeros(env.action_space.shape))
    qpos[backend] = env.sim.data.qpos.clone()
    env.close()
  torch.testing.assert_close(qpos["mujoco"], qpos["mjwarp"], atol=1e-4, rtol=0)


def test_sensors_that_render_need_mjwarp():
  cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-Go1")
  cfg.sim.backend = "mujoco"
  cfg.scene.num_envs = NUM_ENVS
  with pytest.raises(NotImplementedError, match="mjwarp"):
    ManagerBasedRlEnv(cfg, device="cpu")
