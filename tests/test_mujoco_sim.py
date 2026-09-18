"""Characterization tests for the MuJoCo simulation backend."""

import mujoco
import numpy as np
import pytest
import torch

from mjlab.sim.mujoco_sim import (
  MujocoKinematics,
  MujocoSimDataWithKinematics,
  MujocoSimulation,
  MujocoSimulationWithKinematics,
)
from mjlab.sim.sim import SimulationCfg

_CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
      <geom type="box" size="0.2 0.15 0.1" mass="1"/>
      <body name="pole">
        <joint name="hinge_1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 1" size="0.045" mass="0.1"/>
        <site name="tip" pos="0 0 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide" joint="slider" gear="10"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def mj_model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_string(_CARTPOLE_XML)


@pytest.fixture
def mujoco_sim(mj_model: mujoco.MjModel) -> MujocoSimulation:
  return MujocoSimulation(
    num_envs=4, cfg=SimulationCfg(backend="mujoco"), model=mj_model
  )


@pytest.fixture
def mujoco_sim_gpu(mj_model: mujoco.MjModel) -> MujocoSimulation:
  if not torch.cuda.is_available():
    pytest.skip("CUDA not available")
  return MujocoSimulation(
    num_envs=4, cfg=SimulationCfg(backend="mujoco"), model=mj_model, device="cuda"
  )


def test_entity_gpu_regression():
  """entity.initialize() still works on the warp path after the import guard."""
  pytest.importorskip("mujoco_warp", reason="mujoco_warp not installed")
  from conftest import initialize_entity

  from mjlab.actuator.xml_actuator import XmlActuatorCfg
  from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg

  if not torch.cuda.is_available():
    pytest.skip("CUDA not available")

  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(_CARTPOLE_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=("slider",)),)
    ),
  )
  entity = Entity(cfg)
  entity, sim = initialize_entity(entity, device="cuda", num_envs=2)
  assert entity.indexing is not None
  assert entity.data.joint_pos.shape == (2, 2)


def test_step_advances_qpos(mujoco_sim: MujocoSimulation):
  """Step() changes qpos."""
  qpos_before = mujoco_sim.data.qpos.clone()
  mujoco_sim.data.ctrl[:] = 1.0
  mujoco_sim.step()
  assert not torch.allclose(mujoco_sim.data.qpos, qpos_before)


def test_reset_restores_state(mujoco_sim: MujocoSimulation):
  """Reset() with no env_ids restores all envs to default qpos/qvel."""
  qpos0 = mujoco_sim.data.qpos.clone()
  qvel0 = mujoco_sim.data.qvel.clone()

  mujoco_sim.data.ctrl[:] = 1.0
  for _ in range(10):
    mujoco_sim.step()

  assert not torch.allclose(mujoco_sim.data.qpos, qpos0)
  mujoco_sim.reset()

  torch.testing.assert_close(mujoco_sim.data.qpos, qpos0)
  torch.testing.assert_close(mujoco_sim.data.qvel, qvel0)


def test_reset_selective(mujoco_sim: MujocoSimulation):
  """reset(env_ids) only restores the specified envs."""
  qpos0 = mujoco_sim.data.qpos.clone()

  mujoco_sim.data.ctrl[:] = 1.0
  for _ in range(10):
    mujoco_sim.step()

  qpos_after = mujoco_sim.data.qpos.clone()
  mujoco_sim.reset(torch.tensor([1, 3]))

  torch.testing.assert_close(mujoco_sim.data.qpos[1], qpos0[1])
  torch.testing.assert_close(mujoco_sim.data.qpos[3], qpos0[3])
  torch.testing.assert_close(mujoco_sim.data.qpos[0], qpos_after[0])
  torch.testing.assert_close(mujoco_sim.data.qpos[2], qpos_after[2])


def test_forward_is_noop(mujoco_sim: MujocoSimulation):
  """Forward() does not change qpos or qvel."""
  mujoco_sim.data.ctrl[:] = 0.5
  for _ in range(5):
    mujoco_sim.step()

  qpos_before = mujoco_sim.data.qpos.clone()
  qvel_before = mujoco_sim.data.qvel.clone()
  mujoco_sim.forward()

  torch.testing.assert_close(mujoco_sim.data.qpos, qpos_before)
  torch.testing.assert_close(mujoco_sim.data.qvel, qvel_before)


def test_kinematics_matches_mj_forward(
  mujoco_sim: MujocoSimulation, mj_model: mujoco.MjModel
):
  """The mj_kinematics->mj_comPos->mj_comVel chain matches a full mj_forward.

  Guards the swap away from full mj_forward in MujocoKinematics.forward: the cheaper
  chain must still produce identical xpos/xquat/subtree_com/cvel for every env.
  """
  mujoco_sim.data.ctrl[:] = 1.0
  for _ in range(5):
    mujoco_sim.step()

  kin = MujocoKinematics(mujoco_sim)
  kin.forward(mujoco_sim)

  ref = mujoco.MjData(mj_model)
  qpos = mujoco_sim.data.qpos.cpu().numpy()
  qvel = mujoco_sim.data.qvel.cpu().numpy()
  for i in range(mujoco_sim.num_envs):
    ref.qpos[:] = qpos[i]
    ref.qvel[:] = qvel[i]
    mujoco.mj_forward(mj_model, ref)
    np.testing.assert_allclose(kin.body_xpos[i], ref.xpos, atol=1e-9)
    np.testing.assert_allclose(kin.body_xquat[i], ref.xquat, atol=1e-9)
    np.testing.assert_allclose(kin.subtree_com[i], ref.subtree_com, atol=1e-9)
    np.testing.assert_allclose(kin.cvel[i], ref.cvel, atol=1e-9)
    np.testing.assert_allclose(
      kin.body_xmat[i], ref.xmat.reshape(mj_model.nbody, 3, 3), atol=1e-9
    )
    np.testing.assert_allclose(kin.site_xpos[i], ref.site_xpos, atol=1e-9)
    np.testing.assert_allclose(
      kin.site_xmat[i], ref.site_xmat.reshape(mj_model.nsite, 3, 3), atol=1e-9
    )


def test_kinematics_shared_scratch_refreshes_mjdata(mj_model: mujoco.MjModel):
  """A shared scratch MjData is forwarded in place."""
  sim = MujocoSimulation(
    num_envs=1, cfg=SimulationCfg(backend="mujoco"), model=mj_model
  )
  sim.data.ctrl[:] = 1.0
  for _ in range(5):
    sim.step()

  kin = MujocoKinematics(sim, scratch_mjdata=sim.mj_data)
  kin.forward(sim)

  np.testing.assert_allclose(
    sim.mj_data.qpos, sim.data.qpos[0].cpu().numpy(), atol=1e-9
  )
  np.testing.assert_allclose(sim.mj_data.xpos, kin.body_xpos[0], atol=1e-12)


def _assert_data_matches_qpos(
  sim: MujocoSimulationWithKinematics,
  mj_model: mujoco.MjModel,
  qpos: np.ndarray | None = None,
  qvel: np.ndarray | None = None,
) -> None:
  """data fields reflect a full mj_forward of qpos/qvel.

  Defaults to the sim's current data.qpos/qvel, which is what forward() and reset()
  leave the kinematics consistent with. step() refreshes kinematics from the
  pre-step state before advancing physics, so its expected state is the pre-step
  qpos/qvel, passed explicitly here.
  """
  ref = mujoco.MjData(mj_model)
  if qpos is None:
    qpos = sim.data.qpos.cpu().numpy()
  if qvel is None:
    qvel = sim.data.qvel.cpu().numpy()
  for i in range(sim.num_envs):
    ref.qpos[:] = qpos[i]
    ref.qvel[:] = qvel[i]
    mujoco.mj_forward(mj_model, ref)
    np.testing.assert_allclose(sim.data.xpos[i].cpu().numpy(), ref.xpos, atol=1e-5)
    np.testing.assert_allclose(sim.data.xquat[i].cpu().numpy(), ref.xquat, atol=1e-5)
    np.testing.assert_allclose(
      sim.data.subtree_com[i].cpu().numpy(), ref.subtree_com, atol=1e-5
    )
    np.testing.assert_allclose(sim.data.cvel[i].cpu().numpy(), ref.cvel, atol=1e-5)
    np.testing.assert_allclose(
      sim.data.site_xpos[i].cpu().numpy(), ref.site_xpos, atol=1e-5
    )
    np.testing.assert_allclose(
      sim.data.site_xmat[i].cpu().numpy(),
      ref.site_xmat.reshape(mj_model.nsite, 3, 3),
      atol=1e-5,
    )


def test_with_kinematics_step_forward_reset_refresh_data(mj_model: mujoco.MjModel):
  """The sim itself refreshes data kinematics inside step()/forward()/reset().

  Folding the MujocoKinematics engine into MujocoSimulationWithKinematics means
  callers never run a separate forward pass. step() refreshes kinematics from the
  pre-step state before advancing physics, so after step() data.xpos/... reflect the
  pre-step qpos/qvel and lag the post-step qpos by one step. forward() and reset()
  refresh after updating state, so they leave data consistent with the current state.
  """
  sim = MujocoSimulationWithKinematics(
    num_envs=4, cfg=SimulationCfg(backend="mujoco_with_kinematics"), model=mj_model
  )
  assert isinstance(sim.data, MujocoSimDataWithKinematics)

  sim.data.ctrl[:] = 1.0
  for _ in range(4):
    sim.step()
  # step() refreshes kinematics from the current (pre-step) state, then advances
  # physics, so the next step() leaves data.xpos reflecting this captured state.
  pre_qpos = sim.data.qpos.clone().cpu().numpy()
  pre_qvel = sim.data.qvel.clone().cpu().numpy()
  sim.step()
  _assert_data_matches_qpos(sim, mj_model, qpos=pre_qpos, qvel=pre_qvel)  # lags 1 step

  sim.forward()
  _assert_data_matches_qpos(sim, mj_model)  # forward() refreshed (current state)

  sim.reset()
  _assert_data_matches_qpos(sim, mj_model)  # reset() refreshed (default state)


def test_with_kinematics_refreshes_scratch_mjdata(mj_model: mujoco.MjModel):
  """step() forwards the shared scratch mj_data.

  step() refreshes kinematics from the pre-step state before advancing physics, so
  the scratch mj_data and data.xpos are both forwarded from that pre-step qpos and
  agree with each other, while data.qpos has already advanced one step ahead.
  """
  sim = MujocoSimulationWithKinematics(
    num_envs=1, cfg=SimulationCfg(backend="mujoco_with_kinematics"), model=mj_model
  )
  sim.data.ctrl[:] = 1.0
  for _ in range(4):
    sim.step()
  pre_qpos = sim.data.qpos[0].clone().cpu().numpy()
  sim.step()
  # The final step() forwarded the scratch mj_data from the pre-step qpos and
  # published data.xpos from that same forward, so the two agree.
  np.testing.assert_allclose(sim.mj_data.qpos, pre_qpos, atol=1e-6)
  np.testing.assert_allclose(
    sim.mj_data.xpos, sim.data.xpos[0].cpu().numpy(), atol=1e-6
  )


def test_wrong_backend_rejected(mj_model: mujoco.MjModel):
  """Each sim class validates cfg.backend against the variant it implements."""
  with pytest.raises(ValueError, match="mujoco"):
    MujocoSimulation(
      num_envs=1, cfg=SimulationCfg(backend="mujoco_with_kinematics"), model=mj_model
    )
  with pytest.raises(ValueError, match="mujoco_with_kinematics"):
    MujocoSimulationWithKinematics(
      num_envs=1, cfg=SimulationCfg(backend="mujoco"), model=mj_model
    )


def test_end_to_end_cartpole():
  """ManagerBasedRlEnv with backend='mujoco' runs a step with correct output shapes."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

  cfg = cartpole_balance_env_cfg()
  cfg.sim.backend = "mujoco"
  cfg.scene.num_envs = 4

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    action = torch.zeros(4, env.single_action_space.shape[0])
    obs, reward, terminated, timed_out, _ = env.step(action)

    assert isinstance(obs, dict)
    actor_obs: torch.Tensor = obs["actor"]  # type: ignore[assignment]
    assert actor_obs.shape == (
      4,
      5,
    )  # cart_pos(1) + pole_angle(2) + cart_vel(1) + pole_vel(1)
    assert reward.shape == (4,)
    assert terminated.shape == (4,)
  finally:
    env.close()


def test_end_to_end_with_kinematics_backend():
  """backend='mujoco_with_kinematics' makes the env build the kinematics sim."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

  cfg = cartpole_balance_env_cfg()
  cfg.sim.backend = "mujoco_with_kinematics"
  cfg.scene.num_envs = 2

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    assert isinstance(env.sim, MujocoSimulationWithKinematics)
    assert isinstance(env.sim.data, MujocoSimDataWithKinematics)
    env.reset()
    env.step(torch.zeros(2, env.single_action_space.shape[0]))
  finally:
    env.close()


def test_gpu_tensors_on_device(mujoco_sim_gpu: MujocoSimulation) -> None:
  """Tensors in data and model live on the requested GPU device."""
  assert mujoco_sim_gpu.data.qpos.device.type == "cuda"
  assert mujoco_sim_gpu.data.qvel.device.type == "cuda"
  assert mujoco_sim_gpu.data.ctrl.device.type == "cuda"
  assert mujoco_sim_gpu.data.sensordata.device.type == "cuda"
  assert mujoco_sim_gpu.model.jnt_range.device.type == "cuda"


def test_gpu_step_advances_qpos(mujoco_sim_gpu: MujocoSimulation) -> None:
  """step() changes qpos when running CPU sim with GPU tensors."""
  qpos_before = mujoco_sim_gpu.data.qpos.clone()
  mujoco_sim_gpu.data.ctrl[:] = 1.0
  mujoco_sim_gpu.step()
  assert not torch.allclose(mujoco_sim_gpu.data.qpos, qpos_before)


def test_gpu_reset_restores_state(mujoco_sim_gpu: MujocoSimulation) -> None:
  """reset() with no env_ids restores all envs to default qpos/qvel on GPU."""
  qpos0 = mujoco_sim_gpu.data.qpos.clone()
  qvel0 = mujoco_sim_gpu.data.qvel.clone()

  mujoco_sim_gpu.data.ctrl[:] = 1.0
  for _ in range(10):
    mujoco_sim_gpu.step()

  assert not torch.allclose(mujoco_sim_gpu.data.qpos, qpos0)
  mujoco_sim_gpu.reset()

  torch.testing.assert_close(mujoco_sim_gpu.data.qpos, qpos0)
  torch.testing.assert_close(mujoco_sim_gpu.data.qvel, qvel0)


def test_gpu_reset_selective(mujoco_sim_gpu: MujocoSimulation) -> None:
  """reset(env_ids) only restores the specified envs on GPU."""
  qpos0 = mujoco_sim_gpu.data.qpos.clone()

  mujoco_sim_gpu.data.ctrl[:] = 1.0
  for _ in range(10):
    mujoco_sim_gpu.step()

  qpos_after = mujoco_sim_gpu.data.qpos.clone()
  mujoco_sim_gpu.reset(torch.tensor([1, 3], device="cuda"))

  torch.testing.assert_close(mujoco_sim_gpu.data.qpos[1], qpos0[1])
  torch.testing.assert_close(mujoco_sim_gpu.data.qpos[3], qpos0[3])
  torch.testing.assert_close(mujoco_sim_gpu.data.qpos[0], qpos_after[0])
  torch.testing.assert_close(mujoco_sim_gpu.data.qpos[2], qpos_after[2])


# ---------------------------------------------------------------------------
# Mocap tests
# ---------------------------------------------------------------------------

_MOCAP_XML = """
<mujoco model="mocap_ball">
  <option timestep="0.005"/>
  <worldbody>
    <body name="mocap_body" pos="0 0 1" mocap="true">
      <geom type="sphere" size="0.1" contype="0" conaffinity="0"/>
    </body>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="mocap_body" body2="ball" solimp="0.95 0.99 0.001" solref="0.002 1"/>
  </equality>
</mujoco>
"""


@pytest.fixture
def mj_model_mocap() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_string(_MOCAP_XML)


@pytest.fixture
def mujoco_sim_mocap(mj_model_mocap: mujoco.MjModel) -> MujocoSimulation:
  return MujocoSimulation(
    num_envs=4, cfg=SimulationCfg(backend="mujoco"), model=mj_model_mocap
  )


def test_mocap_flows_into_physics(mujoco_sim_mocap: MujocoSimulation) -> None:
  """Moving mocap_pos for one env produces different qpos than the default env."""
  mujoco_sim_mocap.data.mocap_pos[1, 0] = torch.tensor([1.0, 0.0, 1.0])
  for _ in range(20):
    mujoco_sim_mocap.step()
  # The weld constraint pulls env 1's ball toward x=1.0. env 0's mocap stays at
  # x=0.0, so its ball stays near x=0.0. qpos[..., 0] is the freejoint x translation.
  assert mujoco_sim_mocap.data.qpos[1, 0] > mujoco_sim_mocap.data.qpos[0, 0]


def test_reset_restores_mocap(mujoco_sim_mocap: MujocoSimulation) -> None:
  """reset() restores mocap_pos and mocap_quat to model defaults."""
  default_mocap_pos = mujoco_sim_mocap.data.mocap_pos.clone()
  default_mocap_quat = mujoco_sim_mocap.data.mocap_quat.clone()
  mujoco_sim_mocap.data.mocap_pos[:] = torch.tensor([5.0, 0.0, 0.0])
  mujoco_sim_mocap.reset()
  torch.testing.assert_close(mujoco_sim_mocap.data.mocap_pos, default_mocap_pos)
  torch.testing.assert_close(mujoco_sim_mocap.data.mocap_quat, default_mocap_quat)


def test_reset_selective_mocap(mujoco_sim_mocap: MujocoSimulation) -> None:
  """reset(env_ids) only restores mocap_pos and mocap_quat for the specified envs."""
  default_mocap_pos = mujoco_sim_mocap.data.mocap_pos.clone()
  default_mocap_quat = mujoco_sim_mocap.data.mocap_quat.clone()
  mujoco_sim_mocap.data.mocap_pos[:] = torch.tensor([5.0, 0.0, 0.0])
  mujoco_sim_mocap.reset(torch.tensor([0, 2]))
  torch.testing.assert_close(mujoco_sim_mocap.data.mocap_pos[0], default_mocap_pos[0])
  torch.testing.assert_close(mujoco_sim_mocap.data.mocap_pos[2], default_mocap_pos[2])
  torch.testing.assert_close(mujoco_sim_mocap.data.mocap_quat[0], default_mocap_quat[0])
  torch.testing.assert_close(mujoco_sim_mocap.data.mocap_quat[2], default_mocap_quat[2])
  expected = torch.tensor([[5.0, 0.0, 0.0]])
  torch.testing.assert_close(mujoco_sim_mocap.data.mocap_pos[1], expected)
  torch.testing.assert_close(mujoco_sim_mocap.data.mocap_pos[3], expected)


def test_model_bridge_field_cache(mj_model: mujoco.MjModel) -> None:
  """MujocoModelBridge._field_cache contains the correct cached fields."""
  sim = MujocoSimulation(
    num_envs=1, cfg=SimulationCfg(backend="mujoco"), model=mj_model
  )
  model_bridge = sim.model
  # The field cache should contain an entry for each top-level field in the mjModel struct.
  for field in dir(mj_model):
    if not field.startswith("_") and not callable(getattr(mj_model, field)):
      # hasattr will not work because the attribute is lazy-loaded,
      # so we check that the attribute is not None instead.
      assert getattr(model_bridge, field) is not None
