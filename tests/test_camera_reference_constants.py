"""Native parity for randomized camera and light reference poses."""

from importlib.metadata import version

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim import Simulation, SimulationCfg

_PUBLISHED_TRACKING_BUG = pytest.mark.xfail(
  version("mujoco-warp") in ("3.16.0", "3.17.0"),
  reason="Published MJWarp tracking recomputation requires the companion upstream fix",
  strict=False,
)


@pytest.mark.parametrize(
  "mode",
  [
    "fixed",
    pytest.param("track", marks=_PUBLISHED_TRACKING_BUG),
    pytest.param("trackcom", marks=_PUBLISHED_TRACKING_BUG),
  ],
)
def test_camera_light_reference_fields_are_independent_per_world(mode):
  xml = f"""
  <mujoco><option gravity="0 0 0"/><worldbody>
    <body name="base"><freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0.3 0.4 0.2"><geom type="sphere" size="0.1" mass="2"/></body>
      <camera pos="0.2 0.1 1" mode="{mode}"/>
      <light pos="0.2 0.1 1" dir="1 0 -1" mode="{mode}"/>
    </body>
  </worldbody></mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  device = get_test_device()
  sim = Simulation(2, SimulationCfg(), model=model, device=device)
  fields = tuple(getattr(dr.joint_default_pos, "model_fields"))  # noqa: B009
  sim.expand_model_fields(fields)
  poses = np.repeat(model.qpos0[None], 2, axis=0)
  poses[:, 0] = (0.2, 0.8)
  poses[1, 3:7] = (np.sqrt(0.5), 0, 0, np.sqrt(0.5))
  sim.model.qpos0[:] = torch.tensor(poses, device=device, dtype=torch.float32)
  physical_qpos = sim.data.qpos[:].clone()
  sim.recompute_constants(RecomputeLevel.set_const_0)
  torch.testing.assert_close(sim.data.qpos[:], physical_qpos, atol=0, rtol=0)
  sim.forward()

  expected = []
  for pose in poses:
    native = mujoco.MjModel.from_xml_string(xml)
    native.qpos0[:] = pose
    data = mujoco.MjData(native)
    mujoco.mj_setConst(native, data)
    data.qpos[:] = physical_qpos[0].cpu().numpy()
    mujoco.mj_forward(native, data)
    expected.append((native, data))
  references = (
    "cam_pos0",
    "cam_poscom0",
    "cam_mat0",
    "light_pos0",
    "light_poscom0",
    "light_dir0",
  )
  for field in references:
    assert getattr(sim.wp_model, field).shape[0] == 2
    actual = getattr(sim.model, field)[:].cpu().numpy()
    reference = np.stack([getattr(native, field) for native, _ in expected])
    np.testing.assert_allclose(
      actual, reference.reshape(actual.shape), atol=1e-5, rtol=1e-5
    )
  assert not np.allclose(expected[0][0].cam_pos0, expected[1][0].cam_pos0)
  for field in ("cam_xpos", "cam_xmat", "light_xpos", "light_xdir"):
    actual = getattr(sim.data, field)[:].cpu().numpy()
    reference = np.stack([getattr(data, field) for _, data in expected])
    np.testing.assert_allclose(
      actual, reference.reshape(actual.shape), atol=1e-5, rtol=1e-5
    )
