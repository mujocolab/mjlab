"""MuJoCo constant recomputation through the batched MJWarp model."""

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim.sim import Simulation, SimulationCfg


@pytest.mark.parametrize(
  ("constraint", "explicit_relpose"),
  [("connect", False), ("weld", False), ("weld", True)],
)
def test_joint_default_pos_recomputes_each_worlds_constraint_pose(
  constraint, explicit_relpose
):
  relpose = 'relpose="0.1 0.2 0.3 1 0 0 0"' if explicit_relpose else ""
  equality = (
    f'<weld body1="link" body2="anchor" {relpose}/>'
    if constraint == "weld"
    else '<connect body1="link" body2="anchor" anchor="0.5 0.5 0"/>'
  )
  xml = f"""
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body name="base">
        <freejoint name="free_base"/>
        <geom type="sphere" size="0.1" mass="1" contype="0" conaffinity="0"/>
        <body name="link" pos="1 0 0">
          <joint name="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.1" mass="1" contype="0" conaffinity="0"/>
        </body>
      </body>
      <body name="anchor" pos="0 1 0">
        <freejoint/>
        <geom type="sphere" size="0.1" mass="1" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
    <equality>{equality}</equality>
  </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(xml)
  device = get_test_device()
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  fields = tuple(getattr(dr.joint_default_pos, "model_fields"))  # noqa: B009
  sim.expand_model_fields(fields)
  address = model.jnt_qposadr[model.joint("free_base").id]
  positions = (0.2, 0.8)
  sim.model.qpos0[:, address] = torch.tensor(positions, device=device)
  current_qpos = sim.data.qpos.clone()

  sim.recompute_constants(RecomputeLevel.set_const_0)

  expected = []
  for position in positions:
    native_model = mujoco.MjModel.from_xml_string(xml)
    native_model.qpos0[address] = position
    mujoco.mj_setConst(native_model, mujoco.MjData(native_model))
    expected.append(native_model.eq_data.copy())
  expected_tensor = torch.tensor(np.stack(expected), device=device)

  # MuJoCo 3.17 marks inferred weld poses in the new twelfth equality field.
  # Explicit relposes must stay fixed when qpos0 changes, while inferred ones move.
  if constraint == "weld" and model.eq_data.shape[1] == 12:
    assert bool(model.eq_data[0, 11]) == (not explicit_relpose)
    assert np.allclose(expected[0], expected[1]) == explicit_relpose
  elif constraint == "connect":
    assert not np.allclose(expected[0], expected[1])

  torch.testing.assert_close(
    sim.model.eq_data[:], expected_tensor.float(), atol=1e-5, rtol=1e-5
  )
  torch.testing.assert_close(sim.data.qpos[:], current_qpos, atol=0, rtol=0)
