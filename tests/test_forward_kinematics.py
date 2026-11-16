"""Forward kinematics gives MuJoCo's body poses and link velocities."""

import mujoco
import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.utils.forward_kinematics import ForwardKinematics

# Several hinges in one body, welded bodies and a branch.
_STRESS_XML = """
<mujoco>
  <default><geom type="sphere" size="0.03" mass="0.1"/></default>
  <worldbody>
    <body name="root" pos="0.3 -0.2 1.0" quat="0.9 0.1 0.3 -0.2">
      <freejoint/>
      <geom/>
      <body name="multi" pos="0.1 0.05 -0.02" quat="0.8 0.3 -0.4 0.2">
        <joint axis="1 0.2 0" range="-2 2"/>
        <joint axis="0 1 0.3" range="-2 2"/>
        <joint axis="0.1 0 1" range="-2 2"/>
        <geom/>
        <body name="weld" pos="0.0 0.1 0.2" quat="0.5 0.5 0.5 0.5">
          <geom/>
          <body name="after_weld" pos="0.05 0 0.1" quat="0.7 0 0.7 0">
            <joint axis="0 0 1" range="-1 1"/>
            <geom/>
            <body name="leaf_weld" pos="0.1 0.1 0" euler="0.3 0.2 0.1"><geom/></body>
          </body>
        </body>
        <body name="branch" pos="-0.1 0 0.05" euler="0.4 -0.3 0.9">
          <joint axis="0 1 0" range="-1.5 1.5"/>
          <geom/>
        </body>
      </body>
      <body name="weld_root" pos="0 0 0.3" euler="0.2 0.5 -0.1"><geom/></body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize(
  "load",
  [
    lambda: mujoco.MjModel.from_xml_path(str(G1_XML)),
    lambda: mujoco.MjModel.from_xml_string(_STRESS_XML),
  ],
  ids=["g1", "stress"],
)
def test_matches_mujoco(load) -> None:
  model = load()
  hinge = mujoco.mjtJoint.mjJNT_HINGE
  joints = [j for j in range(model.njnt) if model.jnt_type[j] == hinge]
  q_adr, v_adr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
  rng = np.random.default_rng(0)
  model.jnt_pos[joints] = rng.normal(size=(len(joints), 3)) * 0.05
  model.qpos0[q_adr] = rng.uniform(-1.0, 1.0, len(joints))
  n, nb = 32, model.nbody - 1
  qpos = np.zeros((n, model.nq))
  qpos[:, :7] = rng.normal(size=(n, 7))  # MuJoCo normalizes the root quaternion
  low, high = model.jnt_range[joints].T
  qpos[:, q_adr] = rng.uniform(low, high, (n, len(joints)))
  qvel = rng.normal(size=(n, model.nv)) * 3.0

  data, xbody = mujoco.MjData(model), mujoco.mjtObj.mjOBJ_XBODY
  pose, vel = np.zeros((n, nb, 7)), np.zeros((n, nb, 6))  # vel is [ang, lin]
  for i in range(n):
    data.qpos[:], data.qvel[:] = qpos[i], qvel[i]
    mujoco.mj_forward(model, data)
    pose[i] = np.concatenate([data.xpos[1:], data.xquat[1:]], axis=-1)
    for b in range(nb):
      mujoco.mj_objectVelocity(model, data, xbody, b + 1, vel[i, b], 0)

  rows = torch.from_numpy(rng.permutation(nb))
  for dtype, tol in ((torch.float64, 1e-9), (torch.float32, 1e-4)):
    fk = ForwardKinematics(
      model, torch.arange(1, model.nbody), torch.tensor(q_adr), rows, dtype
    )
    state = [
      qpos[:, :3],
      qpos[:, 3:7],
      qpos[:, q_adr],
      vel[:, 0, 3:],
      vel[:, 0, :3],
      qvel[:, v_adr],
    ]
    state = [torch.from_numpy(x).to(dtype).unflatten(0, (4, 8)) for x in state]
    pos, quat, lin, ang = (x.flatten(0, 1).double() for x in fk(*state))
    expected = torch.from_numpy(np.concatenate([pose, vel[..., 3:], vel[..., :3]], -1))
    got = torch.cat([pos, quat, lin, ang], dim=-1)
    assert (got - expected[:, rows]).abs().max() < tol


# Worldbody contents of robots that forward kinematics rejects, with a part of the
# error each one raises. The requested body is "tracked".
_REJECTED = {
  "fixed_base": (
    """
    <body name="root"><geom size="0.1"/>
      <body name="tracked"><joint axis="0 0 1"/><geom size="0.1"/></body>
    </body>
    """,
    "free-joint root",
  ),
  "slide_joint_on_path": (
    """
    <body name="root"><freejoint/><geom size="0.1"/>
      <body name="tracked"><joint type="slide" axis="0 0 1"/><geom size="0.1"/></body>
    </body>
    """,
    "free-joint root",
  ),
  "ball_joint": (
    """
    <body name="root"><freejoint/><geom size="0.1"/>
      <body name="neck"><joint type="ball"/><geom size="0.1"/></body>
      <body name="tracked"><joint axis="0 0 1"/><geom size="0.1"/></body>
    </body>
    """,
    "one degree of freedom",
  ),
  "first_body_not_root": (
    """
    <body name="base"><geom size="0.1"/></body>
    <body name="tracked"><freejoint/><geom size="0.1"/></body>
    """,
    "free-joint root",
  ),
}


@pytest.mark.parametrize(
  ("worldbody", "match"), list(_REJECTED.values()), ids=list(_REJECTED)
)
def test_rejects_unsupported_robot(worldbody: str, match: str) -> None:
  xml = f"<mujoco><worldbody>{worldbody}</worldbody></mujoco>"
  model = mujoco.MjModel.from_xml_string(xml)
  free = mujoco.mjtJoint.mjJNT_FREE
  joint_q_adr = torch.arange(7 if model.jnt_type[0] == free else 0, model.nq)
  body_rows = torch.tensor([model.body("tracked").id - 1])
  with pytest.raises(ValueError, match=match):
    ForwardKinematics(model, torch.arange(1, model.nbody), joint_q_adr, body_rows)
