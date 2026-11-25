"""Forward kinematics: world-frame pose and velocity of a floating-base robot's
bodies from its root and joint state, differentiable, in the convention of mjlab's
``body_link_pose_w`` and ``body_link_vel_w``.

Each body follows MuJoCo's kinematics: the parent frame, then the body's fixed
offset, then each of its hinges, turning ``qpos - qpos0`` about the hinge axis
through the hinge anchor. Every hinge, and every body without one, is a step.
Each returned body composes the steps on its path from the root, root first, and
sums their hinge twists, taken at the root origin; the root pose and twist apply
last. The pass runs on per-component ``(..., R)`` tensors, which a compiler fuses
into few kernels.
"""

import mujoco
import numpy as np
import torch

Vec = tuple[torch.Tensor, ...]


class ForwardKinematics:
  """Forward kinematics of the entity bodies at ``body_rows``, in output order.

  ``body_ids`` are the entity's global body ids, the first one the free-joint
  root, and ``joint_q_adr`` the qpos addresses in joint column order. Every joint
  there has one degree of freedom, and those on the paths to the returned bodies
  are hinges. ``joint_names`` holds the joints' names in column order, without
  scene prefixes.
  """

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    body_ids: torch.Tensor,
    joint_q_adr: torch.Tensor,
    body_rows: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
  ):
    rows = [int(body_ids[row]) for row in body_rows]
    col_of = {int(adr): col for col, adr in enumerate(joint_q_adr)}
    tree = set()
    for body in rows:
      while body and body not in tree:
        tree.add(body)
        body = int(mj_model.body_parentid[body])

    free, hinge = mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE
    zero, identity = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    # Steps are (joint column, qpos0, axis, anchor, offset pos, offset quat); one
    # without a hinge reads column J, a zero pad after the J joints. Step 0, the
    # identity, pads the paths; a body's path is its steps from the root, root first.
    fixed = (len(col_of), 0.0, zero, zero)
    steps: list[tuple] = [(*fixed, zero, identity)]
    path_of: dict[int, list[int]] = {int(body_ids[0]): []}
    for body in sorted(tree):  # MuJoCo numbers parents before children
      parent, first = int(mj_model.body_parentid[body]), int(mj_model.body_jntadr[body])
      joints = range(first, first + int(mj_model.body_jntnum[body]))
      kinds = [mj_model.jnt_type[j] for j in joints]
      root = body == int(body_ids[0])
      if (parent == 0) != root or kinds != ([free] if root else [hinge] * len(kinds)):
        raise ValueError(
          f"body {mj_model.body(body).name}: needs the entity's first body as its "
          "free-joint root under the world, with only hinges below it"
        )
      if root:
        continue
      adrs = [int(mj_model.jnt_qposadr[j]) for j in joints]
      hinges = [
        (col_of[a], mj_model.qpos0[a], mj_model.jnt_axis[j], mj_model.jnt_pos[j])
        for j, a in zip(joints, adrs, strict=True)
      ]
      path = path_of[parent]
      offset = mj_model.body_pos[body], mj_model.body_quat[body]
      for joint in hinges or [fixed]:
        steps.append((*joint, *offset))
        path, offset = [*path, len(steps) - 1], (zero, identity)
      path_of[body] = path

    # The returned bodies' paths as (depth, R) step indices, front-padded with step 0.
    paths = [path_of[body] for body in rows]
    depth = max(map(len, paths))
    paths = np.array([[0] * (depth - len(p)) + p for p in paths], dtype=int).T
    # Per step, with o its offset quaternion: the hinge anchor and axis in the parent
    # frame, the negated anchor in the step frame, o, o * (0, axis) and qpos0; the
    # step's quaternion is cos(h) o + sin(h) o * (0, axis) for h = (qpos - qpos0) / 2.
    consts = []
    for _, qpos0, axis, anchor, pos, quat in steps:
      parent_anchor, parent_axis, turned = np.zeros(3), np.zeros(3), np.zeros(4)
      mujoco.mju_rotVecQuat(parent_anchor, anchor, quat)
      mujoco.mju_rotVecQuat(parent_axis, axis, quat)
      mujoco.mju_mulQuat(turned, quat, np.concatenate([[0.0], axis]))
      consts.append(
        np.hstack([pos + parent_anchor, parent_axis, -anchor, quat, turned, qpos0])
      )
    self.consts = torch.as_tensor(np.array(consts)[paths], device=device, dtype=dtype)
    self.col = torch.as_tensor(np.array([s[0] for s in steps])[paths], device=device)
    joint_of = {int(adr): j for j, adr in enumerate(mj_model.jnt_qposadr)}
    if not col_of.keys() <= joint_of.keys():
      raise ValueError("joint_q_adr: every joint needs one degree of freedom")
    self.joint_names = [mj_model.joint(joint_of[a]).name.split("/")[-1] for a in col_of]

  def __call__(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Positions ``(..., R, 3)``, wxyz quaternions ``(..., R, 4)``, linear and
    angular velocities ``(..., R, 3)`` of the ``R`` bodies, in the world frame,
    from the root link's world pose and velocity and the joints ``(..., J)``."""
    joint_pos, joint_vel = (
      torch.nn.functional.pad(x, (0, 1)) for x in (joint_pos, joint_vel)
    )
    zero, one = root_pos.new_zeros(()), root_pos.new_ones(())
    pos = lin = ang = (zero, zero, zero)
    quat = (one, zero, zero, zero)
    for col, consts in zip(self.col, self.consts, strict=True):
      c = consts.unbind(-1)
      half = (joint_pos[..., col] - c[17]) / 2
      # The hinge anchor and twist in the root frame, from the parent's pose.
      anchor = _add(pos, _rotate(quat, c[:3]))
      twist = tuple(a * joint_vel[..., col] for a in _rotate(quat, c[3:6]))
      turn = tuple(
        half.cos() * o + half.sin() * t for o, t in zip(c[9:13], c[13:17], strict=True)
      )
      quat = _mul(quat, turn)
      pos = _add(anchor, _rotate(quat, c[6:9]))
      ang = _add(ang, twist)
      lin = _add(lin, _cross(anchor, twist))

    w, x, y, z = root_quat.unsqueeze(-2).unbind(-1)
    norm = (w * w + x * x + y * y + z * z).sqrt()
    root = (w / norm, x / norm, y / norm, z / norm)
    pos = _rotate(root, pos)
    ang = _add(root_ang_vel.unsqueeze(-2).unbind(-1), _rotate(root, ang))
    lin = _add(root_lin_vel.unsqueeze(-2).unbind(-1), _rotate(root, lin))
    lin = _add(lin, _cross(ang, pos))
    pos = _add(root_pos.unsqueeze(-2).unbind(-1), pos)
    state = torch.stack([*pos, *_mul(root, quat), *lin, *ang], dim=-1)
    pos_w, quat_w, lin_vel_w, ang_vel_w = state.split((3, 4, 3, 3), dim=-1)
    return pos_w, quat_w, lin_vel_w, ang_vel_w


def _add(a: Vec, b: Vec) -> Vec:
  return tuple(x + y for x, y in zip(a, b, strict=True))


def _cross(a: Vec, b: Vec) -> Vec:
  return (
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  )


def _rotate(q: Vec, v: Vec) -> Vec:
  """``v`` turned by the unit wxyz quaternion ``q``."""
  t = tuple(2 * x for x in _cross(q[1:], v))
  return _add(_add(v, tuple(q[0] * x for x in t)), _cross(q[1:], t))


def _mul(a: Vec, b: Vec) -> Vec:
  """The Hamilton product of wxyz quaternions."""
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return (
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  )
