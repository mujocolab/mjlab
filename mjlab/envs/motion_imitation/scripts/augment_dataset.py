"""Augments motion data from unitreerobotics/LAFAN1_Retargeting_Dataset with additional
information used for our tracking task."""

from pathlib import Path

import mujoco
import numpy as np
import tyro
from mink import SO3

from mjlab.entities.g1 import get_assets, G1_XML
from mjlab import LAFAN1_DATA_DIR, MOTION_DATA_DIR

_FPS = 30.0
_INPUT_DIR = LAFAN1_DATA_DIR
_OUTPUT_DIR = Path(MOTION_DATA_DIR / "processed")


def derivative(data: np.ndarray, dt: float) -> np.ndarray:
  """Computes the derivative along the batch dimension.

  Args:
    data: shape (B, D).
    dt: time step.

  Returns:
    shape (B, D).
  """
  return np.gradient(data, dt, axis=0)


def so3_derivative(rotations, dt: float) -> np.ndarray:
  """Computes the derivative of a sequence of SO3 rotations.

  Args:
    rotations: shape (B, 4).
    dt: time step.

  Returns:
    shape (B, 3).
  """
  rotations = np.array(rotations)
  B = rotations.shape[0]
  omegas = np.empty((B, 3))
  omegas[0] = (SO3(rotations[1]) @ SO3(rotations[0]).inverse()).log() / dt
  for i in range(1, B - 1):
    omegas[i] = ((SO3(rotations[i + 1]) @ SO3(rotations[i - 1]).inverse()).log()) / (
      2 * dt
    )
  omegas[-1] = ((SO3(rotations[-1]) @ SO3(rotations[-2]).inverse()).log()) / dt
  return omegas


def main(
  name: str,
  input_dir: Path = _INPUT_DIR,
  output_dir: Path = _OUTPUT_DIR,
  compare_with_jac: bool = False,
) -> None:
  # Load G1 model.
  model = mujoco.MjModel.from_xml_path(str(G1_XML), assets=get_assets())
  data = mujoco.MjData(model)

  motion_path = input_dir / f"{name}.csv"
  if not motion_path.exists():
    raise FileNotFoundError(f"Motion file {motion_path} does not exist.")
  traj = np.genfromtxt(motion_path, delimiter=",")

  # Reformat qpos since MuJoCo uses `wxyz` quaternion order.
  qpos_traj: list[np.ndarray] = []
  for i in range(traj.shape[0]):
    qpos = np.empty((model.nq,))
    qpos[:3] = traj[i, :3]  # xyz.
    qpos[3:7] = np.roll(traj[i, 3:7], 1)  # xyzw -> wxyz.
    qpos[7:] = traj[i, 7:]  # joint angles.
    qpos_traj.append(qpos)
  qpos_traj = np.array(qpos_traj)  # shape (B, nq)

  # Finite-difference to get velocities.
  dt = 1.0 / _FPS
  qvel_traj = np.zeros((len(qpos_traj), model.nv))
  qvel_traj[:, :3] = derivative(qpos_traj[:, :3], dt)
  qvel_traj[:, 3:6] = so3_derivative(qpos_traj[:, 3:7], dt)
  qvel_traj[:, 6:] = derivative(qpos_traj[:, 7:], dt)

  # Forward kinematics to get body poses in the world frame.
  fk_traj = {
    # Body positions in the world frame.
    "xpos": [],
    # Body orientations in the world frame.
    "xquat": [],
    # Body linear velocities in the world frame.
    "linvel": [],
    # Body angular velocities in the world frame.
    "angvel": [],
    # COM positions in the world frame.
    "com": [],
  }
  for i in range(qpos_traj.shape[0]):
    # NOTE(kevin): Need to set qvel to populate `data.cvel`.
    data.qpos[:] = qpos_traj[i]
    data.qvel[:] = qvel_traj[i]
    mujoco.mj_forward(model, data)

    body_pos = data.xpos.copy()  # Body positions in the world frame.
    body_rot = data.xquat.copy()  # Body orientations in the world frame (quaternions).

    # cvel is stored as rot:lin, see: https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html
    # To transform COM frame velocities to world frame:
    # 1. Linear velocity: v_world = v_com - ω × r, where:
    #    - v_com is linear velocity in COM frame
    #    - ω is angular velocity
    #    - r is offset vector from COM to body
    #    - × is cross product
    # 2. Angular velocity remains the same in both frames
    com_rotlin = data.cvel.copy()
    com_vel_lin = com_rotlin[:, 3:6]
    com_vel_rot = com_rotlin[:, 0:3]
    offset = body_pos - data.subtree_com[model.body_rootid]
    linvel = com_vel_lin - np.cross(offset, com_vel_rot)
    angvel = com_vel_rot

    if compare_with_jac:
      linvel_from_jac = []
      angvel_from_jac = []
      jac = np.empty((6, model.nv))
      for i in range(model.nbody):
        mujoco.mj_jacBody(model, data, jac[:3], jac[3:], i)
        linvel_from_jac.append(jac[:3] @ data.qvel)
        angvel_from_jac.append(jac[3:] @ data.qvel)
      linvel_from_jac = np.array(linvel_from_jac)
      angvel_from_jac = np.array(angvel_from_jac)
      np.testing.assert_allclose(linvel, linvel_from_jac, atol=1e-10)
      np.testing.assert_allclose(angvel, angvel_from_jac, atol=1e-10)

    com_pos = data.subtree_com[0].copy()

    fk_traj["xpos"].append(body_pos)
    fk_traj["xquat"].append(body_rot)
    fk_traj["linvel"].append(linvel)
    fk_traj["angvel"].append(angvel)
    fk_traj["com"].append(com_pos)
  for attr in fk_traj.keys():
    fk_traj[attr] = np.array(fk_traj[attr])

  # Save.
  if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)
  np.savez(
    output_dir / f"{name}.npz",
    qpos=qpos_traj,
    qvel=qvel_traj,
    xpos=fk_traj["xpos"],
    xquat=fk_traj["xquat"],
    linvel=fk_traj["linvel"],
    angvel=fk_traj["angvel"],
    com=fk_traj["com"],
    dt=dt,
  )


if __name__ == "__main__":
  tyro.cli(main)
