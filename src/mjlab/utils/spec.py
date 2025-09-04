"""MjSpec utils."""

import mujoco


def get_non_root_joints(spec: mujoco.MjSpec) -> tuple[mujoco.MjsJoint, ...]:
  """Returns all joints except the root joint."""
  joints: list[mujoco.MjsJoint] = []
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      continue
    joints.append(jnt)
  return tuple(joints)


def get_root_joint(spec: mujoco.MjSpec) -> mujoco.MjsJoint | None:
  """Returns the root joint. None if no root joint exists."""
  joint: mujoco.MjsJoint | None = None
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      joint = jnt
      break
  return joint


def disable_collision(geom: mujoco.MjsGeom) -> None:
  geom.contype = 0
  geom.conaffinity = 0


def set_array_field(field, values):
  if values is None:
    return
  for i, v in enumerate(values):
    field[i] = v


def construct_contact_sensor_intprm(
  data: str,
  reduce: str,
  num: int = 1,
) -> list[int]:
  if num <= 0:
    raise ValueError("'num' must be positive")

  condata_map = {
    "found": 0,
    "force": 1,
    "torque": 2,
    "dist": 3,
    "pos": 4,
    "normal": 5,
    "tangent": 6,
  }
  reduce_map = {"none": 0, "mindist": 1, "maxforce": 2, "netforce": 3}

  if data:
    data_keys = data.split()
    values = [condata_map[k] for k in data_keys]
    for i in range(1, len(values)):
      if values[i] <= values[i - 1]:
        raise ValueError(
          f"Data attributes must be in order: {', '.join(condata_map.keys())}"
        )
    dataspec = sum(1 << v for v in values)
  else:
    dataspec = 1

  return [dataspec, reduce_map.get(reduce, 0), num]
