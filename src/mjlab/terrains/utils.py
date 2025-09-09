import mujoco


def make_border(
  spec: mujoco.MjSpec,
  size: tuple[float, float],
  inner_size: tuple[float, float],
  height: float,
  position: tuple[float, float, float],
):
  boxes = []

  thickness_x = (size[0] - inner_size[0]) / 2.0
  thickness_y = (size[1] - inner_size[1]) / 2.0

  box_dims = (size[0], thickness_y, height)

  # Top.
  box_pos = (
    position[0],
    position[1] + inner_size[1] / 2.0 + thickness_y / 2.0,
    position[2],
  )
  box = spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
    pos=box_pos,
  )
  boxes.append(box)

  # Bottom.
  box_pos = (
    position[0],
    position[1] - inner_size[1] / 2.0 - thickness_y / 2.0,
    position[2],
  )
  box = spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
    pos=box_pos,
  )
  boxes.append(box)

  box_dims = (thickness_x, inner_size[1], height)

  # Left.
  box_pos = (
    position[0] - inner_size[0] / 2.0 - thickness_x / 2.0,
    position[1],
    position[2],
  )
  box = spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
    pos=box_pos,
  )
  boxes.append(box)

  # Right.
  box_pos = (
    position[0] + inner_size[0] / 2.0 + thickness_x / 2.0,
    position[1],
    position[2],
  )
  box = spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(box_dims[0] / 2.0, box_dims[1] / 2.0, box_dims[2] / 2.0),
    pos=box_pos,
  )
  boxes.append(box)

  return boxes
