"""MuJoCo humanoid."""

from mjlab.entities.robots import robot
from typing import Tuple
from mujoco import mjx
import jax
import mujoco

from mjlab.entities.robots.mujoco_humanoid import mujoco_humanoid_constants as consts
from mjlab.utils import spec as spec_utils


class MujocoHumanoid(robot.Robot):
  """MuJoCo ragdoll humanoid."""

  def __init__(self):
    spec = mujoco.MjSpec.from_file(str(consts.RAGDOLL_XML))
    super().__init__(spec=spec)

    self._joints = spec_utils.get_non_root_joints(self.spec)
    self._actuators = self.spec.actuators

    self._torso_body = self.spec.body(consts.TORSO_BODY)
    self._head_body = self.spec.body(consts.HEAD_BODY)
    self._com_vel_sensor = self.spec.sensor("torso_subtreelinvel")

    extremities = []
    for side in ("left", "right"):
      for limb in ("hand", "foot"):
        extremities.append(self.spec.body(f"{side}_{limb}"))
    self._extremities = tuple(extremities)

  @property
  def joints(self) -> Tuple[mujoco.MjsJoint, ...]:
    return self._joints

  @property
  def actuators(self) -> Tuple[mujoco.MjsActuator, ...]:
    return self._actuators

  def torso_upright(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns projection from z-axes of torso to the z-axes of the world."""
    return data.bind(model, self._torso_body).xmat[2, 2]

  def head_height(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns the height of the head above the ground."""
    return data.bind(model, self._head_body).xpos[2]

  def center_of_mass_position(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns position of the center-of-mass."""
    return data.bind(model, self._torso_body).subtree_com

  def center_of_mass_velocity(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns velocity of the center-of-mass."""
    return data.bind(model, self._com_vel_sensor).sensordata

  def torso_vertical_orientation(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns the z-projection of the torso orientation matrix."""
    return data.bind(model, self._torso_body).xmat[2]

  def joint_angles(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns the state without global orientation or position."""
    return data.bind(model, self._joints).qpos

  def extremities(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    """Returns end-effector positions in egocentric frame."""
    torso_frame = data.bind(model, self._torso_body).xmat
    torso_pos = data.bind(model, self._torso_body).xpos
    torso_to_limb = data.bind(model, self._extremities).xpos - torso_pos
    return torso_to_limb @ torso_frame
