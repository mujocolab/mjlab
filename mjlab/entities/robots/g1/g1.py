"""Unitree G1 humanoid."""

from typing import Tuple
import re
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jp

from mjlab.entities.g1 import g1_constants as consts
from mjlab.entities.robots import robot


class UnitreeG1(robot.Robot):
  """Unitree G1 humanoid."""

  def post_init(self):
    self.add_pd_actuators_from_patterns(consts.ACTUATOR_SPECS)

    self._torso_body = self.spec.body(consts.TORSO_BODY)
    self._imu_site = self.spec.site(consts.PELVIS_IMU_SITE)
    self._joints = self.get_non_root_joints()
    self._ankle_joints = tuple(
      [j for j in self._joints if re.match(r".*_ankle_(pitch|roll)_joint", j.name)]
    )
    self._actuators = tuple(self.spec.actuators)
    self._gyro_sensor = self.spec.sensor("gyro")
    self._local_linvel_sensor = self.spec.sensor("local_linvel")
    self._upvector_sensor = self.spec.sensor("upvector")

    freejoint = self.get_root_joint()
    assert freejoint is not None
    self._freejoint = freejoint

  @property
  def freejoint(self) -> mujoco.MjsJoint:
    return self._freejoint

  @property
  def joints(self) -> Tuple[mujoco.MjsJoint]:
    return self._joints

  @property
  def ankle_joints(self) -> Tuple[mujoco.MjsJoint]:
    return self._ankle_joints

  @property
  def actuators(self) -> Tuple[mujoco.MjsActuator]:
    return self._actuators

  @property
  def joint_names(self) -> Tuple[str, ...]:
    return tuple([j.name for j in self._joints])

  # Observations.

  def root_pos(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._freejoint).qpos[:3]

  def root_quat(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._freejoint).qpos[3:7]

  def joint_angles(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._joints).qpos

  def ankle_joint_angles(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._ankle_joints).qpos

  def joint_velocities(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._joints).qvel

  def joint_accelerations(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._joints).qacc

  def joint_torques(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._actuators).force

  def gyro(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._gyro_sensor).sensordata

  def local_linvel(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._local_linvel_sensor).sensordata

  def projected_gravity(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._imu_site).xmat.T @ jp.array([0, 0, -1.0])

  def torso_projected_gravity(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._torso_body).xmat.T @ jp.array([0, 0, -1.0])

  def upvector(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._upvector_sensor).sensordata
