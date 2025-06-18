"""Unitree Go1 quadruped."""

import mujoco
from typing import Tuple
from mujoco import mjx
import jax
import jax.numpy as jp
from mjlab.core import entity

from mjlab.entities.go1 import go1_constants as consts


class UnitreeGo1(entity.Entity):
  """Unitree Go1 quadruped."""

  def post_init(self):
    for keyframe in consts.KEYFRAMES:
      self.add_keyframe(keyframe, ctrl=keyframe.joint_angles)

    for sensor in consts.SENSORS:
      self.add_sensor(sensor)

    self._torso_body = self.spec.body(consts.TORSO_BODY)
    self._imu_site = self.spec.site(consts.IMU_SITE)
    self._joints = self.get_non_root_joints()
    self._actuators = self.spec.actuators
    self._gyro_sensor = self.spec.sensor("gyro")
    self._local_linvel_sensor = self.spec.sensor("local_linvel")
    self._upvector_sensor = self.spec.sensor("upvector")

  @property
  def joints(self) -> Tuple[mujoco.MjsJoint]:
    return self._joints

  @property
  def actuators(self) -> Tuple[mujoco.MjsActuator]:
    return self._actuators

  def joint_angles(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._joints).qpos

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

  def upvector(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._upvector_sensor).sensordata
