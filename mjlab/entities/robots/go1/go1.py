"""Unitree Go1 quadruped."""

import mujoco
from typing import Tuple
from mujoco import mjx
import jax
import jax.numpy as jp

from mjlab.entities.robots.go1 import go1_constants as consts
from mjlab.entities.robots import robot
from mjlab.utils import spec as spec_utils


class UnitreeGo1(robot.Robot):
  """Unitree Go1 quadruped."""

  def __init__(self, config: robot.RobotConfig = consts.DefaultConfig):
    spec = mujoco.MjSpec.from_file(str(consts.GO1_XML), assets=consts.get_assets())
    super().__init__(spec=spec, config=config)

    self._joints = spec_utils.get_non_root_joints(self.spec)
    self._actuators = self.spec.actuators
    self._torso_body = self.spec.body(consts.TORSO_BODY)
    self._imu_site = self.spec.site(consts.IMU_SITE)
    self._gyro_sensor = self.spec.sensor("gyro")
    self._local_linvel_sensor = self.spec.sensor("local_linvel")
    self._upvector_sensor = self.spec.sensor("upvector")

  @property
  def joints(self) -> Tuple[mujoco.MjsJoint]:
    return self._joints

  @property
  def actuators(self) -> Tuple[mujoco.MjsActuator]:
    return self._actuators

  @property
  def joint_names(self) -> Tuple[str, ...]:
    return tuple([j.name for j in self._joints])

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
