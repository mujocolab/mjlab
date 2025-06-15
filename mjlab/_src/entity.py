import abc
import fnmatch
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, TypeVar, Type

import mujoco
import numpy as np

from mjlab._src import sim_structs

T = TypeVar("T", bound="Entity")


class Entity(abc.ABC):
  """A wrapper around MjSpec with convenience methods."""

  def __init__(self, spec: mujoco.MjSpec):
    self._spec = spec

  @classmethod
  def from_file(
    cls: Type[T],
    xml_path: Path,
    assets: Optional[Dict[str, bytes]] = None,
  ) -> T:
    """Creates an entity from an XML file."""
    spec = mujoco.MjSpec.from_file(str(xml_path), assets=assets)
    return cls(spec)

  @classmethod
  def from_xml_str(
    cls: Type[T],
    xml_str: str,
    assets: Optional[Dict[str, bytes]] = None,
  ) -> T:
    spec = mujoco.MjSpec.from_string(xml_str, assets=assets)
    return cls(spec)

  # Methods.

  def compile(self) -> mujoco.MjModel:
    """Compiles the robot model into an MjModel."""
    return self.spec.compile()

  def write_xml(self, xml_path: Path) -> None:
    """Writes the robot model to an XML file."""
    with open(xml_path, "w") as f:
      f.write(self._spec.to_xml())

  def add_skybox(
    self,
    rgb1: Tuple[float, float, float] = (0.3, 0.5, 0.7),
    rgb2: Tuple[float, float, float] = (0.1, 0.2, 0.3),
    width: int = 512,
    height: int = 3072,
  ) -> None:
    """Add a skybox to the scene."""
    self._spec.add_texture(
      type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
      rgb1=rgb1,
      rgb2=rgb2,
      width=width,
      height=height,
    )

  def get_non_root_joints(self) -> Tuple[mujoco.MjsJoint]:
    """Returns all joints except the root joint."""
    joints = []
    for jnt in self._spec.joints:
      if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
        continue
      joints.append(jnt)
    return tuple(joints)

  def add_keyframe(
    self,
    keyframe: sim_structs.Keyframe,
    ctrl: Optional[np.ndarray] = None,
  ) -> None:
    """Adds a keyframe to the robot."""
    key = self._spec.add_key(name=keyframe.name, qpos=keyframe.qpos)
    if ctrl is not None:
      assert ctrl.shape == keyframe.joint_angles.shape
      key.ctrl = ctrl

  def add_collision_pair(self, collision_pair: sim_structs.CollisionPair) -> None:
    """Adds a collision pair to the robot."""
    pair = self._spec.add_pair(
      name=f"{collision_pair.geom1}__{collision_pair.geom2}",
      geomname1=collision_pair.geom1,
      geomname2=collision_pair.geom2,
      condim=collision_pair.condim,
    )
    if collision_pair.friction is not None:
      # TODO(kevin): This is ugly.
      for i in range(len(collision_pair.friction)):
        pair.friction[i] = collision_pair.friction[i]
    if collision_pair.solref is not None:
      pair.solref = collision_pair.solref

  def add_pd_actuator(self, actuator: sim_structs.PDActuator) -> None:
    """Adds a PD actuator to the robot."""
    act = self._spec.add_actuator(
      name=actuator.joint_name,
      target=actuator.joint_name,
      trntype=mujoco.mjtTrn.mjTRN_JOINT,
      gaintype=mujoco.mjtGain.mjGAIN_FIXED,
      biastype=mujoco.mjtBias.mjBIAS_AFFINE,
      inheritrange=1,
    )
    act.gainprm[0] = actuator.kp
    act.biasprm[1] = -actuator.kp
    act.biasprm[2] = -actuator.kv

  def add_pd_actuators_from_patterns(
    self,
    actuator_specs: Sequence[sim_structs.PDActuator],
    zero_out_joint_damping: bool = True,
  ) -> None:
    for joint in self.get_non_root_joints():
      for actuator_spec in actuator_specs:
        if fnmatch.fnmatch(joint.name, actuator_spec.joint_name):
          actual_spec = replace(actuator_spec, joint_name=joint.name)
          if zero_out_joint_damping:
            joint.damping = 0.0
          joint.armature = actual_spec.armature
          if actual_spec.torque_limit is not None:
            joint.actfrcrange[0] = -actual_spec.torque_limit
            joint.actfrcrange[1] = actual_spec.torque_limit
          self.add_pd_actuator(actual_spec)
          break

  def add_sensor(self, sensor: sim_structs.Sensor) -> None:
    """Adds a sensor to the robot."""

    sensor_type_map = {
      "gyro": mujoco.mjtSensor.mjSENS_GYRO,
      "upvector": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
      "velocimeter": mujoco.mjtSensor.mjSENS_VELOCIMETER,
      "framequat": mujoco.mjtSensor.mjSENS_FRAMEQUAT,
      "framepos": mujoco.mjtSensor.mjSENS_FRAMEPOS,
      "framelinvel": mujoco.mjtSensor.mjSENS_FRAMELINVEL,
      "frameangvel": mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
      "framezaxis": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
      "accelerometer": mujoco.mjtSensor.mjSENS_ACCELEROMETER,
    }

    sensor_object_type_map = {
      "site": mujoco.mjtObj.mjOBJ_SITE,
    }

    self._spec.add_sensor(
      type=sensor_type_map[sensor.sensor_type],
      objtype=sensor_object_type_map[sensor.object_type],
      name=sensor.name,
      objname=sensor.object_name,
    )

  # Properties.

  @property
  def spec(self) -> mujoco.MjSpec:
    """Returns the underlying mujoco.MjSpec."""
    return self._spec

  @property
  def assets(self) -> Dict[str, bytes]:
    """Returns the spec assets."""
    return self._spec.assets
