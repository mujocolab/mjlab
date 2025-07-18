from pathlib import Path

import mujoco
import mujoco_warp as mjwarp


class Entity:
  def __init__(self, spec: mujoco.MjSpec):
    self._spec = spec

    self._geoms: tuple[mujoco.MjsGeom, ...] = self._spec.geoms
    self._bodies: tuple[mujoco.MjsBody, ...] = self._spec.bodies
    self._joints: tuple[mujoco.MjsJoint, ...] = self._spec.joints
    self._actuators: tuple[mujoco.MjsActuator, ...] = self._spec.actuators
    self._sensors: tuple[mujoco.MjsSensor, ...] = self._spec.sensors

  @property
  def spec(self) -> mujoco.MjSpec:
    """Returns the underlying mujoco.MjSpec."""
    return self._spec

  def compile(self) -> mujoco.MjModel:
    """Compiles the robot model into an MjModel."""
    return self.spec.compile()

  def write_xml(self, xml_path: Path) -> None:
    """Writes the robot model to an XML file."""
    with open(xml_path, "w") as f:
      f.write(self.spec.to_xml())

  def update(self, dt: float, data: mjwarp.Data) -> None:
    pass
