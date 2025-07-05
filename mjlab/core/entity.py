import abc
from pathlib import Path

import mujoco


class Entity(abc.ABC):
  @property
  @abc.abstractmethod
  def spec(self) -> mujoco.MjSpec:
    """Returns the underlying mujoco.MjSpec."""
    raise NotImplementedError

  def compile(self) -> mujoco.MjModel:
    """Compiles the robot model into an MjModel."""
    return self.spec.compile()

  def write_xml(self, xml_path: Path) -> None:
    """Writes the robot model to an XML file."""
    with open(xml_path, "w") as f:
      f.write(self._spec.to_xml())
