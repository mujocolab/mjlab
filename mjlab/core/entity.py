import abc
from pathlib import Path
from typing import Dict

import mujoco


class Entity(abc.ABC):
  """A thin wrapper around MjSpec with some convenience methods."""

  def __init__(self, spec: mujoco.MjSpec):
    self._spec = spec
    self.post_init()

  def post_init(self):
    """Callback executed after the constructor."""
    pass

  def compile(self) -> mujoco.MjModel:
    """Compiles the robot model into an MjModel."""
    return self.spec.compile()

  def write_xml(self, xml_path: Path) -> None:
    """Writes the robot model to an XML file."""
    with open(xml_path, "w") as f:
      f.write(self._spec.to_xml())

  @property
  def spec(self) -> mujoco.MjSpec:
    """Returns the underlying mujoco.MjSpec."""
    return self._spec

  @property
  def assets(self) -> Dict[str, bytes]:
    """Returns the spec assets."""
    return self._spec.assets
