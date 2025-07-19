from pathlib import Path

import mujoco
import mujoco_warp as mjwarp

from mjlab.entities.indexing import EntityIndexing


class Entity:
  def __init__(self, spec: mujoco.MjSpec):
    self._spec = spec

    self._give_names_to_missing_elems()
    # TODO: Add more sanity checking and processing.

  def _give_names_to_missing_elems(self):
    def _incremental_rename(elem_list, elem_type: str):
      counter = 0
      for elem in elem_list:
        if not elem.name:
          elem.name = f"{elem_type}_{counter}"
          counter += 1

    _incremental_rename(self._spec.bodies, "body")
    _incremental_rename(self._spec.geoms, "geom")
    _incremental_rename(self._spec.sites, "site")
    _incremental_rename(self._spec.sensors, "sensor")

  # Attributes.

  @property
  def spec(self) -> mujoco.MjSpec:
    """Returns the underlying mujoco.MjSpec."""
    return self._spec

  # Methods.

  def compile(self) -> mujoco.MjModel:
    """Compiles the spec into an MjModel."""
    return self.spec.compile()

  def write_xml(self, xml_path: Path) -> None:
    """Writes the spec to an XML file."""
    with open(xml_path, "w") as f:
      f.write(self.spec.to_xml())

  def initialize(
    self, indexing: EntityIndexing, data: mjwarp.Data, device: str
  ) -> None:
    pass

  def update(self, dt: float) -> None:
    pass

  def reset(self):
    pass
