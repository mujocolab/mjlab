from abc import ABC, abstractmethod
import mujoco


class SpecEditor(ABC):
  """Base class for anything that can modify an MjSpec."""

  @abstractmethod
  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    raise NotImplementedError
