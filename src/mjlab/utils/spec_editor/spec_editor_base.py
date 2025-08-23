from __future__ import annotations

import abc

import mujoco


class SpecEditor(abc.ABC):
  @abc.abstractmethod
  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    raise NotImplementedError
