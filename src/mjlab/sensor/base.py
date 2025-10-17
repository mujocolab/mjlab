from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import mujoco_warp as mjwarp
import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass
class SensorCfg(ABC):
  """Configuration for a sensor."""

  name: str

  @abstractmethod
  def build(self) -> Sensor:
    """Build sensor instance from this config."""
    raise NotImplementedError


class Sensor(ABC):
  """Base sensor interface."""

  def __init__(self, cfg: SensorCfg) -> None:
    self.cfg = cfg

  @abstractmethod
  def edit_spec(
    self,
    scene_spec: mujoco.MjSpec,
    entities: dict[str, Entity],
  ) -> None:
    """Edit the MuJoCo scene spec to add this sensor.

    Args:
      scene_spec: The MuJoCo scene specification to edit.
      entities: Dictionary of entities in the scene, keyed by name.
    """
    raise NotImplementedError

  @abstractmethod
  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    raise NotImplementedError

  @property
  @abstractmethod
  def data(self) -> Any:
    """Sensor data in whatever form makes sense for this sensor."""
    raise NotImplementedError

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset sensor state for specified environments.

    Base implementation does nothing. Override in subclasses that maintain state.
    """
    del env_ids  # Unused.

  def update(self, dt: float) -> None:
    """Update sensor state.

    Base implementation does nothing. Override in subclasses that need per-step updates.
    """
    del dt  # Unused.
