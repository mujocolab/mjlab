"""Base sensor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity


T = TypeVar("T")


@dataclass
class SensorCfg(ABC):
  """Base configuration for a sensor."""

  name: str

  @abstractmethod
  def build(self) -> Sensor[Any]:
    """Build sensor instance from this config."""
    raise NotImplementedError


class Sensor(ABC, Generic[T]):
  """Base sensor interface with typed data.

  Type parameter T specifies the type of data returned by the sensor. For example:
  - Sensor[torch.Tensor] for sensors returning raw tensors
  - Sensor[ContactData] for sensors returning structured contact data
  """

  @abstractmethod
  def edit_spec(
    self,
    scene_spec: mujoco.MjSpec,
    entities: dict[str, Entity],
  ) -> None:
    """Edit the scene spec to add this sensor.

    This is called during scene construction to add sensor elements
    to the MjSpec.

    Args:
      scene_spec: The scene MjSpec to edit.
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
    """Initialize the sensor after model compilation.

    This is called after the MjSpec is compiled into an MjModel and the simulation
    is ready to run. Use this to cache sensor indices, allocate buffers, etc.

    Args:
      mj_model: The compiled MuJoCo model.
      model: The mjwarp model wrapper.
      data: The mjwarp data arrays.
      device: Device for tensor operations (e.g., "cuda", "cpu").
    """
    raise NotImplementedError

  @property
  @abstractmethod
  def data(self) -> T:
    """Get the current sensor data.

    This property returns the sensor's current data in its specific type.
    The data type is specified by the type parameter T.

    Returns:
      The sensor data in the format specified by type parameter T.
    """
    raise NotImplementedError

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset sensor state for specified environments.

    Base implementation does nothing. Override in subclasses that maintain
    internal state.

    Args:
      env_ids: Environment indices to reset. If None, reset all environments.
    """
    del env_ids  # Unused.

  def update(self, dt: float) -> None:
    """Update sensor state after a simulation step.

    Base implementation does nothing. Override in subclasses that need
    per-step updates.

    Args:
      dt: Time step in seconds.
    """
    del dt  # Unused.
