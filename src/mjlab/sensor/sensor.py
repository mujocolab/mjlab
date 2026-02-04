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
  from mjlab.viewer.debug_visualizer import DebugVisualizer


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
  """Base sensor interface with typed data and per-step caching.

  Type parameter T specifies the type of data returned by the sensor. For example:
  - Sensor[torch.Tensor] for sensors returning raw tensors
  - Sensor[ContactData] for sensors returning structured contact data

  Subclasses should not forget to:
  - Call `super().__init__()` in their `__init__` method
  - If overriding `reset()` or `update()`, call `super()` FIRST to invalidate cache
  """

  def __init__(self) -> None:
    self._cached_data: T | None = None
    self._cache_valid: bool = False

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
  def data(self) -> T:
    """Get the current sensor data, using cached value if available.

    This property returns the sensor's current data in its specific type.
    The data type is specified by the type parameter T. The data is cached
    per-step and recomputed only when the cache is invalidated (after
    `reset()` or `update()` is called).

    Returns:
      The sensor data in the format specified by type parameter T.
    """
    if not self._cache_valid:
      self._cached_data = self._compute_data()
      self._cache_valid = True
    assert self._cached_data is not None
    return self._cached_data

  @abstractmethod
  def _compute_data(self) -> T:
    """Compute and return the sensor data.

    Subclasses must implement this method to compute the sensor's data.
    This is called by the `data` property when the cache is invalid.

    Returns:
      The computed sensor data.
    """
    raise NotImplementedError

  def _invalidate_cache(self) -> None:
    """Invalidate the cached data, forcing recomputation on next access."""
    self._cache_valid = False

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset sensor state for specified environments.

    Invalidates the data cache. Override in subclasses that maintain
    internal state, but call `super().reset(env_ids)` FIRST.

    Args:
      env_ids: Environment indices to reset. If None, reset all environments.
    """
    del env_ids  # Unused.
    self._invalidate_cache()

  def update(self, dt: float) -> None:
    """Update sensor state after a simulation step.

    Invalidates the data cache. Override in subclasses that need
    per-step updates, but call `super().update(dt)` FIRST.

    Args:
      dt: Time step in seconds.
    """
    del dt  # Unused.
    self._invalidate_cache()

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Visualize sensor data for debugging.

    Base implementation does nothing. Override in subclasses that support
    debug visualization.

    Args:
      visualizer: The debug visualizer to draw to.
    """
    del visualizer  # Unused.
