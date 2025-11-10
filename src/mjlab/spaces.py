"""Lightweight space definitions for environment observations and actions.

This module provides minimal space representations to replace gymnasium.spaces,
focusing only on what mjlab needs (shape and dtype information for batching).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Space:
  """Base space class with shape and dtype information.

  This is a lightweight replacement for gymnasium.Space that only tracks
  shape and dtype for batching purposes.
  """

  shape: tuple[int, ...] = ()
  dtype: str = "float32"

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype})"


@dataclass
class Box(Space):
  """Continuous space with optional bounds.

  Represents a box in R^n with optional low/high bounds.

  Attributes:
      shape: Dimensions of the space
      low: Minimum values (scalar or per-element)
      high: Maximum values (scalar or per-element)
      dtype: Data type
  """

  low: float | tuple[float, ...] = -math.inf
  high: float | tuple[float, ...] = math.inf

  def __repr__(self) -> str:
    return (
      f"Box(shape={self.shape}, low={self.low}, high={self.high}, dtype={self.dtype})"
    )


@dataclass
class Dict(Space):
  """Dictionary space containing multiple named subspaces.

  Represents a dictionary of named spaces (like observation dicts with multiple keys).

  Attributes:
      spaces: Dictionary mapping space names to Space objects
  """

  def __init__(self, spaces: dict[str, Space] | None = None) -> None:
    super().__init__()
    self.spaces: dict[str, Space] = spaces if spaces is not None else {}

  def __setitem__(self, key: str, space: Space) -> None:
    """Add or update a subspace."""
    self.spaces[key] = space

  def __getitem__(self, key: str) -> Space:
    """Get a subspace."""
    return self.spaces[key]

  def __repr__(self) -> str:
    spaces_repr = ", ".join(f"{k}: {v}" for k, v in self.spaces.items())
    return f"Dict({{{spaces_repr}}})"

  def __iter__(self):
    """Iterate over space keys."""
    return iter(self.spaces)

  def keys(self):
    """Get all space keys."""
    return self.spaces.keys()

  def values(self):
    """Get all spaces."""
    return self.spaces.values()

  def items(self):
    """Get all space key-value pairs."""
    return self.spaces.items()


def batch_space(space: Space, batch_size: int) -> Space:
  """Create a batched version of a space.

  Prepends batch_size dimension to the space's shape.

  Args:
      space: The space to batch
      batch_size: Number of parallel environments

  Returns:
      New space with batched shape
  """
  if isinstance(space, Dict):
    # For Dict spaces, batch each subspace
    batched_dict = Dict()
    for key, subspace in space.spaces.items():
      batched_dict[key] = batch_space(subspace, batch_size)
    return batched_dict

  elif isinstance(space, Box):
    # For Box spaces, prepend batch dimension
    batched_shape = (batch_size,) + space.shape
    return Box(
      shape=batched_shape,
      low=space.low,
      high=space.high,
      dtype=space.dtype,
    )

  elif isinstance(space, Space):
    # For generic Space, prepend batch dimension
    batched_shape = (batch_size,) + space.shape
    return Space(shape=batched_shape, dtype=space.dtype)

  else:
    raise TypeError(f"Unknown space type: {type(space)}")
