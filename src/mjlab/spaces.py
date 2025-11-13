"""Lightweight space definitions for environment observations and actions.

This module provides minimal space representations to replace gymnasium.spaces,
focusing only on what mjlab needs (shape and dtype information for batching).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from typing_extensions import assert_never


@dataclass
class Space:
  """Base space class with shape and dtype information."""

  shape: tuple[int, ...] = ()
  dtype: Literal["float32", "int32", "int64", "uint8"] = "float32"


@dataclass
class Box(Space):
  """Continuous space with optional bounds."""

  low: float | tuple[float, ...] = -math.inf
  high: float | tuple[float, ...] = math.inf


@dataclass
class Dict(Space):
  """Dictionary space containing multiple named subspaces."""

  spaces: dict[str, Space] = field(default_factory=dict)


T = TypeVar("T", Dict, Box, Space)


def batch_space(space: T, batch_size: int) -> T:
  """Create a batched version of a space.

  Prepends batch_size dimension to the space's shape.

  Args:
      space: The space to batch
      batch_size: Number of parallel environments

  Returns:
      New space with batched shape
  """
  if isinstance(space, Dict):
    # For Dict spaces, batch each subspace.
    return Dict(
      spaces={
        key: batch_space(subspace, batch_size) for key, subspace in space.spaces.items()
      },
      shape=(batch_size,),
      dtype=space.dtype,
    )

  elif isinstance(space, Box):
    # For Box spaces, prepend batch dimension.
    batched_shape = (batch_size,) + space.shape
    return Box(
      shape=batched_shape,
      low=space.low,
      high=space.high,
      dtype=space.dtype,
    )

  elif isinstance(space, Space):
    # For generic Space, prepend batch dimension.
    batched_shape = (batch_size,) + space.shape
    return Space(shape=batched_shape, dtype=space.dtype)

  assert_never(space)
