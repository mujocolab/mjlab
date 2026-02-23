"""Extensible Operation and Distribution types for domain randomization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import (
  sample_gaussian,
  sample_log_uniform,
  sample_uniform,
)


@dataclass(frozen=True)
class Operation:
  """A domain randomization operation.

  Built-in operations (``abs``, ``scale``, ``add``) are provided as module-level
  instances. Users can create custom operations by instantiating this class directly
  and passing them wherever an operation string is accepted.

  The DR engine samples random values for each target axis, then writes the final
  result as::

      model_field[envs, entities] = operation.combine(base, random)

  where ``base`` is either the compile-time default or the current model value,
  depending on ``uses_defaults``.

  Args:
    name: Human-readable identifier (used in error messages).
    initialize: Called once per randomization call to create the result tensor that
      gets filled axis by axis with sampled values. Receives the base values and
      returns a tensor of the same shape. For example, ``scale`` starts from ones so
      that unsampled axes multiply by 1 (no change), while ``add`` starts from zeros.
    combine: Called after all axes have been sampled. Receives the base values and the
      random values and returns the final tensor to write into the model field. For
      example, ``scale`` returns ``base * random`` and ``add`` returns
      ``base + random``.
    uses_defaults: When ``True``, the engine reads the original default field values
      captured at model compile time (preventing accumulation across repeated calls).
      When ``False``, the engine reads the current model values instead.
  """

  name: str
  initialize: Callable[[torch.Tensor], torch.Tensor]
  combine: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
  uses_defaults: bool

  def __repr__(self) -> str:
    return f"Operation({self.name!r})"


@dataclass(frozen=True)
class Distribution:
  """A domain randomization sampling distribution.

  Built-in distributions (``uniform``, ``log_uniform``, ``gaussian``) are provided as
  module-level instances. Users can create custom distributions by instantiating this
  class directly.

  Args:
    name: Human-readable identifier (used in error messages).
    sample: Called for each target axis. Receives ``(lower, upper, shape, device)`` and
      returns a tensor of random samples with the given shape on the given device.
      ``lower`` and ``upper`` come from the ``ranges`` parameter passed to the DR
      function.
  """

  name: str
  sample: Callable[[torch.Tensor, torch.Tensor, tuple[int, ...], str], torch.Tensor]

  def __repr__(self) -> str:
    return f"Distribution({self.name!r})"


# Built-in operations.

abs = Operation(
  name="abs",
  initialize=torch.Tensor.clone,
  combine=lambda base, random: random,
  uses_defaults=False,
)
scale = Operation(
  name="scale",
  initialize=torch.ones_like,
  combine=torch.mul,
  uses_defaults=True,
)
add = Operation(
  name="add",
  initialize=torch.zeros_like,
  combine=torch.add,
  uses_defaults=True,
)

# Built-in distributions.

uniform = Distribution(
  name="uniform",
  sample=lambda lo, hi, shape, device: sample_uniform(lo, hi, shape, device=device),
)
log_uniform = Distribution(
  name="log_uniform",
  sample=lambda lo, hi, shape, device: sample_log_uniform(lo, hi, shape, device=device),
)
gaussian = Distribution(
  name="gaussian",
  sample=lambda lo, hi, shape, device: sample_gaussian(lo, hi, shape, device=device),
)

# Resolution helpers.

_OPERATION_REGISTRY: dict[str, Operation] = {
  "abs": abs,
  "scale": scale,
  "add": add,
}

_DISTRIBUTION_REGISTRY: dict[str, Distribution] = {
  "uniform": uniform,
  "log_uniform": log_uniform,
  "gaussian": gaussian,
}


def resolve_operation(op: Operation | str) -> Operation:
  """Resolve a string or Operation instance to an Operation.

  Raises:
    ValueError: If the string does not match a built-in operation name.
  """
  if isinstance(op, Operation):
    return op
  try:
    return _OPERATION_REGISTRY[op]
  except KeyError:
    raise ValueError(
      f"Unknown operation: {op!r}. "
      f"Built-in operations: {list(_OPERATION_REGISTRY)}. "
      f"Pass an Operation instance for custom operations."
    ) from None


def resolve_distribution(dist: Distribution | str) -> Distribution:
  """Resolve a string or Distribution instance to a Distribution.

  Raises:
    ValueError: If the string does not match a built-in distribution name.
  """
  if isinstance(dist, Distribution):
    return dist
  try:
    return _DISTRIBUTION_REGISTRY[dist]
  except KeyError:
    raise ValueError(
      f"Unknown distribution: {dist!r}. "
      f"Built-in distributions: {list(_DISTRIBUTION_REGISTRY)}. "
      f"Pass a Distribution instance for custom distributions."
    ) from None
