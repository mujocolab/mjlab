import re
from typing import TypeVar

T = TypeVar("T", int, float)


def resolve_expr(
  pattern_map: dict[str, T],
  names: tuple[str, ...],
  default_val: T = 0.0,
) -> tuple[T, ...]:
  """Resolve a field value (scalar or dict) to a tuple of values matched by patterns."""
  patterns = [(re.compile(pat), val) for pat, val in pattern_map.items()]

  result = []
  for name in names:
    for pat, val in patterns:
      if pat.match(name):
        result.append(val)
        break
    else:
      result.append(default_val)
  return tuple(result)


def filter_exp(
  exprs: list[str] | tuple[str, ...], names: tuple[str, ...]
) -> tuple[str, ...]:
  """Filter names based on regex patterns."""
  patterns = [re.compile(expr) for expr in exprs]
  return tuple(name for name in names if any(pat.match(name) for pat in patterns))


def resolve_field(
  field: T | dict[str, T], names: tuple[str, ...], default_val: T = 0
) -> tuple[T, ...]:
  result = (
    resolve_expr(field, names, default_val)
    if isinstance(field, dict)
    else [field] * len(names)
  )
  return tuple(result)
