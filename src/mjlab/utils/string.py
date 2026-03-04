import re
from typing import Any


def resolve_expr(
  pattern_map: dict[str, Any],
  names: tuple[str, ...],
  default_val: Any = None,
) -> tuple[Any, ...]:
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


def resolve_expr_with_widths(
  pattern_map: dict[str, Any],
  names: tuple[str, ...],
  widths: tuple[int, ...],
  default_vals: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
  """Resolve field values accounting for per-joint widths (ball=4, hinge/slide=1).

  Scalars for ball joints use the default (identity quaternion) for backward
  compatibility with {".*": 0.0} patterns.
  """
  patterns = [(re.compile(pat), val) for pat, val in pattern_map.items()]

  result: list[float] = []
  for name, width, default in zip(names, widths, default_vals, strict=True):
    matched = False
    for pat, val in patterns:
      if pat.match(name):
        if isinstance(val, (tuple, list)):
          if len(val) != width:
            raise ValueError(f"Joint '{name}' expects {width} values, got {len(val)}")
          result.extend(val)
          matched = True
        else:
          # Scalar value.
          if width == 1:
            result.append(val)
            matched = True
          else:
            # Scalar for ball joint - use default (identity quaternion).
            # This handles backward compatibility with {".*": 0.0} patterns.
            result.extend(default)
            matched = True
        break
    if not matched:
      result.extend(default)

  return tuple(result)


def filter_exp(
  exprs: list[str] | tuple[str, ...], names: tuple[str, ...]
) -> tuple[str, ...]:
  """Filter names based on regex patterns."""
  patterns = [re.compile(expr) for expr in exprs]
  return tuple(name for name in names if any(pat.match(name) for pat in patterns))


def resolve_field(
  field: Any, names: tuple[str, ...], default_val: Any = None
) -> tuple[Any, ...]:
  if isinstance(field, dict):
    return resolve_expr(field, names, default_val)
  else:
    return tuple([field] * len(names))
