import re
from typing import Any, Dict, List, Pattern, Tuple


def resolve_expr(
  pattern_map: Dict[str, Any],
  names: List[str],
  default_val: Any = 0.0,
) -> List[Any]:
  # Pre-compile patterns in insertion order.
  compiled: List[Tuple[Pattern[str], Any]] = [
    (re.compile(pat), val) for pat, val in pattern_map.items()
  ]

  result: List[Any] = []
  for name in names:
    for pat, val in compiled:
      if pat.match(name):
        result.append(val)
        break
    else:
      result.append(default_val)
  return result


def filter_exp(exprs: List[str], names: List[str]) -> List[str]:
  patterns: List[Pattern] = [re.compile(expr) for expr in exprs]
  return [name for name in names if any(pat.match(name) for pat in patterns)]


def resolve_field(field: int | dict[str, int], names: list[str], default_val: Any = 0):
  return (
    resolve_expr(field, names, default_val)
    if isinstance(field, dict)
    else [field] * len(names)
  )
