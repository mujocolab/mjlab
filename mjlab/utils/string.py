import re
from typing import Dict, List, Pattern, Any, Tuple


def resolve_expr(
  pattern_map: Dict[str, Any],
  names: List[str],
) -> List[Any]:
  # Pre-compile patterns in insertion order.
  compiled: List[Tuple[Pattern[str], Any]] = [
    (re.compile(pat), val) for pat, val in pattern_map.items()
  ]

  default_val = 0.0
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
