from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict, Type


def get_terms(instance: Any, term_type: Type) -> Dict[str, Any]:
  if not is_dataclass(instance):
    raise TypeError(
      f"get_terms() expects a dataclass instance, got {type(instance).__name__}"
    )

  return {
    f.name: getattr(instance, f.name)
    for f in fields(instance)
    if isinstance(getattr(instance, f.name), term_type)
  }
