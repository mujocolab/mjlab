# Copyright 2025 The MjLab Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
