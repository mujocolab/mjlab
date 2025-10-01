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

import enum
from dataclasses import dataclass


@dataclass
class ViewerConfig:
  lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
  distance: float = 5.0
  elevation: float = -45.0
  azimuth: float = 90.0

  class OriginType(enum.Enum):
    """The frame in which the camera position and target are defined."""

    WORLD = enum.auto()
    """The origin of the world."""
    ASSET_ROOT = enum.auto()
    """The center of the asset defined by asset_name."""
    ASSET_BODY = enum.auto()
    """The center of the body defined by body_name in asset defined by asset_name."""

  origin_type: OriginType = OriginType.WORLD
  asset_name: str | None = None
  body_name: str | None = None
