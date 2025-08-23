from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ContactSensorData:
  net_forces_w: torch.Tensor | None = None
  net_forces_w_history: torch.Tensor | None = None
  last_air_time: torch.Tensor | None = None
  current_air_time: torch.Tensor | None = None
  last_contact_time: torch.Tensor | None = None
  current_contact_time: torch.Tensor | None = None
