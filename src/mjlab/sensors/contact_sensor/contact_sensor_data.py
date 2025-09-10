from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ContactSensorData:
  net_forces_w: torch.Tensor
  net_forces_w_history: torch.Tensor
  last_air_time: torch.Tensor | None
  current_air_time: torch.Tensor | None
  last_contact_time: torch.Tensor | None
  current_contact_time: torch.Tensor | None
