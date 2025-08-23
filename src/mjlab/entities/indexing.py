from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class EntityIndexing:
  """Stores all indexing information for a single entity."""

  # IDs.
  root_body_id: int | None
  body_ids: torch.Tensor
  body_root_ids: torch.Tensor
  geom_ids: torch.Tensor
  geom_body_ids: torch.Tensor
  site_ids: torch.Tensor
  site_body_ids: torch.Tensor
  ctrl_ids: torch.Tensor

  root_body_iquat: torch.Tensor | None
  body_iquats: torch.Tensor

  # Addresses.
  sensor_adr: dict[str, torch.Tensor]
  joint_q_adr: torch.Tensor
  joint_v_adr: torch.Tensor
  free_joint_q_adr: torch.Tensor
  free_joint_v_adr: torch.Tensor

  # Mappings.
  body_local2global: dict[int, int]
  geom_local2global: dict[int, int]
  site_local2global: dict[int, int]
  actuator_local2global: dict[int, int]
  joint_local2global: dict[int, int]


@dataclass
class SceneIndexing:
  entities: dict[str, EntityIndexing] = field(default_factory=dict)
