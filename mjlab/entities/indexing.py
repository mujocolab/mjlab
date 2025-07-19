from dataclasses import dataclass, field
import torch


@dataclass
class EntityIndexing:
  """Stores all indexing information for a single entity."""

  root_body_id: int | None
  body_ids: torch.Tensor
  body_root_ids: torch.Tensor
  geom_ids: torch.Tensor
  site_ids: torch.Tensor
  sensor_adr: dict[str, torch.Tensor]
  joint_q_adr: torch.Tensor
  joint_v_adr: torch.Tensor
  free_joint_q_adr: torch.Tensor
  free_joint_v_adr: torch.Tensor


@dataclass
class SceneIndexing:
  entities: dict[str, EntityIndexing] = field(default_factory=dict)
