from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
  print("hi")
