"""Velocity task configurations for different robots."""

from mjlab.tasks.velocity.config.g1 import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from mjlab.tasks.velocity.config.go1 import (
  unitree_go1_flat_env_cfg,
  unitree_go1_rough_env_cfg,
)
from mjlab.tasks.velocity.config.t1 import (
  booster_t1_flat_env_cfg,
  booster_t1_rough_env_cfg,
)

__all__ = [
  "unitree_g1_flat_env_cfg",
  "unitree_g1_rough_env_cfg",
  "unitree_go1_flat_env_cfg",
  "unitree_go1_rough_env_cfg",
  "booster_t1_flat_env_cfg",
  "booster_t1_rough_env_cfg",
]
