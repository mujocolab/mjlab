import os
import random

import numpy as np
import torch
import warp as wp


def seed_rng(seed: int, torch_deterministic: bool = False) -> None:
  """Seed all random number generators for reproducibility.

  Note: MuJoCo Warp is not fully deterministic yet.
  See: https://github.com/google-deepmind/mujoco_warp/issues/562
  """
  os.environ["PYTHONHASHSEED"] = str(seed)

  random.seed(seed)
  np.random.seed(seed)

  wp.rand_init(wp.int32(seed))

  # Ref: https://docs.pytorch.org/docs/stable/notes/randomness.html
  torch.manual_seed(seed)  # Seed RNG for all devices.
  # Use deterministic algorithms when possible.
  if torch_deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True)
