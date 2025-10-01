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
