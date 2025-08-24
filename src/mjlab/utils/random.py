import os
import random

import numpy as np
import torch
import warp as wp


# Reference: https://github.com/isaac-sim/IsaacSim/blob/1791e14867bc9f269cf6c430a2bff5a1fb43ca66/source/extensions/isaacsim.core.utils/python/impl/torch/maths.py#L109-L135
def seed_rng(seed: int, torch_deterministic: bool = False) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  wp.rand_init(seed)  # type: ignore

  if torch_deterministic:
    # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
  else:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
