import os
import numpy as np
import random
import torch
import warp as wp


def seed_rng(seed: int, torch_deterministic: bool = False) -> None:
  np.random.seed(seed)
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)

  wp.rand_init(seed)

  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  if torch_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
  else:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
