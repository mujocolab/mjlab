"""Utilities for GPU selection and management."""

import os
from typing import Literal


def select_gpus(
  gpus: list[int] | Literal["all"],
) -> tuple[list[int], int]:
  """Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.

  This function treats the `gpus` parameter as indices into the existing
  CUDA_VISIBLE_DEVICES environment variable. If CUDA_VISIBLE_DEVICES is not set,
  it defaults to all available GPUs.

  Args:
    gpus: Either a list of GPU indices (into CUDA_VISIBLE_DEVICES) or "all".

  Returns:
    A tuple of (selected_gpu_ids, num_gpus) where:
    - selected_gpu_ids: List of physical GPU IDs to use
    - num_gpus: Number of GPUs selected

  Examples:
    >>> os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    >>> select_gpus([0, 1])
    ([0, 1], 2)

    >>> os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
    >>> select_gpus([0])  # Selects physical GPU 1
    ([1], 1)

    >>> select_gpus("all")  # Selects all GPUs in CUDA_VISIBLE_DEVICES
    ([1, 3], 2)
  """
  # Get existing CUDA_VISIBLE_DEVICES or default to all GPUs.
  existing_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

  if existing_visible_devices is not None:
    # Parse existing CUDA_VISIBLE_DEVICES.
    available_gpus = [
      int(x.strip()) for x in existing_visible_devices.split(",") if x.strip()
    ]
  else:
    # If not set, default to all available GPUs.
    import torch.cuda

    available_gpus = list(range(torch.cuda.device_count()))

  # Map gpus indices to actual GPU IDs.
  if gpus == "all":
    selected_gpus = available_gpus
  else:
    # gpus are indices into available_gpus.
    selected_gpus = [available_gpus[i] for i in gpus]

  num_gpus = len(selected_gpus)

  return selected_gpus, num_gpus
