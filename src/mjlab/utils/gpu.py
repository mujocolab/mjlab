"""Utilities for GPU selection and management."""

import os
from typing import Literal

GpuId = int | str


def select_gpus(
  gpu_ids: list[int] | Literal["all"] | None,
) -> tuple[list[GpuId] | None, int]:
  """Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.

  This function treats the `gpu_ids` parameter as indices into the existing
  CUDA_VISIBLE_DEVICES environment variable. If CUDA_VISIBLE_DEVICES is not set,
  it defaults to all available GPUs.

  Args:
    gpu_ids: Either a list of GPU indices (into CUDA_VISIBLE_DEVICES), "all", or None
    for CPU.

  Returns:
    A tuple of (selected_gpu_ids, num_gpus) where:
    - selected_gpu_ids: List of physical GPU IDs (int for numeric, str for MIG
      UUIDs), or None for CPU mode
    - num_gpus: Number of GPUs selected (0 for CPU mode)

  Examples:
    >>> os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    >>> select_gpus([0, 1])
    ([0, 1], 2)

    >>> os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
    >>> select_gpus([0])  # Selects physical GPU 1
    ([1], 1)

    >>> select_gpus("all")  # Selects all GPUs in CUDA_VISIBLE_DEVICES
    ([1, 3], 2)

    >>> select_gpus(None)  # CPU mode
    (None, 0)

    >>> os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Empty CUDA_VISIBLE_DEVICES
    >>> select_gpus([0])
    (None, 0)
  """
  # CPU mode requested explicitly.
  if gpu_ids is None:
    return None, 0

  # Get existing CUDA_VISIBLE_DEVICES or default to all GPUs.
  existing_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

  if existing_visible_devices is not None:
    # Parse existing CUDA_VISIBLE_DEVICES.
    # Use int for numeric IDs, keep as string for MIG UUIDs.
    available_gpus: list[GpuId] = [
      int(x.strip()) if x.strip().isdigit() else x.strip()
      for x in existing_visible_devices.split(",")
      if x.strip()
    ]
    # Empty CUDA_VISIBLE_DEVICES means CPU mode.
    if not available_gpus:
      return None, 0
  else:
    # If not set, default to all available GPUs.
    import torch.cuda

    available_gpus: list[GpuId] = list(range(torch.cuda.device_count()))

  # Map gpu_ids indices to actual GPU IDs.
  selected: list[GpuId]
  if gpu_ids == "all":
    selected = available_gpus
  else:
    # gpu_ids are indices into available_gpus.
    selected = [available_gpus[i] for i in gpu_ids]

  num_gpus = len(selected)

  return selected, num_gpus
