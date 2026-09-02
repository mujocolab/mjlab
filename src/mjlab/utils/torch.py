import torch
from packaging.version import parse


def configure_torch_backends(allow_tf32: bool = True, deterministic: bool = False):
  """Configure PyTorch CUDA and cuDNN backends for performance/reproducibility.

  Args:
    allow_tf32: If True, use TF32 precision for faster computation on Ampere+ GPUs. If
      False, use standard IEEE FP32 precision.
    deterministic: If True, use deterministic algorithms (slower but reproducible).
      If False, allow cuDNN to benchmark and select fastest algorithms.

  Note:
    TF32 uses reduced precision (10-bit mantissa vs 23-bit for FP32) for internal
    matrix multiplications providing a speedup with minimal impact on accuracy.

    See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere for details.
  """
  torch_version = parse(torch.__version__.split("+")[0])  # Handle e.g., "2.9.0+cu118".
  if torch_version >= parse("2.9.0"):
    _configure_29(allow_tf32)
  else:
    _configure_pre29(allow_tf32)

  torch.backends.cudnn.benchmark = not deterministic  # Find fastest algorithms.
  torch.backends.cudnn.deterministic = deterministic  # Ensure reproducibility.


def _configure_29(allow_tf32: bool):
  """Configure PyTorch CUDA and cuDNN backends for PyTorch 2.9+."""
  # tf32 for performance, ieee for full FP32 accuracy.
  precision = "tf32" if allow_tf32 else "ieee"
  torch.backends.cuda.matmul.fp32_precision = precision
  torch.backends.cudnn.fp32_precision = precision  # type: ignore


def _configure_pre29(allow_tf32: bool):
  """Configure PyTorch CUDA and cuDNN backends for PyTorch <2.9."""
  torch.backends.cuda.matmul.allow_tf32 = allow_tf32
  torch.backends.cudnn.allow_tf32 = allow_tf32


def as_index(ids: torch.Tensor) -> slice | torch.Tensor:
  """Return ``ids`` as a slice when it is a contiguous ascending range.

  Indexing a tensor with a slice produces a view, whereas indexing with an index
  tensor launches a gather (or scatter) kernel. Element ids of an entity are
  usually contiguous, so converting them once at setup lets the hot path index
  simulation arrays without any kernel launches. Reading ``ids`` synchronizes with
  the device, so only call this at setup time.
  """
  if ids.ndim != 1 or ids.numel() == 0:
    return ids
  values = ids.tolist()
  start = values[0]
  if values == list(range(start, start + len(values))):
    return slice(start, start + len(values))
  return ids


def compose_index(
  sel: slice | torch.Tensor,
  ids: torch.Tensor,
  local: torch.Tensor | slice | None,
) -> slice | torch.Tensor:
  """Map local indices through ``ids`` (i.e. ``ids[local]``), preferring slices.

  ``sel`` is the :func:`as_index` form of ``ids``. When both ``sel`` and ``local``
  are slices the result is a slice too, so the caller can index without a gather.
  ``None`` selects everything.
  """
  if local is None:
    return sel
  if isinstance(local, slice) and isinstance(sel, slice):
    start, stop, step = local.indices(sel.stop - sel.start)
    if step == 1 and stop >= start:
      return slice(sel.start + start, sel.start + stop)
  return ids[local]
