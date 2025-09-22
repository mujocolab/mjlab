import torch


def configure_torch_backends(allow_tf32: bool = True, deterministic: bool = False):
  """Configure PyTorch CUDA and cuDNN backends for performance/reproducibility."""
  # https://docs.pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
  torch.backends.cuda.matmul.allow_tf32 = allow_tf32
  torch.backends.cudnn.allow_tf32 = allow_tf32

  torch.backends.cudnn.deterministic = deterministic
  torch.backends.cudnn.benchmark = not deterministic
