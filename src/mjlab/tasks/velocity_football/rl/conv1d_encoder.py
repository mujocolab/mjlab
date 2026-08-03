"""One-dimensional convolutional encoder for observation histories."""

from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {
  "elu": nn.ELU,
  "relu": nn.ReLU,
  "tanh": nn.Tanh,
  "leaky_relu": nn.LeakyReLU,
}

_POOLS = {
  "avg": nn.AdaptiveAvgPool1d,
  "max": nn.AdaptiveMaxPool1d,
}


class Conv1dEncoder(nn.Module):
  """Encode a ``(batch, features, history)`` sequence into one latent vector."""

  def __init__(
    self,
    input_channels: int,
    output_channels: tuple[int, ...] = (256, 128, 64),
    kernel_size: int = 3,
    activation: str = "elu",
    global_pool: str = "avg",
    dilations: tuple[int, ...] | None = None,
    causal: bool = False,
    output_mode: str = "global_pool",
  ) -> None:
    super().__init__()
    if not output_channels:
      raise ValueError("output_channels must not be empty")
    if activation not in _ACTIVATIONS:
      raise ValueError(f"Unsupported activation: {activation!r}")
    if global_pool not in _POOLS:
      raise ValueError(f"Unsupported global pool: {global_pool!r}")
    if kernel_size <= 0:
      raise ValueError("kernel_size must be positive")
    if output_mode not in {"global_pool", "last"}:
      raise ValueError(f"Unsupported output mode: {output_mode!r}")
    if dilations is None:
      dilations = (1,) * len(output_channels)
    if len(dilations) != len(output_channels):
      raise ValueError("dilations must match output_channels")
    if any(dilation <= 0 for dilation in dilations):
      raise ValueError("dilations must be positive")

    layers: list[nn.Module] = []
    in_channels = input_channels
    for out_channels, dilation in zip(output_channels, dilations, strict=True):
      if causal:
        layers.append(nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0.0))
      layers.extend(
        (
          nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0 if causal else (kernel_size // 2) * dilation,
          ),
          _ACTIVATIONS[activation](),
        )
      )
      in_channels = out_channels
    if output_mode == "global_pool":
      layers.extend((_POOLS[global_pool](1), nn.Flatten(start_dim=1)))
    self.net = nn.Sequential(*layers)
    self.output_mode = output_mode
    self._output_dim = output_channels[-1]

  @property
  def output_dim(self) -> int:
    return self._output_dim

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    encoded = self.net(history)
    if self.output_mode == "last":
      return encoded[:, :, -1]
    return encoded
