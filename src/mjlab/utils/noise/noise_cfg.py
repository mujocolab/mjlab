from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import torch
from typing_extensions import override

from mjlab.utils.noise import noise_model

# Type alias for noise parameters: scalar or per-dimension values.
NoiseParam = float | tuple[float, ...]


@dataclass(kw_only=True)
class NoiseCfg(abc.ABC):
  """Base configuration for a noise term."""

  operation: Literal["add", "scale", "abs"] = "add"

  # Cache for converted tensors, keyed by device string.
  _tensor_cache: dict[str, dict[str, torch.Tensor]] = field(
    default_factory=dict, init=False, repr=False
  )

  def _get_cached_tensor(
    self, name: str, value: NoiseParam, device: torch.device
  ) -> torch.Tensor:
    """Get a cached tensor for the given parameter on the specified device."""
    device_key = str(device)
    if device_key not in self._tensor_cache:
      self._tensor_cache[device_key] = {}
    if name not in self._tensor_cache[device_key]:
      self._tensor_cache[device_key][name] = torch.tensor(value, device=device)
    return self._tensor_cache[device_key][name]

  @abc.abstractmethod
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    """Apply noise to the input data."""


@dataclass
class ConstantNoiseCfg(NoiseCfg):
  bias: NoiseParam = 0.0

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    bias = self._get_cached_tensor("bias", self.bias, data.device)

    if self.operation == "add":
      return data + bias
    elif self.operation == "scale":
      return data * bias
    elif self.operation == "abs":
      return torch.zeros_like(data) + bias
    else:
      raise ValueError(f"Unsupported noise operation: {self.operation}")


@dataclass
class UniformNoiseCfg(NoiseCfg):
  n_min: NoiseParam = -1.0
  n_max: NoiseParam = 1.0

  def __post_init__(self):
    if isinstance(self.n_min, float) and isinstance(self.n_max, float):
      if self.n_min >= self.n_max:
        raise ValueError(f"n_min ({self.n_min}) must be less than n_max ({self.n_max})")

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    n_min = self._get_cached_tensor("n_min", self.n_min, data.device)
    cache = self._tensor_cache[str(data.device)]
    n_range = cache.get("n_range")
    if n_range is None:
      n_max = self._get_cached_tensor("n_max", self.n_max, data.device)
      n_range = cache["n_range"] = n_max - n_min

    # Generate uniform noise in [0, 1) and scale to [n_min, n_max).
    if self.operation == "add":
      # data + rand * range + n_min, fused into two launches after the draw.
      return torch.addcmul(data + n_min, torch.rand_like(data), n_range)
    noise = torch.rand_like(data) * n_range + n_min

    if self.operation == "scale":
      return data * noise
    elif self.operation == "abs":
      return noise
    else:
      raise ValueError(f"Unsupported noise operation: {self.operation}")


@dataclass
class GaussianNoiseCfg(NoiseCfg):
  mean: NoiseParam = 0.0
  std: NoiseParam = 1.0

  def __post_init__(self):
    if isinstance(self.std, float) and self.std <= 0:
      raise ValueError(f"std ({self.std}) must be positive")

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    mean = self._get_cached_tensor("mean", self.mean, data.device)
    std = self._get_cached_tensor("std", self.std, data.device)

    # Generate standard normal noise and scale.
    noise = mean + std * torch.randn_like(data)

    if self.operation == "add":
      return data + noise
    elif self.operation == "scale":
      return data * noise
    elif self.operation == "abs":
      return noise
    else:
      raise ValueError(f"Unsupported noise operation: {self.operation}")


##
# Noise models.
##


@dataclass(kw_only=True)
class NoiseModelCfg:
  """Configuration for a noise model."""

  noise_cfg: NoiseCfg

  class_type: ClassVar[type[noise_model.NoiseModel]] = noise_model.NoiseModel

  def __init_subclass__(cls, class_type: type[noise_model.NoiseModel]):
    cls.class_type = class_type


@dataclass(kw_only=True)
class NoiseModelWithAdditiveBiasCfg(
  NoiseModelCfg, class_type=noise_model.NoiseModelWithAdditiveBias
):
  """Configuration for an additive Gaussian noise with bias model."""

  bias_noise_cfg: NoiseCfg | None = None
  sample_bias_per_component: bool = True

  def __post_init__(self):
    if self.bias_noise_cfg is None:
      raise ValueError(
        "bias_noise_cfg must be specified for NoiseModelWithAdditiveBiasCfg"
      )
