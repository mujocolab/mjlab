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

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from typing_extensions import override

from mjlab.utils.noise import noise_model


def _ensure_tensor_device(
  value: torch.Tensor | float, device: torch.device
) -> torch.Tensor | float:
  """Ensure tensor is on the correct device, leave scalars unchanged."""
  if isinstance(value, torch.Tensor):
    return value.to(device=device)
  return value


@dataclass(kw_only=True)
class NoiseCfg(abc.ABC):
  """Base configuration for a noise term."""

  operation: Literal["add", "scale", "abs"] = "add"

  @abc.abstractmethod
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    """Apply noise to the input data."""


@dataclass
class ConstantNoiseCfg(NoiseCfg):
  bias: torch.Tensor | float = 0.0

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    self.bias = _ensure_tensor_device(self.bias, data.device)

    if self.operation == "add":
      return data + self.bias
    elif self.operation == "scale":
      return data * self.bias
    elif self.operation == "abs":
      return torch.zeros_like(data) + self.bias
    else:
      raise ValueError(f"Unsupported noise operation: {self.operation}")


@dataclass
class UniformNoiseCfg(NoiseCfg):
  n_min: torch.Tensor | float = -1.0
  n_max: torch.Tensor | float = 1.0

  def __post_init__(self):
    if isinstance(self.n_min, (int, float)) and isinstance(self.n_max, (int, float)):
      if self.n_min >= self.n_max:
        raise ValueError(f"n_min ({self.n_min}) must be less than n_max ({self.n_max})")

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    self.n_min = _ensure_tensor_device(self.n_min, data.device)
    self.n_max = _ensure_tensor_device(self.n_max, data.device)

    # Generate uniform noise in [0, 1) and scale to [n_min, n_max).
    noise = torch.rand_like(data) * (self.n_max - self.n_min) + self.n_min

    if self.operation == "add":
      return data + noise
    elif self.operation == "scale":
      return data * noise
    elif self.operation == "abs":
      return noise
    else:
      raise ValueError(f"Unsupported noise operation: {self.operation}")


@dataclass
class GaussianNoiseCfg(NoiseCfg):
  mean: torch.Tensor | float = 0.0
  std: torch.Tensor | float = 1.0

  def __post_init__(self):
    if isinstance(self.std, (int, float)) and self.std <= 0:
      raise ValueError(f"std ({self.std}) must be positive")

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    self.mean = _ensure_tensor_device(self.mean, data.device)
    self.std = _ensure_tensor_device(self.std, data.device)

    # Generate standard normal noise and scale.
    noise = self.mean + self.std * torch.randn_like(data)

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
