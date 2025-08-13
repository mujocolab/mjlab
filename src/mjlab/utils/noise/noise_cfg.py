from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import torch

from mjlab.utils.noise import noise_model

FuncType = Callable[[torch.Tensor, "NoiseCfg"], torch.Tensor]


@dataclass
class NoiseCfg:
  """Base configuration for a noise term."""

  func: FuncType = field(default_factory=lambda: None)
  operation: Literal["add", "scale", "abs"] = "add"

  def __post_init__(self):
    if self.func is None:
      raise ValueError("func must be specified for NoiseCfg")


@dataclass
class ConstantNoiseCfg(NoiseCfg):
  func: FuncType = field(default_factory=lambda: noise_model.constant_noise)
  bias: torch.Tensor | float = 0.0


@dataclass
class UniformNoiseCfg(NoiseCfg):
  func: FuncType = field(default_factory=lambda: noise_model.uniform_noise)
  n_min: torch.Tensor | float = -1.0
  n_max: torch.Tensor | float = 1.0

  def __post_init__(self):
    super().__post_init__()
    if isinstance(self.n_min, (int, float)) and isinstance(self.n_max, (int, float)):
      if self.n_min >= self.n_max:
        raise ValueError(f"n_min ({self.n_min}) must be less than n_max ({self.n_max})")


@dataclass
class GaussianNoiseCfg(NoiseCfg):
  func: FuncType = field(default_factory=lambda: noise_model.gaussian_noise)
  mean: torch.Tensor | float = 0.0
  std: torch.Tensor | float = 1.0

  def __post_init__(self):
    super().__post_init__()
    if isinstance(self.std, (int, float)) and self.std <= 0:
      raise ValueError(f"std ({self.std}) must be positive")


##
# Noise models.
##


@dataclass(kw_only=True)
class NoiseModelCfg:
  """Configuration for a noise model."""

  class_type: type = field(default_factory=lambda: noise_model.NoiseModel)
  noise_cfg: NoiseCfg = field(default_factory=lambda: None)
  func: Callable[[torch.Tensor], torch.Tensor] | None = None

  def __post_init__(self):
    if self.noise_cfg is None:
      raise ValueError("noise_cfg must be specified for NoiseModelCfg")

    # Validate class type.
    if not issubclass(self.class_type, noise_model.NoiseModel):
      raise ValueError(
        f"class_type must be a subclass of NoiseModel, got {self.class_type}"
      )


@dataclass(kw_only=True)
class NoiseModelWithAdditiveBiasCfg(NoiseModelCfg):
  """Configuration for an additive Gaussian noise with bias model."""

  class_type: type = field(
    default_factory=lambda: noise_model.NoiseModelWithAdditiveBias
  )
  bias_noise_cfg: NoiseCfg = field(default_factory=lambda: None)
  sample_bias_per_component: bool = True

  def __post_init__(self):
    super().__post_init__()
    if self.bias_noise_cfg is None:
      raise ValueError(
        "bias_noise_cfg must be specified for NoiseModelWithAdditiveBiasCfg"
      )

    # Validate class_type
    if not issubclass(self.class_type, noise_model.NoiseModelWithAdditiveBias):
      raise ValueError(
        f"class_type must be a subclass of NoiseModelWithAdditiveBias, got {self.class_type}"
      )
