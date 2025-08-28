from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.utils.noise import noise_cfg


class NoiseModel:
  """Base class for noise models."""

  def __init__(
    self, noise_model_cfg: noise_cfg.NoiseModelCfg, num_envs: int, device: str
  ):
    self._noise_model_cfg = noise_model_cfg
    self._num_envs = num_envs
    self._device = device

    # Validate configuration.
    if not hasattr(noise_model_cfg, "noise_cfg") or noise_model_cfg.noise_cfg is None:
      raise ValueError("NoiseModelCfg must have a valid noise_cfg")

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    """Reset noise model state. Override in subclasses if needed."""

  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    """Apply noise to input data."""
    return self._noise_model_cfg.noise_cfg.apply(data)


class NoiseModelWithAdditiveBias(NoiseModel):
  """Noise model with additional additive bias that is constant for the duration
  of the entire episode."""

  def __init__(
    self,
    noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg,
    num_envs: int,
    device: str,
  ):
    super().__init__(noise_model_cfg, num_envs, device)

    # Validate bias configuration.
    if (
      not hasattr(noise_model_cfg, "bias_noise_cfg")
      or noise_model_cfg.bias_noise_cfg is None
    ):
      raise ValueError("NoiseModelWithAdditiveBiasCfg must have a valid bias_noise_cfg")

    self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
    self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

    # Initialize bias tensor.
    self._bias = torch.zeros((num_envs, 1), device=self._device)
    self._num_components: int | None = None
    self._bias_initialized = False

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    """Reset bias values for specified environments."""
    indices = slice(None) if env_ids is None else env_ids
    # Sample new bias values.
    self._bias[indices] = self._bias_noise_cfg.apply(self._bias[indices])

  def _initialize_bias_shape(self, data_shape: torch.Size) -> None:
    """Initialize bias tensor shape based on data and configuration."""
    if self._sample_bias_per_component and not self._bias_initialized:
      *_, self._num_components = data_shape
      # Expand bias to match number of components.
      self._bias = self._bias.repeat(1, self._num_components)
      self._bias_initialized = True
      # Resample bias with new shape.
      self.reset()

  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    """Apply noise and additive bias to input data."""
    self._initialize_bias_shape(data.shape)
    noisy_data = super().__call__(data)
    return noisy_data + self._bias
