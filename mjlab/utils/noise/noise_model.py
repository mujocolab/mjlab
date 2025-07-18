from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
  from mjlab.utils.noise import noise_cfg

##
# Noise as functions.
##


def _ensure_tensor_device(
  value: torch.Tensor | float, device: torch.device
) -> torch.Tensor | float:
  """Ensure tensor is on the correct device, leave scalars unchanged."""
  if isinstance(value, torch.Tensor):
    return value.to(device=device)
  return value


def constant_noise(data: torch.Tensor, cfg: noise_cfg.ConstantNoiseCfg) -> torch.Tensor:
  cfg.bias = _ensure_tensor_device(cfg.bias, data.device)

  if cfg.operation == "add":
    return data + cfg.bias
  elif cfg.operation == "scale":
    return data * cfg.bias
  elif cfg.operation == "abs":
    return torch.zeros_like(data) + cfg.bias
  else:
    raise ValueError(f"Unsupported noise operation: {cfg.operation}")


def uniform_noise(data: torch.Tensor, cfg: noise_cfg.UniformNoiseCfg) -> torch.Tensor:
  cfg.n_min = _ensure_tensor_device(cfg.n_min, data.device)
  cfg.n_max = _ensure_tensor_device(cfg.n_max, data.device)

  # Generate uniform noise in [0, 1) and scale to [n_min, n_max).
  noise = torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min

  if cfg.operation == "add":
    return data + noise
  elif cfg.operation == "scale":
    return data * noise
  elif cfg.operation == "abs":
    return noise
  else:
    raise ValueError(f"Unsupported noise operation: {cfg.operation}")


def gaussian_noise(data: torch.Tensor, cfg: noise_cfg.GaussianNoiseCfg) -> torch.Tensor:
  cfg.mean = _ensure_tensor_device(cfg.mean, data.device)
  cfg.std = _ensure_tensor_device(cfg.std, data.device)

  # Generate standard normal noise and scale.
  noise = cfg.mean + cfg.std * torch.randn_like(data)

  if cfg.operation == "add":
    return data + noise
  elif cfg.operation == "scale":
    return data * noise
  elif cfg.operation == "abs":
    return noise
  else:
    raise ValueError(f"Unsupported noise operation: {cfg.operation}")


##
# Noise as classes.
##


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

  def reset(self, env_ids: Sequence[int] | None = None):
    """Reset noise model state. Override in subclasses if needed."""

  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    """Apply noise to input data."""
    return self._noise_model_cfg.noise_cfg.func(data, self._noise_model_cfg.noise_cfg)


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

  def reset(self, env_ids: Sequence[int] | None = None):
    """Reset bias values for specified environments."""
    if env_ids is None:
      env_ids = slice(None)
    # Sample new bias values.
    self._bias[env_ids] = self._bias_noise_cfg.func(
      self._bias[env_ids], self._bias_noise_cfg
    )

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
    result = noisy_data + self._bias
    return result
