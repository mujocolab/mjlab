"""Temporal depth encoder with an explicitly supervised ball bottleneck."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models import CNNModel, MLPModel
from tensordict import TensorDict

from mjlab.rl.spatial_softmax import SpatialSoftmax


class TemporalDepthBallEncoder(nn.Module):
  """Encode depth history while preserving spatial and metric information."""

  output_channels: None = None

  def __init__(
    self,
    input_dim: tuple[int, int],
    input_channels: int,
    output_channels: tuple[int, ...] | list[int] = (16, 32, 64),
    latent_dim: int = 128,
  ) -> None:
    super().__init__()
    if len(output_channels) != 3:
      raise ValueError("TemporalDepthBallEncoder expects exactly three CNN stages")

    channels = [input_channels, *output_channels]
    layers: list[nn.Module] = []
    for index in range(3):
      kernel_size = 5 if index == 0 else 3
      layers.extend(
        (
          nn.Conv2d(
            channels[index],
            channels[index + 1],
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
          ),
          nn.ELU(),
        )
      )
    self.features = nn.Sequential(*layers)

    with torch.no_grad():
      sample = torch.zeros(1, input_channels, *input_dim)
      feature_sample = self.features(sample)
    feature_channels = feature_sample.shape[1]
    feature_height, feature_width = feature_sample.shape[-2:]
    self.spatial_softmax = SpatialSoftmax(feature_height, feature_width)

    metric_dim = input_channels * 3
    fusion_dim = feature_channels * 4 + metric_dim
    self.fusion = nn.Sequential(nn.Linear(fusion_dim, latent_dim), nn.ELU())
    self.ball_head = nn.Sequential(
      nn.Linear(latent_dim, 64),
      nn.ELU(),
      nn.Linear(64, 4),
    )
    self._output_dim = latent_dim + 4
    self._last_ball_prediction: torch.Tensor | None = None

  @property
  def output_dim(self) -> int:
    return self._output_dim

  @property
  def last_ball_prediction(self) -> torch.Tensor:
    if self._last_ball_prediction is None:
      raise RuntimeError("The depth encoder has not run a forward pass")
    return self._last_ball_prediction

  def clear_auxiliary_cache(self) -> None:
    """Drop the autograd tensor cached for the current PPO mini-batch."""
    self._last_ball_prediction = None

  def forward(self, depth_history: torch.Tensor) -> torch.Tensor:
    features = self.features(depth_history)
    keypoints = self.spatial_softmax(features)
    feature_mean = features.mean(dim=(-2, -1))
    feature_max = features.amax(dim=(-2, -1))
    depth_mean = depth_history.mean(dim=(-2, -1))
    depth_min = depth_history.amin(dim=(-2, -1))
    depth_max = depth_history.amax(dim=(-2, -1))
    fused = torch.cat(
      (keypoints, feature_mean, feature_max, depth_mean, depth_min, depth_max),
      dim=-1,
    )
    latent = self.fusion(fused)

    raw_prediction = self.ball_head(latent)
    xy = torch.tanh(raw_prediction[:, :2])
    distance = torch.sigmoid(raw_prediction[:, 2:3])
    confidence_logit = raw_prediction[:, 3:4]
    self._last_ball_prediction = torch.cat((xy, distance, confidence_logit), dim=-1)
    policy_prediction = torch.cat(
      (xy, distance, torch.sigmoid(confidence_logit)), dim=-1
    )
    return torch.cat((latent, policy_prediction), dim=-1)


class DepthAuxCNNModel(CNNModel):
  """CNNModel whose temporal encoder exposes supervised ball predictions."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    cnn_cfg: dict[str, Any],
    cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
  ) -> None:
    self._get_obs_dim(obs, obs_groups, obs_set)
    if cnns is None:
      if self.obs_groups_2d != ["depth"]:
        raise ValueError(f"Expected one 'depth' image group, got {self.obs_groups_2d}")
      depth = obs["depth"]
      cnns = {
        "depth": TemporalDepthBallEncoder(
          input_dim=(depth.shape[-2], depth.shape[-1]),
          input_channels=depth.shape[1],
          output_channels=cnn_cfg.get("output_channels", (16, 32, 64)),
          latent_dim=cnn_cfg.get("latent_dim", 128),
        )
      }
    elif set(cnns) != set(self.obs_groups_2d):
      raise ValueError("Shared CNN keys do not match the 2D observation groups")

    self.cnn_latent_dim = sum(int(cnn.output_dim) for cnn in cnns.values())  # type: ignore[arg-type]
    MLPModel.__init__(
      self,
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    self.cnns = cnns if isinstance(cnns, nn.ModuleDict) else nn.ModuleDict(cnns)

  @property
  def auxiliary_prediction(self) -> torch.Tensor:
    encoder = self.cnns["depth"]
    if not isinstance(encoder, TemporalDepthBallEncoder):
      raise TypeError(f"Unexpected depth encoder type: {type(encoder).__name__}")
    return encoder.last_ball_prediction

  def _clear_auxiliary_cache(self) -> None:
    encoder = self.cnns["depth"]
    if isinstance(encoder, TemporalDepthBallEncoder):
      encoder.clear_auxiliary_cache()

  def as_jit(self) -> nn.Module:
    self._clear_auxiliary_cache()
    return super().as_jit()

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    self._clear_auxiliary_cache()
    return super().as_onnx(verbose)
