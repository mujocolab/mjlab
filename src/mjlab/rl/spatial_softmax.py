"""CNN with a spatial softmax layer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN
from tensordict import TensorDict


class SpatialSoftmax(nn.Module):
  """Spatial soft-argmax over feature maps.

  Given input of shape ``(B, C, H, W)``, computes a softmax over each channel's spatial
  locations and returns the expected (x, y) coordinates, yielding output shape
  ``(B, C * 2)``.

  Args:
    height: Height of the input feature maps.
    width: Width of the input feature maps.
    temperature: Temperature for the spatial softmax. Lower values produce sharper
      distributions.
  """

  def __init__(self, height: int, width: int, temperature: float = 1.0) -> None:
    super().__init__()
    # Create normalised coordinate grids in [-1, 1].
    pos_x, pos_y = torch.meshgrid(
      torch.linspace(-1.0, 1.0, height),
      torch.linspace(-1.0, 1.0, width),
      indexing="ij",
    )
    # Register as buffers so they move with the module's device.
    self.register_buffer("pos_x", pos_x.reshape(1, 1, -1))
    self.register_buffer("pos_y", pos_y.reshape(1, 1, -1))
    self.temperature = temperature

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute spatial soft-argmax.

    Args:
        x: Feature maps of shape ``(B, C, H, W)``.

    Returns:
        Keypoint coordinates of shape ``(B, C * 2)``.
    """
    B, C, H, W = x.shape
    # Flatten spatial dims: (B, C, H*W).
    features = x.reshape(B, C, -1)
    # Spatial softmax.
    weights = torch.softmax(features / self.temperature, dim=-1)
    # Expected coordinates.
    pos_x: torch.Tensor = self.pos_x  # type: ignore[assignment]
    pos_y: torch.Tensor = self.pos_y  # type: ignore[assignment]
    expected_x = (weights * pos_x).sum(dim=-1)
    expected_y = (weights * pos_y).sum(dim=-1)
    # Interleave: (B, C*2).
    return torch.stack([expected_x, expected_y], dim=-1).reshape(B, C * 2)


class SpatialSoftmaxCNN(nn.Module):
  """CNN encoder with spatial-softmax pooling.

  Wraps ``rsl_rl.modules.CNN`` (created with ``global_pool="none"``
  and ``flatten=False``) followed by :class:`SpatialSoftmax`. Exposes the same
  interface as ``CNN``: :attr:`output_dim` (int) and :attr:`output_channels`
  (``None``, signalling flattened output).

  Args:
    input_dim: ``(H, W)`` of the input images.
    input_channels: Number of input channels.
    temperature: Temperature for the spatial softmax.
    **cnn_kwargs: Remaining keyword arguments forwarded to ``rsl_rl.modules.CNN``.
  """

  def __init__(
    self,
    input_dim: tuple[int, int],
    input_channels: int,
    temperature: float = 1.0,
    **cnn_kwargs: Any,
  ) -> None:
    super().__init__()
    # Override pooling/flatten — spatial softmax replaces these.
    cnn_kwargs.pop("global_pool", None)
    cnn_kwargs.pop("flatten", None)
    self.cnn = CNN(
      input_dim=input_dim,
      input_channels=input_channels,
      global_pool="none",
      flatten=False,
      **cnn_kwargs,
    )
    # cnn.output_dim is (H, W) when flatten=False.
    out_h, out_w = self.cnn.output_dim  # type: ignore[misc]
    num_channels: int = self.cnn.output_channels  # type: ignore[assignment]
    self.spatial_softmax = SpatialSoftmax(out_h, out_w, temperature)
    self._output_dim = num_channels * 2

  @property
  def output_dim(self) -> int:
    """Total flattened output dimension (C * 2)."""
    return self._output_dim

  @property
  def output_channels(self) -> None:
    """Always ``None`` (output is flattened keypoints)."""
    return None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.cnn(x)
    return self.spatial_softmax(features)


class SpatialSoftmaxCNNModel(CNNModel):
  """CNN model that uses spatial-softmax pooling.

  Drop-in replacement for ``rsl_rl.models.CNNModel``.  The only difference is that each
  CNN encoder is wrapped with :class:`SpatialSoftmaxCNN` instead of a plain ``CNN``.

  The ``spatial_softmax_temperature`` parameter is extracted from ``cnn_cfg`` before
  the remaining keys are forwarded to ``CNN``.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    cnn_cfg: dict[str, dict] | dict[str, Any],
    cnns: nn.ModuleDict | None = None,
    hidden_dims: tuple[int] | list[int] = [256, 256, 256],  # noqa: B006
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
  ) -> None:
    # Separate 1D / 2D observation groups (sets self.obs_groups_2d,
    # obs_dims_2d, obs_channels_2d; returns 1D info for MLPModel).
    self._get_obs_dim(obs, obs_groups, obs_set)

    if cnns is not None:
      if set(cnns.keys()) != set(self.obs_groups_2d):
        raise ValueError(
          "The 2D observations must be identical for all models sharing CNN encoders."
        )
      print(
        "Sharing CNN encoders between models, the CNN "
        "configurations of the receiving model are ignored."
      )
      _cnns = cnns
    else:
      # Expand a single flat config to per-group configs.
      if not all(isinstance(v, dict) for v in cnn_cfg.values()):
        cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
      assert len(cnn_cfg) == len(self.obs_groups_2d), (
        "The number of CNN configurations must match the "
        "number of 2D observation groups."
      )
      _cnns = {}
      for idx, obs_group in enumerate(self.obs_groups_2d):
        group_cfg = dict(cnn_cfg[obs_group])
        group_cfg.pop("spatial_softmax", None)
        temperature = group_cfg.pop("spatial_softmax_temperature", 1.0)
        _cnns[obs_group] = SpatialSoftmaxCNN(
          input_dim=self.obs_dims_2d[idx],
          input_channels=self.obs_channels_2d[idx],
          temperature=temperature,
          **group_cfg,
        )

    self.cnn_latent_dim = 0
    for cnn in _cnns.values():
      if cnn.output_channels is not None:
        raise ValueError(
          "The output of the CNN must be flattened before passing it to the MLP."
        )
      self.cnn_latent_dim += int(cnn.output_dim)  # type: ignore[arg-type]

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

    if isinstance(_cnns, nn.ModuleDict):
      self.cnns = _cnns
    else:
      self.cnns = nn.ModuleDict(_cnns)
