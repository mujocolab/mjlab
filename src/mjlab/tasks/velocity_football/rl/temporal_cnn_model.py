"""Temporal CNN model for current observations plus unflattened history."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState
from tensordict import TensorDict

from .conv1d_encoder import Conv1dEncoder


def _export_input_name(group_name: str) -> str:
  if group_name in {"actor", "critic"}:
    return "obs"
  for prefix in ("actor_", "critic_"):
    if group_name.startswith(prefix):
      return "obs_" + group_name[len(prefix) :]
  return group_name


class TemporalCNNModel(MLPModel):
  """Encode each 3-D history group with Conv1d, then use the standard MLP."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    cnn_cfg: dict[str, Any] | None = None,
  ) -> None:
    self._get_obs_dim(obs, obs_groups, obs_set)
    self.cnn_encoders_dict: dict[str, Conv1dEncoder] = {}
    self.cnn_latent_dim = 0
    for group_name, obs_dim in zip(self.obs_groups_3d, self.obs_dims_3d, strict=True):
      encoder = Conv1dEncoder(input_channels=obs_dim, **(cnn_cfg or {}))
      self.cnn_encoders_dict[group_name] = encoder
      self.cnn_latent_dim += encoder.output_dim
    self._obs_normalization_3d = obs_normalization

    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims,
      activation,
      obs_normalization,
      distribution_cfg,
    )
    self.cnn_encoders = nn.ModuleDict(self.cnn_encoders_dict)
    if obs_normalization:
      self.obs_normalizers_3d = nn.ModuleDict(
        {
          group_name: EmpiricalNormalization(obs_dim)
          for group_name, obs_dim in zip(
            self.obs_groups_3d, self.obs_dims_3d, strict=True
          )
        }
      )
    else:
      self.obs_normalizers_3d = nn.ModuleDict(
        {group_name: nn.Identity() for group_name in self.obs_groups_3d}
      )

  def _get_obs_dim(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str], int]:
    groups_1d: list[str] = []
    groups_3d: list[str] = []
    dims_3d: list[int] = []
    history_lengths: list[int] = []
    obs_dim_1d = 0
    for group_name in obs_groups[obs_set]:
      ndim = len(obs[group_name].shape)
      if ndim == 2:
        groups_1d.append(group_name)
        obs_dim_1d += obs[group_name].shape[-1]
      elif ndim == 3:
        groups_3d.append(group_name)
        history_lengths.append(obs[group_name].shape[1])
        dims_3d.append(obs[group_name].shape[-1])
      else:
        raise ValueError(
          "TemporalCNNModel expects (B,D) or (B,T,D), "
          f"got {tuple(obs[group_name].shape)} for {group_name!r}"
        )
    self.obs_groups_1d = groups_1d
    self.obs_groups_3d = groups_3d
    self.obs_dims_3d = dims_3d
    self.history_lengths = history_lengths
    return groups_1d, obs_dim_1d

  def _get_latent_dim(self) -> int:
    return self.obs_dim + self.cnn_latent_dim

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    latent_parts = [super().get_latent(obs, masks, hidden_state)]
    for group_name in self.obs_groups_3d:
      history = self.obs_normalizers_3d[group_name](obs[group_name])
      latent_parts.append(self.cnn_encoders[group_name](history.permute(0, 2, 1)))
    return torch.cat(latent_parts, dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    super().update_normalization(obs)
    if self._obs_normalization_3d:
      for group_name in self.obs_groups_3d:
        history = obs[group_name]
        batch, length, features = history.shape
        self.obs_normalizers_3d[group_name].update(  # type: ignore[operator]
          history.reshape(batch * length, features)
        )

  def as_jit(self) -> nn.Module:
    return _TorchTemporalCNNModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxTemporalCNNModel(self, verbose)


class _TorchTemporalCNNModel(nn.Module):
  def __init__(self, model: TemporalCNNModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.cnn_normalizers = nn.ModuleList(
      [copy.deepcopy(model.obs_normalizers_3d[g]) for g in model.obs_groups_3d]
    )
    self.cnn_encoders = nn.ModuleList(
      [copy.deepcopy(model.cnn_encoders[g]) for g in model.obs_groups_3d]
    )
    self.mlp = copy.deepcopy(model.mlp)
    self.deterministic_output = (
      model.distribution.as_deterministic_output_module()
      if model.distribution is not None
      else nn.Identity()
    )

  def forward(
    self,
    obs_1d: torch.Tensor,
    *obs_temporal: torch.Tensor,
  ) -> torch.Tensor:
    latent_parts = [self.obs_normalizer(obs_1d)]
    for history, normalizer, encoder in zip(
      obs_temporal,
      self.cnn_normalizers,
      self.cnn_encoders,
      strict=True,
    ):
      latent_parts.append(encoder(normalizer(history).permute(0, 2, 1)))
    return self.deterministic_output(self.mlp(torch.cat(latent_parts, dim=-1)))

  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxTemporalCNNModel(_TorchTemporalCNNModel):
  def __init__(self, model: TemporalCNNModel, verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose
    self._obs_dim_1d = model.obs_dim
    self._obs_dims_3d = model.obs_dims_3d
    self._history_lengths = model.history_lengths
    self._temporal_input_names = [
      _export_input_name(group_name) for group_name in model.obs_groups_3d
    ]

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    return (
      torch.zeros(1, self._obs_dim_1d),
      *(
        torch.zeros(1, history_length, obs_dim)
        for history_length, obs_dim in zip(
          self._history_lengths,
          self._obs_dims_3d,
          strict=True,
        )
      ),
    )

  @property
  def input_names(self) -> list[str]:
    return ["obs", *self._temporal_input_names]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
