from collections.abc import Mapping
from typing import Any

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner with walking-to-football Actor transfer support."""

  env: RslRlVecEnvWrapper

  _PRETRAIN_ACTOR_OBS_DIM = 490
  _FOOTBALL_ACTOR_OBS_DIM = 520
  _TEMPORAL_PRETRAIN_CURRENT_DIM = 98
  _TEMPORAL_FOOTBALL_CURRENT_DIM = 105
  _TEMPORAL_LATENT_DIM = 64
  _CURRENT_PRETRAIN_DIM = 98
  _CURRENT_FOOTBALL_DIM = 105
  _CURRENT_B1_FOOTBALL_MLP_DIM = 169
  _STACKED_B1_FOOTBALL_MLP_DIM = 554
  _FIRST_LAYER_KEY = "mlp.0.weight"
  _TEMPORAL_CNN_FIRST_LAYER_KEY = "cnn_encoders.actor_history.net.0.weight"
  _NORMALIZER_VECTOR_KEYS = frozenset(
    {
      "obs_normalizer._mean",
      "obs_normalizer._var",
      "obs_normalizer._std",
    }
  )
  _TEMPORAL_HISTORY_NORMALIZER_VECTOR_KEYS = frozenset(
    {
      "obs_normalizers_3d.actor_history._mean",
      "obs_normalizers_3d.actor_history._var",
      "obs_normalizers_3d.actor_history._std",
    }
  )

  @staticmethod
  def _extract_actor_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
      raise ValueError("Pretrained checkpoint must contain a mapping")

    actor_state = checkpoint.get("actor_state_dict")
    if actor_state is None and "model_state_dict" in checkpoint:
      model_state = checkpoint["model_state_dict"]
      if not isinstance(model_state, Mapping):
        raise ValueError("Legacy model_state_dict must be a mapping")
      actor_state = {}
      for key, value in model_state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
          continue
        if key.startswith("actor."):
          actor_state[key.replace("actor.", "mlp.", 1)] = value
        elif key.startswith("actor_obs_normalizer."):
          actor_state[key.replace("actor_obs_normalizer.", "obs_normalizer.", 1)] = (
            value
          )
        elif key in {"std", "log_std"}:
          actor_state[key] = value

    if not isinstance(actor_state, Mapping):
      raise ValueError("Pretrained checkpoint has no Actor state dictionary")

    result = {
      key: value
      for key, value in actor_state.items()
      if isinstance(key, str) and isinstance(value, torch.Tensor)
    }
    if "std" in result:
      result["distribution.std_param"] = result.pop("std")
    if "log_std" in result:
      result["distribution.log_std_param"] = result.pop("log_std")
    return result

  def load_pretrained(self, path: str, map_location: str | None = None) -> None:
    """Initialize the football Actor from a compatible walking checkpoint.

    The Critic, optimizer, learning iteration, and environment state remain fresh.
    Supported contracts copy the shared walking observation prefix, leave new
    normalizer entries at their target defaults, and zero new MLP input columns.
    A newly added B1 history encoder remains at its target initialization.
    """
    checkpoint = torch.load(
      path,
      map_location=map_location or self.device,
      weights_only=False,
    )
    source = self._extract_actor_state_dict(checkpoint)
    target = self.alg.actor.state_dict()

    unexpected = source.keys() - target.keys()
    if unexpected:
      raise ValueError(
        f"Pretrained Actor contains unexpected parameters: {sorted(unexpected)}"
      )

    source_first = source[self._FIRST_LAYER_KEY]
    target_first = target[self._FIRST_LAYER_KEY]
    if source_first.ndim != 2 or target_first.ndim != 2:
      raise ValueError(
        "The Actor first-layer weight must be a matrix: "
        f"source={tuple(source_first.shape)}, target={tuple(target_first.shape)}"
      )
    legacy_transfer = (
      source_first.shape[1] == self._PRETRAIN_ACTOR_OBS_DIM
      and target_first.shape[1] == self._FOOTBALL_ACTOR_OBS_DIM
    )
    temporal_source_dim = (
      self._TEMPORAL_PRETRAIN_CURRENT_DIM + self._TEMPORAL_LATENT_DIM
    )
    temporal_target_dim = (
      self._TEMPORAL_FOOTBALL_CURRENT_DIM + self._TEMPORAL_LATENT_DIM
    )
    temporal_transfer = (
      source_first.shape[1] == temporal_source_dim
      and target_first.shape[1] == temporal_target_dim
      and self._TEMPORAL_CNN_FIRST_LAYER_KEY in source
      and self._TEMPORAL_CNN_FIRST_LAYER_KEY in target
    )
    current_mlp_transfer = source_first.shape[
      1
    ] == self._CURRENT_PRETRAIN_DIM and target_first.shape[1] in {
      self._CURRENT_FOOTBALL_DIM,
      self._CURRENT_B1_FOOTBALL_MLP_DIM,
    }
    stacked_b1_transfer = (
      source_first.shape[1] == self._PRETRAIN_ACTOR_OBS_DIM
      and target_first.shape[1] == self._STACKED_B1_FOOTBALL_MLP_DIM
      and any(key.startswith("cnn_encoders.actor_history.") for key in target)
    )
    if not (
      legacy_transfer
      or temporal_transfer
      or current_mlp_transfer
      or stacked_b1_transfer
    ):
      raise ValueError(
        "Unsupported walking-to-football Actor dimensions: "
        f"source={source_first.shape[1]}, target={target_first.shape[1]}. "
        "Expected legacy 490->520, TemporalCNN 162->169, current MLP "
        "98->105, current MLP to B1 98->169, or stacked B1 490->554."
      )

    target_only = target.keys() - source.keys()
    allowed_target_only = {
      key
      for key in target
      if key.startswith("cnn_encoders.actor_history.")
      or key.startswith("obs_normalizers_3d.actor_history.")
    }
    if target_only and (
      not (current_mlp_transfer or stacked_b1_transfer)
      or not target_only <= allowed_target_only
    ):
      raise ValueError(
        "Pretrained Actor parameters do not match the football Actor: "
        f"missing={sorted(target_only)}"
      )

    transferred: dict[str, torch.Tensor] = {}
    retained_target_keys: list[str] = []
    for key, target_value in target.items():
      if key not in source:
        transferred[key] = target_value
        retained_target_keys.append(key)
        continue
      source_value = source[key]
      if source_value.shape == target_value.shape:
        transferred[key] = source_value
      elif (
        legacy_transfer
        and key == self._FIRST_LAYER_KEY
        and (
          source_value.shape[0] == target_value.shape[0]
          and source_value.shape[1] == self._PRETRAIN_ACTOR_OBS_DIM
        )
      ):
        expanded = torch.zeros_like(target_value)
        expanded[:, : self._PRETRAIN_ACTOR_OBS_DIM] = source_value
        transferred[key] = expanded
      elif (
        legacy_transfer
        and key in self._NORMALIZER_VECTOR_KEYS
        and (
          source_value.shape[:-1] == target_value.shape[:-1]
          and source_value.shape[-1] == self._PRETRAIN_ACTOR_OBS_DIM
        )
      ):
        expanded = target_value.clone()
        expanded[..., : self._PRETRAIN_ACTOR_OBS_DIM] = source_value
        transferred[key] = expanded
      elif temporal_transfer and key == self._FIRST_LAYER_KEY:
        expanded = torch.zeros_like(target_value)
        source_current = self._TEMPORAL_PRETRAIN_CURRENT_DIM
        target_current = self._TEMPORAL_FOOTBALL_CURRENT_DIM
        expanded[:, :source_current] = source_value[:, :source_current]
        expanded[:, target_current:] = source_value[:, source_current:]
        transferred[key] = expanded
      elif temporal_transfer and key == self._TEMPORAL_CNN_FIRST_LAYER_KEY:
        expanded = torch.zeros_like(target_value)
        expanded[:, : self._TEMPORAL_PRETRAIN_CURRENT_DIM, :] = source_value
        transferred[key] = expanded
      elif temporal_transfer and key in (
        self._NORMALIZER_VECTOR_KEYS | self._TEMPORAL_HISTORY_NORMALIZER_VECTOR_KEYS
      ):
        expanded = target_value.clone()
        expanded[..., : self._TEMPORAL_PRETRAIN_CURRENT_DIM] = source_value
        transferred[key] = expanded
      elif (
        current_mlp_transfer
        and key == self._FIRST_LAYER_KEY
        and (source_value.shape[0] == target_value.shape[0])
      ):
        expanded = torch.zeros_like(target_value)
        expanded[:, : self._CURRENT_PRETRAIN_DIM] = source_value
        transferred[key] = expanded
      elif (
        current_mlp_transfer
        and key in self._NORMALIZER_VECTOR_KEYS
        and (
          source_value.shape[:-1] == target_value.shape[:-1]
          and source_value.shape[-1] == self._CURRENT_PRETRAIN_DIM
          and target_value.shape[-1] == self._CURRENT_FOOTBALL_DIM
        )
      ):
        expanded = target_value.clone()
        expanded[..., : self._CURRENT_PRETRAIN_DIM] = source_value
        transferred[key] = expanded
      elif (
        stacked_b1_transfer
        and key == self._FIRST_LAYER_KEY
        and source_value.shape[0] == target_value.shape[0]
      ):
        expanded = torch.zeros_like(target_value)
        expanded[:, : self._PRETRAIN_ACTOR_OBS_DIM] = source_value
        transferred[key] = expanded
      else:
        raise ValueError(
          f"Incompatible pretrained Actor parameter {key!r}: "
          f"source={tuple(source_value.shape)}, "
          f"target={tuple(target_value.shape)}"
        )

    self.alg.actor.load_state_dict(transferred, strict=True)
    if current_mlp_transfer or stacked_b1_transfer:
      source_parameter_count = sum(value.numel() for value in source.values())
      retained_parameter_count = sum(
        target[key].numel() for key in retained_target_keys
      )
      print(
        "[INFO] Walk->Football Actor transfer: "
        f"source tensors={len(source)}, source parameters={source_parameter_count}, "
        f"new tensors={retained_target_keys}, "
        f"new parameters={retained_parameter_count}."
      )

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
