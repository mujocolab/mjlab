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
  _FOOTBALL_ACTOR_OBS_DIM = 535
  _FIRST_LAYER_KEY = "mlp.0.weight"
  _NORMALIZER_VECTOR_KEYS = frozenset(
    {
      "obs_normalizer._mean",
      "obs_normalizer._var",
      "obs_normalizer._std",
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
    The walking observations are a strict prefix of the football observations, so
    the first layer copies its first 490 columns and zeros the 45 new columns.
    """
    checkpoint = torch.load(
      path,
      map_location=map_location or self.device,
      weights_only=False,
    )
    source = self._extract_actor_state_dict(checkpoint)
    target = self.alg.actor.state_dict()

    missing = target.keys() - source.keys()
    unexpected = source.keys() - target.keys()
    if missing or unexpected:
      raise ValueError(
        "Pretrained Actor parameters do not match the football Actor: "
        f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
      )

    source_first = source[self._FIRST_LAYER_KEY]
    target_first = target[self._FIRST_LAYER_KEY]
    if source_first.ndim != 2 or target_first.ndim != 2:
      raise ValueError(
        "The Actor first-layer weight must be a matrix: "
        f"source={tuple(source_first.shape)}, target={tuple(target_first.shape)}"
      )
    if source_first.shape[1] != self._PRETRAIN_ACTOR_OBS_DIM:
      raise ValueError(
        f"Expected a walking Actor with 490 observations, got {source_first.shape[1]}"
      )
    if target_first.shape[1] != self._FOOTBALL_ACTOR_OBS_DIM:
      raise ValueError(
        "--pretrained-checkpoint requires the football Actor with 535 "
        f"observations, got {target_first.shape[1]}"
      )

    transferred: dict[str, torch.Tensor] = {}
    for key, target_value in target.items():
      source_value = source[key]
      if source_value.shape == target_value.shape:
        transferred[key] = source_value
      elif key == self._FIRST_LAYER_KEY and (
        source_value.shape[0] == target_value.shape[0]
        and source_value.shape[1] == self._PRETRAIN_ACTOR_OBS_DIM
      ):
        expanded = torch.zeros_like(target_value)
        expanded[:, : self._PRETRAIN_ACTOR_OBS_DIM] = source_value
        transferred[key] = expanded
      elif key in self._NORMALIZER_VECTOR_KEYS and (
        source_value.shape[:-1] == target_value.shape[:-1]
        and source_value.shape[-1] == self._PRETRAIN_ACTOR_OBS_DIM
      ):
        expanded = target_value.clone()
        expanded[..., : self._PRETRAIN_ACTOR_OBS_DIM] = source_value
        transferred[key] = expanded
      else:
        raise ValueError(
          f"Incompatible pretrained Actor parameter {key!r}: "
          f"source={tuple(source_value.shape)}, "
          f"target={tuple(target_value.shape)}"
        )

    self.alg.actor.load_state_dict(transferred, strict=True)

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
