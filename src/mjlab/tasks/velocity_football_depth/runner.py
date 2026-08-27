"""Runner support for walking-to-temporal-depth Actor transfer."""

from __future__ import annotations

from typing import Any, cast

import torch
import wandb

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.velocity_football.rl.runner import VelocityOnPolicyRunner

from .distillation import (
  BallPerceptionDistillation,
  DepthCoordinateStudentModel,
  DepthTemporalLatentStudentModel,
)


class DepthAuxVelocityOnPolicyRunner(VelocityOnPolicyRunner):
  """Initialize V1 from the 490-dimensional walking Actor."""

  def load_pretrained(self, path: str, map_location: str | None = None) -> None:
    checkpoint = torch.load(
      path, map_location=map_location or self.device, weights_only=False
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
      raise ValueError("Actor first-layer weights must be matrices")
    if source_first.shape[1] != self._PRETRAIN_ACTOR_OBS_DIM:
      raise ValueError(
        "Depth V1 requires a 490-dimensional walking Actor, got "
        f"{source_first.shape[1]}"
      )
    if source_first.shape[0] != target_first.shape[0]:
      raise ValueError(
        "Walking and depth Actor hidden dimensions differ: "
        f"{source_first.shape[0]} != {target_first.shape[0]}"
      )

    transferred: dict[str, torch.Tensor] = {}
    retained: list[str] = []
    for key, target_value in target.items():
      if key not in source:
        if not key.startswith("cnns.depth."):
          raise ValueError(f"Walking Actor is missing target parameter {key!r}")
        transferred[key] = target_value
        retained.append(key)
        continue

      source_value = source[key]
      if source_value.shape == target_value.shape:
        transferred[key] = source_value
      elif key == self._FIRST_LAYER_KEY:
        expanded = torch.zeros_like(target_value)
        expanded[:, : self._PRETRAIN_ACTOR_OBS_DIM] = source_value
        transferred[key] = expanded
      else:
        raise ValueError(
          f"Incompatible pretrained Actor parameter {key!r}: "
          f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
        )

    self.alg.actor.load_state_dict(transferred, strict=True)
    visual_dim = target_first.shape[1] - self._PRETRAIN_ACTOR_OBS_DIM
    print(
      "[INFO] Walk->Depth V1 Actor transfer: copied the 490-dimensional "
      f"walking policy, zero-initialized {visual_dim} visual input columns, "
      f"and retained {len(retained)} new depth-encoder tensors."
    )


class DepthTeacherDistillationRunner(MjlabOnPolicyRunner):
  """Load one coordinate checkpoint into the frozen Teacher and Student MLP."""

  def __init__(self, env, train_cfg: dict, *args, **kwargs) -> None:
    for key in ("student", "teacher"):
      model_cfg = train_cfg[key]
      if model_cfg.get("cnn_cfg") is None:
        model_cfg.pop("cnn_cfg", None)
      if model_cfg.get("distribution_cfg") is None:
        model_cfg.pop("distribution_cfg", None)
      if model_cfg.get("rnn_type") is None:
        for option in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
          model_cfg.pop(option, None)
    super().__init__(env, train_cfg, *args, **kwargs)

  def load_pretrained(self, path: str, map_location: str | None = None) -> None:
    checkpoint = torch.load(
      path,
      map_location=map_location or self.device,
      weights_only=False,
    )
    source = VelocityOnPolicyRunner._extract_actor_state_dict(checkpoint)
    algorithm = cast(BallPerceptionDistillation, self.alg)
    teacher = algorithm._raw_teacher
    student = algorithm._raw_student
    if not isinstance(
      student,
      (DepthCoordinateStudentModel, DepthTemporalLatentStudentModel),
    ):
      raise TypeError("Distillation Student has the wrong model type")

    teacher.load_state_dict(source, strict=True)
    student_source = source
    ignored_teacher_keys: list[str] = []
    if isinstance(student, DepthTemporalLatentStudentModel):
      target = student.state_dict()
      student_source = {}
      for key, value in source.items():
        if key in target:
          if target[key].shape != value.shape:
            raise ValueError(
              f"Incompatible frozen control parameter {key!r}: "
              f"teacher={tuple(value.shape)}, student={tuple(target[key].shape)}"
            )
          student_source[key] = value
        elif key.startswith(("cnn_encoders.", "obs_normalizers_3d.")):
          ignored_teacher_keys.append(key)
        else:
          raise ValueError(f"Teacher has unexpected Student key: {key!r}")

    incompatible = student.load_state_dict(student_source, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    if unexpected:
      raise ValueError(f"Teacher has unexpected Student keys: {unexpected}")
    missing_prefix = (
      "depth_encoder."
      if isinstance(student, DepthTemporalLatentStudentModel)
      else "perception."
    )
    if not missing or any(not key.startswith(missing_prefix) for key in missing):
      raise ValueError(
        "Student initialization must leave only depth-encoder parameters missing, "
        f"got {missing}"
      )
    algorithm.teacher_loaded = True
    print(
      "[INFO] Coordinate Teacher loaded; copied the frozen control backbone, "
      f"replaced {len(ignored_teacher_keys)} coordinate-CNN tensors, and retained "
      f"{len(missing)} randomly initialized depth-encoder tensors."
    )

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    algorithm = cast(BallPerceptionDistillation, self.alg)
    if not algorithm.teacher_loaded:
      raise ValueError("Load the coordinate Teacher before distillation")
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def save(self, path: str, infos=None) -> None:
    """Save and export the two-input depth Student with matching metadata.

    Depth history length/height/width are read back from the exported
    wrapper's own dummy inputs rather than hardcoded, since this runner
    serves both the 5-frame coordinate-reconstruction Student
    (``DepthCoordinateStudentModel``) and the 10-frame direct-latent
    Student (``DepthTemporalLatentStudentModel``).
    """
    MjlabOnPolicyRunner.save(self, path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      algorithm = cast(BallPerceptionDistillation, self.alg)
      onnx_model = cast(Any, algorithm.get_policy().as_onnx(verbose=False))
      _, depth_dummy = onnx_model.get_dummy_inputs()
      depth_history_length, depth_height, depth_width = depth_dummy.shape[1:]
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name = (
        (wandb.run.name or "local")
        if self.logger.logger_type == "wandb" and wandb.run
        else "local"
      )
      metadata = get_base_metadata(
        self.env.unwrapped,
        run_name,
        observation_group="student_proprio",
      )
      metadata.update(
        {
          "depth_history_length": float(depth_history_length),
          "depth_height": float(depth_height),
          "depth_width": float(depth_width),
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type == "wandb" and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as error:
      print(f"[WARN] Depth Student ONNX export failed (training continues): {error}")


class DepthStudentPpoRunner(VelocityOnPolicyRunner):
  """Initialize PPO from a completed depth-distillation checkpoint."""

  def load_pretrained(self, path: str, map_location: str | None = None) -> None:
    checkpoint: dict[str, Any] = torch.load(
      path,
      map_location=map_location or self.device,
      weights_only=False,
    )
    source = checkpoint.get("student_state_dict")
    if not isinstance(source, dict):
      raise ValueError(
        "Depth PPO initialization requires a distillation checkpoint with "
        "student_state_dict"
      )
    self.alg.actor.load_state_dict(source, strict=True)
    print("[INFO] Loaded distilled depth Student for PPO fine-tuning.")

  def save(self, path: str, infos=None) -> None:
    """Save and export the two-input depth Student with matching metadata."""
    MjlabOnPolicyRunner.save(self, path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name = (
        (wandb.run.name or "local")
        if self.logger.logger_type == "wandb" and wandb.run
        else "local"
      )
      metadata = get_base_metadata(
        self.env.unwrapped,
        run_name,
        observation_group="student_proprio",
      )
      metadata.update(
        {
          "depth_history_length": 5.0,
          "depth_height": 30.0,
          "depth_width": 40.0,
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type == "wandb" and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as error:
      print(f"[WARN] Depth Student ONNX export failed (training continues): {error}")
