"""Coordinate-teacher distillation for depth-based football control."""

from __future__ import annotations

import copy
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import Distillation
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab.rl.spatial_softmax import SpatialSoftmax

PROPRIO_HISTORY_DIM = 490
BALL_POSITION_HISTORY_DIM = 10
BALL_TO_FEET_HISTORY_DIM = 20
BALL_VISIBILITY_HISTORY_DIM = 5
FOOTBALL_HISTORY_DIM = (
  BALL_POSITION_HISTORY_DIM + BALL_TO_FEET_HISTORY_DIM + BALL_VISIBILITY_HISTORY_DIM
)
TEACHER_ACTOR_DIM = PROPRIO_HISTORY_DIM + FOOTBALL_HISTORY_DIM
TEMPORAL_FOOTBALL_HISTORY_LENGTH = 10
TEMPORAL_FOOTBALL_DIM = 7


class TemporalDepthFootballEncoder(nn.Module):
  """Predict the coordinate Teacher's 35 football-history inputs from depth."""

  def __init__(
    self,
    input_dim: tuple[int, int],
    input_channels: int,
    output_channels: tuple[int, ...] | list[int] = (16, 32, 64),
    latent_dim: int = 128,
  ) -> None:
    super().__init__()
    if len(output_channels) != 3:
      raise ValueError("The depth encoder expects exactly three CNN stages")

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

    # SpatialSoftmax contributes 2*C coordinates. Mean and max pooling each
    # contribute C values, while three metric statistics are kept per frame.
    metric_dim = input_channels * 3
    fusion_dim = feature_channels * 4 + metric_dim
    self.fusion = nn.Sequential(
      nn.Linear(fusion_dim, latent_dim),
      nn.ELU(),
    )
    self.football_head = nn.Sequential(
      nn.Linear(latent_dim, 64),
      nn.ELU(),
      nn.Linear(64, FOOTBALL_HISTORY_DIM),
    )

  def encode(self, depth_history: torch.Tensor) -> torch.Tensor:
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
    return self.fusion(fused)

  def forward(self, depth_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    latent = self.encode(depth_history)
    raw = self.football_head(latent)

    coordinates = raw[:, : BALL_POSITION_HISTORY_DIM + BALL_TO_FEET_HISTORY_DIM]
    visibility_logits = raw[:, -BALL_VISIBILITY_HISTORY_DIM:]
    visibility = torch.sigmoid(visibility_logits)
    coordinate_mask = torch.cat(
      (
        visibility.repeat_interleave(2, dim=-1),
        visibility.repeat_interleave(4, dim=-1),
      ),
      dim=-1,
    )
    football_history = torch.cat(
      (coordinates * coordinate_mask, visibility),
      dim=-1,
    )
    return football_history, visibility_logits


class DepthCoordinateStudentModel(MLPModel):
  """Depth front-end followed by an unchanged 525-input coordinate Actor."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    cnn_cfg: dict[str, Any],
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict[str, Any] | None = None,
  ) -> None:
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    if self.obs_dim != TEACHER_ACTOR_DIM:
      raise ValueError(
        f"Depth Student requires a {TEACHER_ACTOR_DIM}-input Teacher contract, "
        f"got {self.obs_dim}"
      )
    proprio = cast(torch.Tensor, obs["student_proprio"])
    depth = cast(torch.Tensor, obs["depth"])
    if proprio.shape[-1] != PROPRIO_HISTORY_DIM:
      raise ValueError(
        f"Expected {PROPRIO_HISTORY_DIM} proprio inputs, got {proprio.shape[-1]}"
      )
    if depth.ndim != 4:
      raise ValueError(f"Expected depth shape (B,C,H,W), got {tuple(depth.shape)}")

    self.perception = TemporalDepthFootballEncoder(
      input_dim=(depth.shape[-2], depth.shape[-1]),
      input_channels=depth.shape[1],
      output_channels=cnn_cfg.get("output_channels", (16, 32, 64)),
      latent_dim=cnn_cfg.get("latent_dim", 128),
    )
    self.freeze_coordinate_actor = bool(cnn_cfg.get("freeze_coordinate_actor", True))
    if self.freeze_coordinate_actor:
      for module in (self.obs_normalizer, self.mlp, self.distribution):
        if module is None:
          continue
        for parameter in module.parameters():
          parameter.requires_grad_(False)

    self._predicted_football_history: torch.Tensor | None = None
    self._visibility_logits: torch.Tensor | None = None

  @property
  def predicted_football_history(self) -> torch.Tensor:
    if self._predicted_football_history is None:
      raise RuntimeError("Student has not completed a forward pass")
    return self._predicted_football_history

  @property
  def visibility_logits(self) -> torch.Tensor:
    if self._visibility_logits is None:
      raise RuntimeError("Student has not completed a forward pass")
    return self._visibility_logits

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
  ) -> torch.Tensor:
    del masks, hidden_state
    depth = cast(torch.Tensor, obs["depth"])
    proprio = cast(torch.Tensor, obs["student_proprio"])
    football_history, visibility_logits = self.perception(depth)
    self._predicted_football_history = football_history
    self._visibility_logits = visibility_logits
    reconstructed = torch.cat(
      (proprio, football_history),
      dim=-1,
    )
    return self.obs_normalizer(reconstructed)

  def update_normalization(self, obs: TensorDict) -> None:
    """Keep the coordinate Teacher's frozen normalization statistics."""
    del obs

  def as_jit(self) -> nn.Module:
    return _ExportedDepthCoordinateStudent(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _ExportedDepthCoordinateStudent(self)


class TemporalDepthLatentEncoder(TemporalDepthFootballEncoder):
  """Map depth history directly to a control latent without coordinates."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.football_head = nn.Identity()

  def forward(self, depth_history: torch.Tensor) -> torch.Tensor:
    return self.encode(depth_history)


class DepthTemporalLatentStudentModel(MLPModel):
  """Replace the B1 coordinate CNN with a direct 64-D depth encoder."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    cnn_cfg: dict[str, Any],
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict[str, Any] | None = None,
  ) -> None:
    self.depth_latent_dim = int(cnn_cfg.get("latent_dim", 64))
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    if self.obs_dim != PROPRIO_HISTORY_DIM:
      raise ValueError(
        f"Temporal depth Student requires {PROPRIO_HISTORY_DIM} proprio inputs, "
        f"got {self.obs_dim}"
      )
    if self.depth_latent_dim != 64:
      raise ValueError(
        "The B1 control MLP requires a 64-dimensional visual latent, "
        f"got {self.depth_latent_dim}"
      )
    proprio = cast(torch.Tensor, obs["student_proprio"])
    depth = cast(torch.Tensor, obs["depth"])
    if proprio.shape[-1] != PROPRIO_HISTORY_DIM:
      raise ValueError(
        f"Expected {PROPRIO_HISTORY_DIM} proprio inputs, got {proprio.shape[-1]}"
      )
    if depth.ndim != 4 or depth.shape[1] != TEMPORAL_FOOTBALL_HISTORY_LENGTH:
      raise ValueError(f"Expected depth shape (B,10,H,W), got {tuple(depth.shape)}")

    self.depth_encoder = TemporalDepthLatentEncoder(
      input_dim=(depth.shape[-2], depth.shape[-1]),
      input_channels=depth.shape[1],
      output_channels=cnn_cfg.get("output_channels", (16, 32, 64)),
      latent_dim=self.depth_latent_dim,
    )
    self.freeze_coordinate_actor = bool(cnn_cfg.get("freeze_coordinate_actor", True))
    self.train_mlp_last_layer_only = bool(
      cnn_cfg.get("train_mlp_last_layer_only", False)
    )
    if self.freeze_coordinate_actor and self.train_mlp_last_layer_only:
      raise ValueError(
        "freeze_coordinate_actor and train_mlp_last_layer_only are mutually exclusive"
      )
    if self.freeze_coordinate_actor:
      for module in (
        self.obs_normalizer,
        self.mlp,
        self.distribution,
      ):
        if module is None:
          continue
        for parameter in module.parameters():
          parameter.requires_grad_(False)
    elif self.train_mlp_last_layer_only:
      for module in (self.obs_normalizer, self.mlp, self.distribution):
        if module is None:
          continue
        for parameter in module.parameters():
          parameter.requires_grad_(False)
      for parameter in self.last_mlp_linear.parameters():
        parameter.requires_grad_(True)

    self._visual_latent: torch.Tensor | None = None

  @property
  def last_mlp_linear(self) -> nn.Linear:
    linear_layers = [
      module for module in self.mlp.modules() if isinstance(module, nn.Linear)
    ]
    if not linear_layers:
      raise RuntimeError("Student control MLP contains no linear layer")
    return linear_layers[-1]

  @property
  def visual_latent(self) -> torch.Tensor:
    if self._visual_latent is None:
      raise RuntimeError("Student has not completed a forward pass")
    return self._visual_latent

  def _get_latent_dim(self) -> int:
    return self.obs_dim + self.depth_latent_dim

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
  ) -> torch.Tensor:
    del masks, hidden_state
    proprio = cast(torch.Tensor, obs["student_proprio"])
    depth = cast(torch.Tensor, obs["depth"])
    visual_latent = self.depth_encoder(depth)
    self._visual_latent = visual_latent
    return torch.cat((self.obs_normalizer(proprio), visual_latent), dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    """Keep all frozen Teacher normalizers unchanged."""
    del obs

  def as_jit(self) -> nn.Module:
    return _ExportedDepthTemporalLatentStudent(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _ExportedDepthTemporalLatentStudent(self)


class _ExportedDepthCoordinateStudent(nn.Module):
  """Deterministic two-input export wrapper for the depth Student."""

  is_recurrent: bool = False

  def __init__(self, model: DepthCoordinateStudentModel) -> None:
    super().__init__()
    self.perception = copy.deepcopy(model.perception)
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is None:
      self.deterministic_output = nn.Identity()
    else:
      self.deterministic_output = model.distribution.as_deterministic_output_module()

  def forward(
    self,
    proprio: torch.Tensor,
    depth: torch.Tensor,
  ) -> torch.Tensor:
    football_history, _ = self.perception(depth)
    reconstructed = torch.cat((proprio, football_history), dim=-1)
    output = self.mlp(self.obs_normalizer(reconstructed))
    return self.deterministic_output(output)

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, PROPRIO_HISTORY_DIM),
      torch.zeros(1, 5, 30, 40),
    )

  @property
  def input_names(self) -> list[str]:
    return ["proprio", "depth"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]

  @torch.jit.export
  def reset(self) -> None:
    pass


class _ExportedDepthTemporalLatentStudent(nn.Module):
  """Two-input deployment wrapper with no coordinate reconstruction path."""

  is_recurrent: bool = False

  def __init__(self, model: DepthTemporalLatentStudentModel) -> None:
    super().__init__()
    self.depth_encoder = copy.deepcopy(model.depth_encoder)
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is None:
      self.deterministic_output = nn.Identity()
    else:
      self.deterministic_output = model.distribution.as_deterministic_output_module()

  def forward(self, proprio: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    visual_latent = self.depth_encoder(depth)
    latent = torch.cat((self.obs_normalizer(proprio), visual_latent), dim=-1)
    return self.deterministic_output(self.mlp(latent))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, PROPRIO_HISTORY_DIM),
      torch.zeros(1, TEMPORAL_FOOTBALL_HISTORY_LENGTH, 30, 40),
    )

  @property
  def input_names(self) -> list[str]:
    return ["proprio", "depth"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]

  @torch.jit.export
  def reset(self) -> None:
    pass


class TeacherRolloutDistillation(Distillation):
  """Behavior cloning with Teacher, Student, or scheduled mixed rollouts."""

  def __init__(
    self,
    *args,
    rollout_policy: str = "teacher",
    student_rollout_warmup_updates: int = 0,
    student_rollout_ramp_updates: int = 1,
    student_rollout_final_probability: float = 1.0,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if rollout_policy not in {"teacher", "student", "mixed"}:
      raise ValueError("rollout_policy must be 'teacher', 'student', or 'mixed'")
    if student_rollout_warmup_updates < 0:
      raise ValueError("student_rollout_warmup_updates must be non-negative")
    if student_rollout_ramp_updates <= 0:
      raise ValueError("student_rollout_ramp_updates must be positive")
    if not 0.0 <= student_rollout_final_probability <= 1.0:
      raise ValueError("student_rollout_final_probability must be in [0, 1]")
    self.rollout_policy = rollout_policy
    self.student_rollout_warmup_updates = student_rollout_warmup_updates
    self.student_rollout_ramp_updates = student_rollout_ramp_updates
    self.student_rollout_final_probability = student_rollout_final_probability

  @property
  def student_rollout_probability(self) -> float:
    if self.rollout_policy == "teacher":
      return 0.0
    if self.rollout_policy == "student":
      return 1.0
    progress = max(0, self.num_updates - self.student_rollout_warmup_updates)
    fraction = min(1.0, progress / self.student_rollout_ramp_updates)
    return fraction * self.student_rollout_final_probability

  def act(self, obs: TensorDict) -> torch.Tensor:
    with torch.no_grad():
      student_actions = self.student(obs, stochastic_output=True)
      teacher_actions = self.teacher(obs, stochastic_output=False)
    self.transition.actions = student_actions.detach()
    self.transition.privileged_actions = teacher_actions.detach()
    self.transition.observations = obs
    probability = self.student_rollout_probability
    if probability <= 0.0:
      return teacher_actions
    if probability >= 1.0:
      return student_actions
    use_student = (
      torch.rand(student_actions.shape[0], 1, device=student_actions.device)
      < probability
    )
    return torch.where(use_student, student_actions, teacher_actions)

  def update(self) -> dict[str, float]:
    losses = super().update()
    losses["student_rollout_probability"] = self.student_rollout_probability
    return losses


class ConstrainedLatentDistillation(TeacherRolloutDistillation):
  """Action imitation with latent alignment and an anchored final control layer."""

  def __init__(
    self,
    *args,
    latent_loss_coef: float = 0.1,
    mlp_anchor_loss_coef: float = 1.0e-3,
    mlp_learning_rate: float = 1.0e-5,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if not isinstance(self._raw_student, DepthTemporalLatentStudentModel):
      raise TypeError(
        "ConstrainedLatentDistillation requires DepthTemporalLatentStudentModel"
      )
    if not self._raw_student.train_mlp_last_layer_only:
      raise ValueError(
        "ConstrainedLatentDistillation requires train_mlp_last_layer_only=True"
      )
    if latent_loss_coef < 0.0 or mlp_anchor_loss_coef < 0.0:
      raise ValueError("Loss coefficients must be non-negative")
    if mlp_learning_rate <= 0.0:
      raise ValueError("mlp_learning_rate must be positive")

    self.latent_loss_coef = latent_loss_coef
    self.mlp_anchor_loss_coef = mlp_anchor_loss_coef
    self.mlp_learning_rate = mlp_learning_rate
    self._mlp_anchor: dict[str, torch.Tensor] = {}
    self._capture_mlp_anchor()

    last_layer_ids = {
      id(parameter) for parameter in self._raw_student.last_mlp_linear.parameters()
    }
    depth_parameters = [
      parameter
      for parameter in self._raw_student.depth_encoder.parameters()
      if parameter.requires_grad
    ]
    mlp_parameters = [
      parameter
      for parameter in self._raw_student.parameters()
      if parameter.requires_grad and id(parameter) in last_layer_ids
    ]
    unexpected_trainable = [
      name
      for name, parameter in self._raw_student.named_parameters()
      if parameter.requires_grad
      and not name.startswith("depth_encoder.")
      and id(parameter) not in last_layer_ids
    ]
    if unexpected_trainable:
      raise ValueError(
        f"Unexpected trainable Student parameters: {unexpected_trainable}"
      )
    if not depth_parameters or not mlp_parameters:
      raise ValueError("Both depth encoder and final MLP layer must be trainable")

    optimizer_type = type(self.optimizer)
    self.optimizer = optimizer_type(
      [
        {"params": depth_parameters, "lr": self.learning_rate},
        {"params": mlp_parameters, "lr": self.mlp_learning_rate},
      ]
    )

  def _capture_mlp_anchor(self) -> None:
    student = cast(DepthTemporalLatentStudentModel, self._raw_student)
    self._mlp_anchor = {
      name: parameter.detach().clone()
      for name, parameter in student.last_mlp_linear.named_parameters()
    }

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    """Load model weights and iteration, but start a fresh constrained optimizer."""
    del load_cfg
    load_iteration = super().load(
      loaded_dict,
      {
        "student": True,
        "teacher": True,
        "optimizer": False,
        "iteration": True,
      },
      strict,
    )
    self._capture_mlp_anchor()
    return load_iteration

  def _anchor_loss(self) -> torch.Tensor:
    student = cast(DepthTemporalLatentStudentModel, self._raw_student)
    losses = [
      F.mse_loss(parameter, self._mlp_anchor[name])
      for name, parameter in student.last_mlp_linear.named_parameters()
    ]
    return torch.stack(losses).mean()

  def update(self) -> dict[str, float]:
    student = cast(DepthTemporalLatentStudentModel, self._raw_student)
    self.num_updates += 1
    totals = {"behavior": 0.0, "latent": 0.0, "mlp_anchor": 0.0}
    count = 0

    for _ in range(self.num_learning_epochs):
      for batch in self.storage.generator():
        observations = batch.observations
        privileged_actions = batch.privileged_actions
        if observations is None or privileged_actions is None:
          raise RuntimeError("Distillation batch is missing Teacher supervision")

        student_actions = self.student(observations)
        behavior_loss = self.loss_fn(student_actions, privileged_actions)
        with torch.no_grad():
          teacher_latent = self._raw_teacher.get_latent(observations)[
            ..., -student.depth_latent_dim :
          ]
        if teacher_latent.shape != student.visual_latent.shape:
          raise RuntimeError(
            "Teacher/Student visual latent shapes differ: "
            f"{tuple(teacher_latent.shape)} != {tuple(student.visual_latent.shape)}"
          )
        latent_loss = F.smooth_l1_loss(student.visual_latent, teacher_latent)
        anchor_loss = self._anchor_loss()
        loss = (
          behavior_loss
          + self.latent_loss_coef * latent_loss
          + self.mlp_anchor_loss_coef * anchor_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
          self.reduce_parameters()
        if self.max_grad_norm:
          nn.utils.clip_grad_norm_(
            (
              parameter
              for group in self.optimizer.param_groups
              for parameter in group["params"]
            ),
            self.max_grad_norm,
          )
        self.optimizer.step()

        totals["behavior"] += behavior_loss.item()
        totals["latent"] += latent_loss.item()
        totals["mlp_anchor"] += anchor_loss.item()
        count += 1

    self.storage.clear()
    if count == 0:
      raise RuntimeError("Distillation storage produced no training batches")
    losses = {name: value / count for name, value in totals.items()}
    losses["student_rollout_probability"] = self.student_rollout_probability
    return losses


class BallPerceptionDistillation(Distillation):
  """Distill actions while explicitly supervising reconstructed football inputs."""

  def __init__(
    self,
    *args,
    action_loss_coef: float = 1.0,
    position_loss_coef: float = 1.0,
    visibility_loss_coef: float = 0.2,
    rollout_policy: str = "teacher",
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if rollout_policy not in {"teacher", "student"}:
      raise ValueError("rollout_policy must be 'teacher' or 'student'")
    self.action_loss_coef = action_loss_coef
    self.position_loss_coef = position_loss_coef
    self.visibility_loss_coef = visibility_loss_coef
    self.rollout_policy = rollout_policy

  def act(self, obs: TensorDict) -> torch.Tensor:
    with torch.no_grad():
      # Update the Student distribution so the common runner can log its
      # standard deviation, even while Teacher actions control the rollout.
      student_actions = self.student(obs, stochastic_output=True)
      teacher_actions = self.teacher(obs, stochastic_output=False)
    self.transition.actions = student_actions.detach()
    self.transition.privileged_actions = teacher_actions.detach()
    self.transition.observations = obs
    if self.rollout_policy == "teacher":
      return teacher_actions
    return student_actions

  def _perception_loss(
    self,
    student: DepthCoordinateStudentModel,
    observations: TensorDict,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_actor = cast(torch.Tensor, observations["teacher_actor"])
    target = teacher_actor[:, -FOOTBALL_HISTORY_DIM:]
    target_coordinates = target[:, :-BALL_VISIBILITY_HISTORY_DIM]
    target_visibility = target[:, -BALL_VISIBILITY_HISTORY_DIM:]
    prediction = student.predicted_football_history
    predicted_coordinates = prediction[:, :-BALL_VISIBILITY_HISTORY_DIM]

    coordinate_mask = torch.cat(
      (
        target_visibility.repeat_interleave(2, dim=-1),
        target_visibility.repeat_interleave(4, dim=-1),
      ),
      dim=-1,
    )
    position_error = F.smooth_l1_loss(
      predicted_coordinates,
      target_coordinates,
      reduction="none",
    )
    position_loss = (position_error * coordinate_mask).sum() / (
      coordinate_mask.sum().clamp_min(1.0)
    )
    visibility_loss = F.binary_cross_entropy_with_logits(
      student.visibility_logits,
      target_visibility,
    )
    return position_loss, visibility_loss

  def update(self) -> dict[str, float]:
    if not isinstance(self._raw_student, DepthCoordinateStudentModel):
      raise TypeError("BallPerceptionDistillation requires DepthCoordinateStudentModel")

    totals = {
      "behavior": 0.0,
      "football_position": 0.0,
      "football_visibility": 0.0,
    }
    count = 0
    for _ in range(self.num_learning_epochs):
      for batch in self.storage.generator():
        observations = batch.observations
        privileged_actions = batch.privileged_actions
        if observations is None or privileged_actions is None:
          raise RuntimeError("Distillation batch is missing Teacher supervision")
        student_actions = self.student(observations)
        behavior_loss = self.loss_fn(
          student_actions,
          privileged_actions,
        )
        position_loss, visibility_loss = self._perception_loss(
          self._raw_student,
          observations,
        )
        loss = (
          self.action_loss_coef * behavior_loss
          + self.position_loss_coef * position_loss
          + self.visibility_loss_coef * visibility_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
          self.reduce_parameters()
        if self.max_grad_norm:
          nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
        self.optimizer.step()

        totals["behavior"] += behavior_loss.item()
        totals["football_position"] += position_loss.item()
        totals["football_visibility"] += visibility_loss.item()
        count += 1

    self.storage.clear()
    if count == 0:
      raise RuntimeError("Distillation storage produced no training batches")
    return {name: value / count for name, value in totals.items()}
