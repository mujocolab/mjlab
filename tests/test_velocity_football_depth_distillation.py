"""Tests for coordinate-teacher depth perception distillation."""

from typing import Any, cast

import torch
from tensordict import TensorDict

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity_football_depth import (
  DISTILLATION_TASK_ID,
  STUDENT_PPO_TASK_ID,
  TEMPORAL_DISTILLATION_TASK_ID,
  TEMPORAL_MOUNT_RANGE_STRONG_CONSTRAINED_DISTILLATION_TASK_ID,
)
from mjlab.tasks.velocity_football_depth.distillation import (
  FOOTBALL_HISTORY_DIM,
  PROPRIO_HISTORY_DIM,
  TEMPORAL_FOOTBALL_DIM,
  TEMPORAL_FOOTBALL_HISTORY_LENGTH,
  BallPerceptionDistillation,
  DepthCoordinateStudentModel,
  DepthTemporalLatentStudentModel,
)
from mjlab.tasks.velocity_football_depth.runner import (
  DepthStudentPpoRunner,
  DepthTeacherDistillationRunner,
)


def _make_observations(batch_size: int = 4) -> TensorDict:
  return TensorDict(
    {
      "teacher_actor": torch.randn(batch_size, 525),
      "student_proprio": torch.randn(batch_size, PROPRIO_HISTORY_DIM),
      "depth": torch.rand(batch_size, 5, 30, 40),
    },
    batch_size=[batch_size],
  )


def _make_student(batch_size: int = 4) -> DepthCoordinateStudentModel:
  return DepthCoordinateStudentModel(
    obs=_make_observations(batch_size),
    obs_groups={"student": ["teacher_actor"]},
    obs_set="student",
    output_dim=29,
    hidden_dims=(512, 256, 128),
    obs_normalization=True,
    cnn_cfg={
      "output_channels": (8, 16, 32),
      "latent_dim": 32,
      "freeze_coordinate_actor": True,
    },
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def _make_temporal_observations(batch_size: int = 4) -> TensorDict:
  return TensorDict(
    {
      "actor": torch.randn(batch_size, PROPRIO_HISTORY_DIM),
      "actor_history": torch.randn(
        batch_size,
        TEMPORAL_FOOTBALL_HISTORY_LENGTH,
        TEMPORAL_FOOTBALL_DIM,
      ),
      "student_proprio": torch.randn(batch_size, PROPRIO_HISTORY_DIM),
      "depth": torch.rand(
        batch_size,
        TEMPORAL_FOOTBALL_HISTORY_LENGTH,
        30,
        40,
      ),
    },
    batch_size=[batch_size],
  )


def _make_temporal_student(
  batch_size: int = 4,
) -> DepthTemporalLatentStudentModel:
  return DepthTemporalLatentStudentModel(
    obs=_make_temporal_observations(batch_size),
    obs_groups={"student": ["actor"]},
    obs_set="student",
    output_dim=29,
    hidden_dims=(512, 256, 128),
    obs_normalization=True,
    cnn_cfg={
      "output_channels": (8, 16, 32),
      "latent_dim": 64,
      "freeze_coordinate_actor": True,
    },
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def test_depth_distillation_tasks_expose_teacher_student_contracts() -> None:
  env_cfg = load_env_cfg(DISTILLATION_TASK_ID)
  distill_cfg = cast(Any, load_rl_cfg(DISTILLATION_TASK_ID))
  ppo_cfg = cast(Any, load_rl_cfg(STUDENT_PPO_TASK_ID))

  assert env_cfg.observations["teacher_actor"].history_length == 5
  assert env_cfg.observations["student_proprio"].history_length == 5
  assert tuple(env_cfg.observations["student_proprio"].terms)[-1] == "actions"
  assert env_cfg.observations["depth"].history_length == 5
  assert not env_cfg.observations["depth"].flatten_history_dim
  assert "actor" not in env_cfg.observations
  assert "critic_history" not in env_cfg.observations

  assert distill_cfg.obs_groups == {
    "student": ("teacher_actor",),
    "teacher": ("teacher_actor",),
  }
  assert distill_cfg.algorithm.rollout_policy == "teacher"
  assert distill_cfg.student.cnn_cfg["freeze_coordinate_actor"]
  assert ppo_cfg.algorithm.learning_rate == 1.0e-4
  assert not ppo_cfg.actor.cnn_cfg["freeze_coordinate_actor"]
  assert ppo_cfg.obs_groups["critic"] == ("critic",)
  assert load_runner_cls(DISTILLATION_TASK_ID) is DepthTeacherDistillationRunner
  assert load_runner_cls(STUDENT_PPO_TASK_ID) is DepthStudentPpoRunner


def test_temporal_teacher_distillation_contract_has_no_student_coordinates() -> None:
  env_cfg = load_env_cfg(TEMPORAL_DISTILLATION_TASK_ID)
  distill_cfg = cast(Any, load_rl_cfg(TEMPORAL_DISTILLATION_TASK_ID))

  assert env_cfg.observations["actor"].history_length == 5
  assert env_cfg.observations["actor_history"].history_length == 10
  assert env_cfg.observations["depth"].history_length == 10
  assert env_cfg.observations["student_proprio"].history_length == 5
  assert set(env_cfg.observations["student_proprio"].terms).isdisjoint(
    {"ball_pos_b", "ball_to_feet_vectors_b", "ball_visible_mask"}
  )
  assert distill_cfg.student.class_name.endswith("DepthTemporalLatentStudentModel")
  assert distill_cfg.obs_groups["student"] == ("actor",)
  assert distill_cfg.obs_groups["teacher"] == ("actor", "actor_history")
  assert distill_cfg.algorithm.class_name.endswith("TeacherRolloutDistillation")
  assert load_runner_cls(TEMPORAL_DISTILLATION_TASK_ID) is (
    DepthTeacherDistillationRunner
  )


def test_constrained_latent_distillation_configuration() -> None:
  cfg = cast(
    Any,
    load_rl_cfg(TEMPORAL_MOUNT_RANGE_STRONG_CONSTRAINED_DISTILLATION_TASK_ID),
  )

  assert cfg.algorithm.class_name.endswith("ConstrainedLatentDistillation")
  assert cfg.algorithm.rollout_policy == "mixed"
  assert cfg.algorithm.student_rollout_warmup_updates == 0
  assert cfg.algorithm.student_rollout_ramp_updates == 2_000
  assert cfg.algorithm.student_rollout_final_probability == 0.3
  assert cfg.algorithm.learning_rate == 3.0e-4
  assert cfg.algorithm.mlp_learning_rate == 1.0e-5
  assert cfg.algorithm.latent_loss_coef == 0.1
  assert cfg.algorithm.mlp_anchor_loss_coef == 1.0e-3
  assert cfg.student.cnn_cfg["freeze_coordinate_actor"] is False
  assert cfg.student.cnn_cfg["train_mlp_last_layer_only"] is True


def test_temporal_student_can_train_only_final_mlp_layer() -> None:
  observations = _make_temporal_observations(batch_size=2)
  student = DepthTemporalLatentStudentModel(
    obs=observations,
    obs_groups={"student": ["actor"]},
    obs_set="student",
    output_dim=29,
    hidden_dims=(512, 256, 128),
    obs_normalization=True,
    cnn_cfg={
      "output_channels": (8, 16, 32),
      "latent_dim": 64,
      "freeze_coordinate_actor": False,
      "train_mlp_last_layer_only": True,
    },
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )

  final_ids = {id(parameter) for parameter in student.last_mlp_linear.parameters()}
  assert all(
    parameter.requires_grad for parameter in student.depth_encoder.parameters()
  )
  assert all(
    parameter.requires_grad for parameter in student.last_mlp_linear.parameters()
  )
  assert all(
    not parameter.requires_grad
    for parameter in student.mlp.parameters()
    if id(parameter) not in final_ids
  )
  assert all(
    not parameter.requires_grad for parameter in student.distribution.parameters()
  )


def test_depth_student_reconstructs_teacher_input_and_freezes_coordinate_actor() -> (
  None
):
  observations = _make_observations()
  student = _make_student()
  actions = student(observations)

  assert actions.shape == (4, 29)
  assert student.predicted_football_history.shape == (4, FOOTBALL_HISTORY_DIM)
  assert student.visibility_logits.shape == (4, 5)
  assert all(parameter.requires_grad for parameter in student.perception.parameters())
  frozen = [
    parameter.requires_grad
    for name, parameter in student.named_parameters()
    if not name.startswith("perception.")
  ]
  assert frozen and not any(frozen)


def test_depth_student_export_matches_deterministic_model() -> None:
  observations = _make_observations(batch_size=2)
  student = _make_student(batch_size=2)
  student.eval()
  exported = student.as_onnx()

  with torch.inference_mode():
    expected = student(observations)
    actual = exported(observations["student_proprio"], observations["depth"])
  torch.testing.assert_close(actual, expected)
  assert exported.input_names == ["proprio", "depth"]


def test_perception_loss_uses_visible_coordinate_history() -> None:
  observations = _make_observations(batch_size=3)
  target = observations["teacher_actor"]
  target[:, -FOOTBALL_HISTORY_DIM:] = 0.0
  target[:, -5:] = 1.0
  student = _make_student(batch_size=3)
  student(observations)
  algorithm = object.__new__(BallPerceptionDistillation)

  position, visibility = algorithm._perception_loss(student, observations)

  assert position.ndim == 0 and torch.isfinite(position)
  assert visibility.ndim == 0 and torch.isfinite(visibility)


def test_temporal_depth_student_ignores_direct_coordinate_observations() -> None:
  observations = _make_temporal_observations(batch_size=2)
  student = _make_temporal_student(batch_size=2)
  student.eval()
  exported = student.as_onnx()

  with torch.inference_mode():
    expected = student(observations, stochastic_output=False)
    observations["actor"] = torch.full_like(observations["actor"], 999.0)
    observations["actor_history"] = torch.full_like(
      observations["actor_history"], 999.0
    )
    actual = student(observations, stochastic_output=False)
    exported_actions = exported(observations["student_proprio"], observations["depth"])

  torch.testing.assert_close(actual, expected)
  torch.testing.assert_close(exported_actions, expected)
  assert student.mlp[0].in_features == 554
  assert student.depth_encoder.encode(observations["depth"]).shape == (2, 64)
  assert not hasattr(student, "predicted_football_history")
  assert not any(name.startswith("cnn_encoders") for name, _ in student.named_modules())
  assert exported.input_names == ["proprio", "depth"]
