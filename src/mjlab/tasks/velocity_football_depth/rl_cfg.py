"""PPO configurations for depth-image asymmetric football tasks."""

from dataclasses import asdict, dataclass, field
from typing import Literal

from mjlab.rl import (
  RslRlBaseRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.velocity_football.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

_DEPTH_MODEL_CLASS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
_DEPTH_CNN_CFG = {
  "output_channels": [16, 32, 64],
  "kernel_size": [5, 3, 3],
  "stride": [2, 2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "none",
  "spatial_softmax_temperature": 1.0,
}


def unitree_g1_depth_asymmetric_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Configure separate Actor and Critic depth encoders for initial training."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.actor.class_name = _DEPTH_MODEL_CLASS
  cfg.actor.cnn_cfg = _DEPTH_CNN_CFG.copy()
  cfg.critic.class_name = _DEPTH_MODEL_CLASS
  cfg.critic.cnn_cfg = _DEPTH_CNN_CFG.copy()
  cfg.algorithm.share_cnn_encoders = False
  cfg.obs_groups = {
    "actor": ("actor", "depth"),
    "critic": ("critic", "depth", "critic_ball"),
  }
  cfg.experiment_name = "g1_velocity_football_depth"
  return cfg


@dataclass
class BallAuxiliaryPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Configuration fields consumed by :class:`BallAuxiliaryPPO`."""

  auxiliary_target_group: str = "ball_aux_target"
  auxiliary_loss_coef: float = 1.0
  auxiliary_position_coef: float = 1.0
  auxiliary_visibility_coef: float = 0.2


def unitree_g1_depth_auxiliary_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Configure V1 temporal depth perception and its supervised PPO loss."""
  cfg = unitree_g1_ppo_runner_cfg()
  model_class = "mjlab.tasks.velocity_football_depth.model:DepthAuxCNNModel"
  model_cfg = {"output_channels": (16, 32, 64), "latent_dim": 128}
  cfg.actor.class_name = model_class
  cfg.actor.cnn_cfg = model_cfg
  cfg.critic.class_name = model_class
  cfg.critic.cnn_cfg = model_cfg.copy()

  algorithm_cfg = asdict(cfg.algorithm)
  algorithm_cfg["class_name"] = (
    "mjlab.tasks.velocity_football_depth.algorithm:BallAuxiliaryPPO"
  )
  cfg.algorithm = BallAuxiliaryPpoAlgorithmCfg(**algorithm_cfg)
  cfg.obs_groups = {
    "actor": ("actor", "depth"),
    "critic": ("critic", "depth", "critic_ball"),
  }
  cfg.experiment_name = "g1_velocity_football_depth_auxiliary"
  return cfg


@dataclass
class BallPerceptionDistillationAlgorithmCfg:
  """Configuration for coordinate-policy depth distillation."""

  class_name: str = (
    "mjlab.tasks.velocity_football_depth.distillation:BallPerceptionDistillation"
  )
  num_learning_epochs: int = 1
  gradient_length: int = 1
  learning_rate: float = 3.0e-4
  max_grad_norm: float = 1.0
  loss_type: Literal["mse", "huber"] = "huber"
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  action_loss_coef: float = 1.0
  position_loss_coef: float = 1.0
  visibility_loss_coef: float = 0.2
  rollout_policy: Literal["teacher", "student"] = "teacher"
  rnd_cfg: None = None
  symmetry_cfg: None = None


@dataclass
class TeacherRolloutDistillationAlgorithmCfg:
  """Pure action behavior cloning with Teacher-controlled rollouts."""

  class_name: str = (
    "mjlab.tasks.velocity_football_depth.distillation:TeacherRolloutDistillation"
  )
  num_learning_epochs: int = 1
  gradient_length: int = 1
  learning_rate: float = 3.0e-4
  max_grad_norm: float = 1.0
  loss_type: Literal["mse", "huber"] = "huber"
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  rollout_policy: Literal["teacher", "student", "mixed"] = "teacher"
  student_rollout_warmup_updates: int = 0
  student_rollout_ramp_updates: int = 1
  student_rollout_final_probability: float = 1.0
  rnd_cfg: None = None
  symmetry_cfg: None = None


@dataclass
class ConstrainedLatentDistillationAlgorithmCfg(TeacherRolloutDistillationAlgorithmCfg):
  """Constrained fine-tuning of depth latent and final control layer."""

  class_name: str = (
    "mjlab.tasks.velocity_football_depth.distillation:ConstrainedLatentDistillation"
  )
  latent_loss_coef: float = 0.1
  mlp_anchor_loss_coef: float = 1.0e-3
  mlp_learning_rate: float = 1.0e-5


@dataclass
class BallPerceptionDistillationRunnerCfg(RslRlBaseRunnerCfg):
  """Student/Teacher configuration consumed by RSL-RL Distillation."""

  class_name: str = "DistillationRunner"
  student: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      class_name=(
        "mjlab.tasks.velocity_football_depth.distillation:DepthCoordinateStudentModel"
      ),
      cnn_cfg={
        "output_channels": (16, 32, 64),
        "latent_dim": 128,
        "freeze_coordinate_actor": True,
      },
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    )
  )
  teacher: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      class_name="MLPModel",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    )
  )
  algorithm: BallPerceptionDistillationAlgorithmCfg = field(
    default_factory=BallPerceptionDistillationAlgorithmCfg
  )


@dataclass
class TemporalTeacherDistillationRunnerCfg(BallPerceptionDistillationRunnerCfg):
  """Runner schema for direct depth-latent behavior cloning."""

  algorithm: TeacherRolloutDistillationAlgorithmCfg = field(
    default_factory=TeacherRolloutDistillationAlgorithmCfg
  )


@dataclass
class ConstrainedLatentDistillationRunnerCfg(TemporalTeacherDistillationRunnerCfg):
  """Runner schema for constrained direct-latent fine-tuning."""

  algorithm: ConstrainedLatentDistillationAlgorithmCfg = field(
    default_factory=ConstrainedLatentDistillationAlgorithmCfg
  )


def unitree_g1_depth_teacher_distillation_runner_cfg() -> (
  BallPerceptionDistillationRunnerCfg
):
  """Train only the depth front-end against the frozen coordinate Teacher."""
  return BallPerceptionDistillationRunnerCfg(
    seed=42,
    num_steps_per_env=24,
    max_iterations=10_000,
    obs_groups={
      "student": ("teacher_actor",),
      "teacher": ("teacher_actor",),
    },
    save_interval=500,
    experiment_name="g1_velocity_football_depth_distillation",
    upload_model=False,
  )


def unitree_g1_depth_temporal_teacher_distillation_runner_cfg() -> (
  BallPerceptionDistillationRunnerCfg
):
  """Distill the frozen B1 coordinate TemporalCNN into a depth Student."""
  temporal_model = (
    "mjlab.tasks.velocity_football.rl.temporal_cnn_model:TemporalCNNModel"
  )
  temporal_cnn_cfg = {
    "output_channels": (64, 64, 64),
    "kernel_size": 3,
    "activation": "elu",
    "dilations": (1, 2, 4),
    "causal": True,
    "output_mode": "last",
  }
  cfg = TemporalTeacherDistillationRunnerCfg(
    seed=42,
    num_steps_per_env=24,
    max_iterations=10_000,
    obs_groups={
      "student": ("actor",),
      "teacher": ("actor", "actor_history"),
    },
    save_interval=500,
    experiment_name="g1_velocity_football_depth_temporal_distillation",
    upload_model=False,
  )
  cfg.teacher.class_name = temporal_model
  cfg.teacher.cnn_cfg = temporal_cnn_cfg.copy()
  cfg.student.class_name = (
    "mjlab.tasks.velocity_football_depth.distillation:DepthTemporalLatentStudentModel"
  )
  cfg.student.cnn_cfg = {
    "output_channels": (16, 32, 64),
    "latent_dim": 64,
    "freeze_coordinate_actor": False,
  }
  return cfg


def unitree_g1_depth_temporal_deployment_robust_v2_runner_cfg() -> (
  TemporalTeacherDistillationRunnerCfg
):
  """Distill with a Teacher-to-Student DAgger-style rollout schedule."""
  cfg = unitree_g1_depth_temporal_teacher_distillation_runner_cfg()
  assert isinstance(cfg.algorithm, TeacherRolloutDistillationAlgorithmCfg)
  cfg.algorithm.rollout_policy = "mixed"
  cfg.algorithm.student_rollout_warmup_updates = 1_000
  cfg.algorithm.student_rollout_ramp_updates = 3_000
  cfg.algorithm.student_rollout_final_probability = 1.0
  return cfg


def unitree_g1_depth_temporal_calibrated_frozen_mlp_runner_cfg() -> (
  TemporalTeacherDistillationRunnerCfg
):
  """Run V2 mixed rollout while preserving the coordinate Teacher backbone."""
  cfg = unitree_g1_depth_temporal_deployment_robust_v2_runner_cfg()
  assert cfg.student.cnn_cfg is not None
  cfg.student.cnn_cfg["freeze_coordinate_actor"] = True
  cfg.max_iterations = 10_000
  return cfg


def unitree_g1_depth_temporal_constrained_latent_runner_cfg() -> (
  ConstrainedLatentDistillationRunnerCfg
):
  """Fine-tune strong visual DR without allowing the walking MLP to drift."""
  base = unitree_g1_depth_temporal_teacher_distillation_runner_cfg()
  cfg = ConstrainedLatentDistillationRunnerCfg(
    **{
      **vars(base),
      "algorithm": ConstrainedLatentDistillationAlgorithmCfg(),
    }
  )
  assert cfg.student.cnn_cfg is not None
  cfg.student.cnn_cfg["freeze_coordinate_actor"] = False
  cfg.student.cnn_cfg["train_mlp_last_layer_only"] = True
  cfg.algorithm.rollout_policy = "mixed"
  cfg.algorithm.student_rollout_warmup_updates = 0
  cfg.algorithm.student_rollout_ramp_updates = 2_000
  cfg.algorithm.student_rollout_final_probability = 0.3
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.algorithm.mlp_learning_rate = 1.0e-5
  cfg.algorithm.latent_loss_coef = 0.1
  cfg.algorithm.mlp_anchor_loss_coef = 1.0e-3
  cfg.max_iterations = 10_000
  return cfg


def unitree_g1_depth_student_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Low-learning-rate PPO fine-tuning for a distilled depth Student."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.actor.class_name = (
    "mjlab.tasks.velocity_football_depth.distillation:DepthCoordinateStudentModel"
  )
  cfg.actor.cnn_cfg = {
    "output_channels": (16, 32, 64),
    "latent_dim": 128,
    "freeze_coordinate_actor": False,
  }
  cfg.critic.class_name = "MLPModel"
  cfg.critic.cnn_cfg = None
  cfg.obs_groups = {
    "actor": ("teacher_actor",),
    "critic": ("critic",),
  }
  cfg.algorithm.learning_rate = 1.0e-4
  cfg.experiment_name = "g1_velocity_football_depth_student_ppo"
  return cfg
