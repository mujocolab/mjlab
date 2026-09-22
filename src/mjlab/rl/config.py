"""RSL-RL configuration."""

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple


@dataclass
class RslRlModelCfg:
  """Config for a single neural network model (Actor or Critic)."""

  hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the network."""
  activation: str = "elu"
  """The activation function."""
  obs_normalization: bool = False
  """Whether to normalize the observations. Default is False."""
  cnn_cfg: dict[str, Any] | None = None
  """CNN encoder config. When set, class_name should be "CNNModel".

  Passed to ``rsl_rl.modules.CNN``. Common keys: output_channels,
  kernel_size, stride, padding, activation, global_pool, max_pool.
  """
  distribution_cfg: dict[str, Any] | None = None
  """Distribution config dict passed to rsl_rl. Example::

    {"class_name": "GaussianDistribution",
     "init_std": 1.0, "std_type": "scalar"}

  ``None`` means deterministic output (use for critic).
  """
  rnn_type: str | None = None
  """RNN type ("lstm" or "gru"). When set, class_name should be "RNNModel"."""
  rnn_hidden_dim: int = 256
  """Hidden state dimension for the RNN."""
  rnn_num_layers: int = 1
  """Number of stacked RNN layers."""
  class_name: str = "MLPModel"
  """Model class name resolved by RSL-RL (MLPModel, CNNModel, or RNNModel)."""


@dataclass
class RslRlPpoAlgorithmCfg:
  """Config for the PPO algorithm."""

  num_learning_epochs: int = 5
  """The number of learning epochs per update."""
  num_mini_batches: int = 4
  """The number of mini-batches per update.
  mini batch size = num_envs * num_steps / num_mini_batches
  """
  learning_rate: float = 1e-3
  """The learning rate."""
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  """The learning rate schedule."""
  gamma: float = 0.99
  """The discount factor."""
  lam: float = 0.95
  """The lambda parameter for Generalized Advantage Estimation (GAE)."""
  entropy_coef: float = 0.005
  """The coefficient for the entropy loss."""
  desired_kl: float = 0.01
  """The desired KL divergence between the new and old policies."""
  max_grad_norm: float = 1.0
  """The maximum gradient norm for the policy."""
  value_loss_coef: float = 1.0
  """The coefficient for the value loss."""
  use_clipped_value_loss: bool = True
  """Whether to use clipped value loss."""
  clip_param: float = 0.2
  """The clipping parameter for the policy."""
  normalize_advantage_per_mini_batch: bool = False
  """Whether to normalize the advantage per mini-batch. Default is False. If True, the
  advantage is normalized over the mini-batches only. Otherwise, the advantage is
  normalized over the entire collected trajectories.
  """
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  """The optimizer to use."""
  share_cnn_encoders: bool = False
  """Share CNN encoders between actor and critic."""
  class_name: str = "PPO"
  """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlBaseRunnerCfg:
  seed: int = 42
  """The seed for the experiment. Default is 42."""
  num_steps_per_env: int = 24
  """The number of steps per environment update."""
  max_iterations: int = 300
  """The maximum number of iterations."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)},
  )
  save_interval: int = 50
  """The number of iterations between saves."""
  experiment_name: str = "exp1"
  """Directory name used to group runs under ``{log_root}/{experiment_name}/``.
  The log root defaults to ``logs/rsl_rl`` and can be overridden with
  ``--log-root`` on the CLI."""
  run_name: str = ""
  """Optional label appended to the timestamped run directory
  (e.g. ``2025-01-27_14-30-00_{run_name}``). Also becomes the
  display name for the run in wandb."""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""
  wandb_tags: Tuple[str, ...] = ()
  """Tags for the wandb run. Default is empty tuple."""
  resume: bool = False
  """Whether to resume the experiment. Default is False."""
  load_run: str = ".*"
  """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
  clip_actions: float | None = None
  """The clipping range for action values. If None (default), no clipping is applied."""
  upload_model: bool = True
  """Whether to upload model files (.pt, .onnx) to W&B on save. Set to
  False to keep metric logging but avoid storage usage. Default is True."""


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  """The runner class name. Default is OnPolicyRunner."""
  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      }
    )
  )
  """The actor configuration."""
  critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  """The critic configuration."""
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  """The algorithm configuration."""
  init_checkpoint: str | None = None
  """Optional checkpoint to initialize the actor from before training.

  Unlike ``resume``, only model weights are restored (no optimizer state or
  iteration count). Supports PPO checkpoints (``actor_state_dict``) and
  distillation checkpoints (``student_state_dict``), enabling RL fine-tuning
  of a distilled student policy. Distribution parameters (e.g. the action
  std) are intentionally not restored so the configured ``init_std``
  applies, following the reduced-initial-std recipe of arXiv:2505.11164.
  """


@dataclass
class RslRlCriticWarmupPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO with an initial critic-only warmup phase.

  For the first ``critic_warmup_updates`` updates the actor is frozen and
  only the critic (and normalizers) train. Used when fine-tuning a distilled
  policy with RL: the pre-trained policy would otherwise be destroyed by
  gradients from a randomly initialized value function (arXiv:2505.11164).
  """

  critic_warmup_updates: int = 0
  """Number of updates during which the actor parameters stay frozen."""
  class_name: str = "mjlab.rl.ppo:CriticWarmupPPO"


@dataclass
class RslRlDistillationAlgorithmCfg:
  """Config for DAgger-style teacher-student distillation.

  The student acts in the environment (sampling from its own action
  distribution for exploration noise) while the teacher labels every visited
  state; the student regresses onto the teacher actions. This is on-policy
  dataset aggregation (DAgger), not behavior cloning of teacher rollouts.
  """

  num_learning_epochs: int = 1
  """The number of learning epochs per update."""
  gradient_length: int = 15
  """Number of consecutive steps per backpropagation (truncated BPTT length
  for recurrent students). Should divide num_learning_epochs *
  num_steps_per_env, otherwise trailing steps are never backpropagated.
  """
  learning_rate: float = 1e-3
  """The learning rate for the student."""
  max_grad_norm: float | None = 1.0
  """The maximum gradient norm. None disables clipping."""
  loss_type: Literal["mse", "huber"] = "mse"
  """The regression loss between student and teacher actions."""
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  """The optimizer to use."""
  class_name: str = "Distillation"
  """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlMultiTeacherModelCfg:
  """Config for a multi-expert teacher (arXiv:2505.11164).

  Wraps several frozen expert models; each environment is labeled by the
  expert selected via an integer observation group (e.g. the terrain type),
  so a single student distills all experts at once.
  """

  teachers: Tuple[RslRlModelCfg, ...] = ()
  """One model config per expert. All experts share the teacher obs set."""
  assignment_group: str = "teacher_assignment"
  """Observation group holding the per-env expert index, shape (num_envs, 1).
  This group must exist in the environment observations but should not be
  listed in the runner's ``obs_groups``.
  """
  class_name: str = "mjlab.rl.multi_teacher:MultiTeacherModel"
  """Model class resolved by RSL-RL."""


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "DistillationRunner"
  """The runner class name."""
  student: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      }
    )
  )
  """The student configuration. Give the student a distribution so rollouts
  carry zero-mean exploration noise (the std receives no gradient from the
  distillation loss, so ``init_std`` sets a fixed noise scale)."""
  teacher: RslRlModelCfg | RslRlMultiTeacherModelCfg = field(
    default_factory=RslRlModelCfg
  )
  """The teacher configuration. Must match the architecture the teacher
  checkpoint was trained with (including its distribution config)."""
  algorithm: RslRlDistillationAlgorithmCfg = field(
    default_factory=RslRlDistillationAlgorithmCfg
  )
  """The algorithm configuration."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"student": ("student",), "teacher": ("teacher",)},
  )
  teacher_checkpoints: Tuple[str, ...] = ()
  """Checkpoints to load the teacher(s) from. One path for a single teacher,
  one per expert for a multi-teacher setup. PPO (``actor_state_dict``) and
  distillation (``student_state_dict``) checkpoints are supported."""
  inherit_env_state_from_teacher: bool = True
  """Restore the env's common_step_counter from the (first) teacher
  checkpoint so time-based curricula and randomization schedules start in
  their end-of-training state rather than from scratch."""
