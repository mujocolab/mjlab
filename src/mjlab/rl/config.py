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


@dataclass
class RslRlFlashSacActorCfg:
  """Config for the FlashSAC actor model."""

  num_blocks: int = 2
  """Number of residual blocks in the actor trunk."""
  hidden_dim: int = 128
  """Hidden dimension of the actor."""
  obs_normalization: bool = False
  """Whether to apply empirical observation normalization (the network also
  self-normalizes via BatchNorm, so this is off by default)."""
  log_std_min: float = -5.0
  """Lower bound for the (squashed) policy log-std."""
  log_std_max: float = 2.0
  """Upper bound for the (squashed) policy log-std."""
  class_name: str = "FlashSACActorModel"
  """Model class name resolved by RSL-RL."""


@dataclass
class RslRlFlashSacCriticCfg:
  """Config for the FlashSAC distributional double critic."""

  num_blocks: int = 2
  """Number of residual blocks per critic ensemble member."""
  hidden_dim: int = 256
  """Hidden dimension of the critic."""
  num_bins: int = 101
  """Number of atoms in the categorical (C51) value distribution."""
  min_v: float = -5.0
  """Lower bound of the value support. Should match ``-normalized_g_max``."""
  max_v: float = 5.0
  """Upper bound of the value support. Should match ``normalized_g_max``."""
  num_qs: int = 2
  """Number of Q-ensemble members (clipped double-Q uses 2)."""
  obs_normalization: bool = False
  """Whether to apply empirical observation normalization."""
  class_name: str = "FlashSACCriticModel"
  """Model class name resolved by RSL-RL."""


@dataclass
class RslRlReplayBufferCfg:
  """Config for the off-policy replay buffer."""

  capacity: int = 1_000_000
  """Maximum number of transitions stored."""
  min_length: int = 10_000
  """Minimum number of transitions before sampling/updates begin."""
  sample_batch_size: int = 2048
  """Mini-batch size drawn from the buffer per gradient step."""


@dataclass
class RslRlFlashSacAlgorithmCfg:
  """Config for the FlashSAC algorithm."""

  gamma: float = 0.99
  """The discount factor."""
  n_step: int = 1
  """Number of steps for n-step return accumulation."""
  learning_rate_init: float = 3e-4
  """Initial learning rate (start of warmup)."""
  learning_rate_peak: float = 3e-4
  """Peak learning rate (end of warmup)."""
  learning_rate_end: float = 1.5e-4
  """Final learning rate after cosine decay."""
  learning_rate_warmup_steps: int = 0
  """Number of linear warmup steps."""
  learning_rate_decay_steps: int = 1_000_000
  """Total schedule length (warmup + cosine decay), in gradient steps."""
  critic_target_update_tau: float = 0.01
  """EMA coefficient for the target critic update."""
  num_bins: int = 101
  """Number of atoms in the categorical TD target. Must match the critic."""
  min_v: float = -5.0
  """Lower bound of the value support. Must match the critic and ``-normalized_g_max``."""
  max_v: float = 5.0
  """Upper bound of the value support. Must match the critic and ``normalized_g_max``."""
  temp_initial_value: float = 0.01
  """Initial entropy-temperature value."""
  temp_target_sigma: float = 0.15
  """Target action std used to auto-compute the target entropy when
  ``temp_target_entropy`` is None."""
  temp_target_entropy: float | None = None
  """Target entropy. If None, it is auto-computed from the action dim and
  ``temp_target_sigma``."""
  actor_update_period: int = 2
  """Delayed policy update period (actor/temperature update every N critic updates)."""
  actor_bc_alpha: float = 0.0
  """Behavior-cloning regularization coefficient (0 disables it)."""
  actor_noise_zeta_mu: float = 2.0
  """Zeta-distribution exponent for action-noise repetition."""
  actor_noise_zeta_max: int = 16
  """Maximum noise-repetition length."""
  normalize_reward: bool = True
  """Whether to normalize rewards (required True in this version)."""
  normalized_g_max: float = 5.0
  """Return-normalization cap; also sets the critic value support magnitude."""
  use_amp: bool = False
  """Whether to use automatic mixed precision (must be False in this version)."""
  class_name: str = "FlashSAC"
  """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlOffPolicyRunnerCfg(RslRlBaseRunnerCfg):
  """Runner config for off-policy (FlashSAC) training.

  A drop-in sibling of :class:`RslRlOnPolicyRunnerCfg`: it reuses the base
  runner fields (``obs_groups``, ``num_steps_per_env``, ``save_interval``,
  ``clip_actions``, ...) and adds the FlashSAC model/algorithm/replay configs.
  These dataclass defaults are the single source of default hyperparameters;
  ``asdict`` materializes them into the plain dict RSL-RL consumes (fail-loud:
  RSL-RL substitutes no defaults of its own).
  """

  class_name: str = "OffPolicyRunner"
  """The runner class name."""
  updates_per_step: float = 1.0
  """Gradient updates per collected environment step (may be < 1.0)."""
  torch_compile_mode: str | None = None
  """torch.compile mode. Must be None in this version (eager-only)."""
  actor: RslRlFlashSacActorCfg = field(default_factory=RslRlFlashSacActorCfg)
  """The actor configuration."""
  critic: RslRlFlashSacCriticCfg = field(default_factory=RslRlFlashSacCriticCfg)
  """The critic configuration."""
  algorithm: RslRlFlashSacAlgorithmCfg = field(
    default_factory=RslRlFlashSacAlgorithmCfg
  )
  """The algorithm configuration."""
  replay: RslRlReplayBufferCfg = field(default_factory=RslRlReplayBufferCfg)
  """The replay-buffer configuration."""
