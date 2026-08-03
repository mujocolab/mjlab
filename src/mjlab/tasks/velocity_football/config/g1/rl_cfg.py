"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner configuration for G1 football training."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity_football",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_temporal_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the TemporalCNN runner configuration for masked ball vision."""
  cfg = unitree_g1_ppo_runner_cfg()
  temporal_model = (
    "mjlab.tasks.velocity_football.rl.temporal_cnn_model:TemporalCNNModel"
  )
  cnn_cfg = {
    "output_channels": (256, 128, 64),
    "kernel_size": 3,
    "activation": "elu",
    "global_pool": "avg",
  }
  cfg.actor.class_name = temporal_model
  cfg.actor.cnn_cfg = cnn_cfg
  cfg.critic.class_name = temporal_model
  cfg.critic.cnn_cfg = cnn_cfg.copy()
  cfg.obs_groups = {
    "actor": ("actor", "actor_history"),
    "critic": ("critic", "critic_history"),
  }
  return cfg


def unitree_g1_visual_mask_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Current-frame Actor control with the same temporal Critic as the test arm."""
  cfg = unitree_g1_temporal_ppo_runner_cfg()
  cfg.actor.class_name = "MLPModel"
  cfg.actor.cnn_cfg = None
  cfg.obs_groups["actor"] = ("actor",)
  return cfg


def unitree_g1_factorial_ppo_runner_cfg(
  *,
  use_b1_history: bool,
) -> RslRlOnPolicyRunnerCfg:
  """Create the frozen A0/A1 runner while keeping one shared temporal Critic."""
  cfg = unitree_g1_ppo_runner_cfg()
  temporal_model = (
    "mjlab.tasks.velocity_football.rl.temporal_cnn_model:TemporalCNNModel"
  )
  cfg.critic.class_name = temporal_model
  cfg.critic.cnn_cfg = {
    "output_channels": (256, 128, 64),
    "kernel_size": 3,
    "activation": "elu",
    "global_pool": "avg",
  }
  cfg.obs_groups = {
    "actor": ("actor",),
    "critic": ("critic", "critic_history"),
  }
  if use_b1_history:
    cfg.actor.class_name = temporal_model
    cfg.actor.cnn_cfg = {
      "output_channels": (64, 64, 64),
      "kernel_size": 3,
      "activation": "elu",
      "dilations": (1, 2, 4),
      "causal": True,
      "output_mode": "last",
    }
    cfg.obs_groups["actor"] = ("actor", "actor_history")
  return cfg


def unitree_g1_temporal_velocity_pretrain_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """TemporalCNN configuration for the matched walking pretraining stage."""
  cfg = unitree_g1_temporal_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_football_pretrain"
  return cfg


def unitree_g1_velocity_pretrain_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner configuration for football-compatible walking."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_velocity_football_pretrain"
  return cfg
