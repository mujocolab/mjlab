"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlFlashSacActorCfg,
  RslRlFlashSacAlgorithmCfg,
  RslRlFlashSacCriticCfg,
  RslRlModelCfg,
  RslRlOffPolicyRunnerCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
  RslRlReplayBufferCfg,
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
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
    experiment_name="g1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_flashsac_runner_cfg() -> RslRlOffPolicyRunnerCfg:
  """Create the FlashSAC (off-policy) runner configuration for the G1 velocity task."""
  return RslRlOffPolicyRunnerCfg(
    actor=RslRlFlashSacActorCfg(
      num_blocks=2,
      hidden_dim=512,
      obs_normalization=False,
    ),
    critic=RslRlFlashSacCriticCfg(
      num_blocks=2,
      hidden_dim=512,
      num_bins=101,
      min_v=-5.0,
      max_v=5.0,
      num_qs=2,
      obs_normalization=False,
    ),
    algorithm=RslRlFlashSacAlgorithmCfg(
      gamma=0.99,
      n_step=1,
      critic_target_update_tau=0.01,
      num_bins=101,
      min_v=-5.0,
      max_v=5.0,
      actor_update_period=2,
      normalize_reward=True,
      normalized_g_max=5.0,
      # Schedule length is expressed in gradient steps
      # (num_steps_per_env * updates_per_step * max_iterations).
      learning_rate_decay_steps=60_000,
    ),
    replay=RslRlReplayBufferCfg(
      capacity=1_000_000,
      min_length=10_000,
      sample_batch_size=2048,
    ),
    # Off-policy collection/update cadence. With ~512 envs this collects 512
    # transitions per iteration and takes `num_steps_per_env * updates_per_step`
    # gradient steps.
    num_steps_per_env=1,
    updates_per_step=2.0,
    experiment_name="g1_velocity_flashsac",
    save_interval=50,
    max_iterations=30_000,
  )
