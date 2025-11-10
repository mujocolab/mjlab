"""RL configuration for Unitree Go1 velocity task."""

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)

UNITREE_GO1_PPO_RUNNER_CFG = RslRlOnPolicyRunnerCfg(
  policy=RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_obs_normalization=False,
    critic_obs_normalization=False,
    actor_hidden_dims=(512, 256, 128),
    critic_hidden_dims=(512, 256, 128),
    activation="elu",
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
  experiment_name="go1_velocity",
  save_interval=50,
  num_steps_per_env=24,
  max_iterations=10_000,
)
