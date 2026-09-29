"""Task-specific PPO parameters for the native Gymnasium MuJoCo v5 tasks.

Batching overrides and parameter provenance are documented in
``docs/source/gym_mujoco.rst``. All entries configure v5 environments directly.
"""

from dataclasses import dataclass
from math import exp

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass(frozen=True)
class PpoParameters:
  learning_rate: float = 3e-4
  gamma: float = 0.99
  gae_lambda: float = 0.95
  ent_coef: float = 0.0
  clip_range: float = 0.2
  max_grad_norm: float = 0.5
  vf_coef: float = 0.5
  hidden_dims: tuple[int, ...] = (64, 64)
  activation: str = "tanh"
  log_std_init: float = 0.0
  normalize_observations: bool = True


TASK_PARAMETERS = {
  "Ant-v5": PpoParameters(),
  "HalfCheetah-v5": PpoParameters(
    learning_rate=2.0633e-5,
    gamma=0.98,
    gae_lambda=0.92,
    ent_coef=0.000401762,
    clip_range=0.1,
    max_grad_norm=0.8,
    vf_coef=0.58096,
    hidden_dims=(256, 256),
    activation="relu",
    log_std_init=-2,
  ),
  "Hopper-v5": PpoParameters(
    learning_rate=9.80828e-5,
    gamma=0.999,
    gae_lambda=0.99,
    ent_coef=0.00229519,
    clip_range=0.2,
    max_grad_norm=0.7,
    vf_coef=0.835671,
    hidden_dims=(256, 256),
    activation="relu",
    log_std_init=-2,
  ),
  "Humanoid-v5": PpoParameters(
    learning_rate=3.56987e-5,
    # Validated in continuation; see the documented warm-start qualification.
    gamma=0.98,
    gae_lambda=0.9,
    ent_coef=0.00238306,
    clip_range=0.3,
    max_grad_norm=2,
    vf_coef=0.431892,
    hidden_dims=(256, 256),
    activation="relu",
    log_std_init=-2,
  ),
  "HumanoidStandup-v5": PpoParameters(
    learning_rate=2.55673e-5,
    gamma=0.99,
    gae_lambda=0.9,
    ent_coef=3.62109e-6,
    clip_range=0.3,
    max_grad_norm=0.7,
    vf_coef=0.430793,
    hidden_dims=(256, 256),
    activation="relu",
    log_std_init=-2,
  ),
  "InvertedDoublePendulum-v5": PpoParameters(
    learning_rate=0.000155454,
    gamma=0.98,
    gae_lambda=0.8,
    ent_coef=1.05057e-6,
    clip_range=0.4,
    max_grad_norm=0.5,
    vf_coef=0.695929,
  ),
  "InvertedPendulum-v5": PpoParameters(
    learning_rate=0.000222425,
    gamma=0.999,
    gae_lambda=0.9,
    ent_coef=1.37976e-7,
    clip_range=0.4,
    max_grad_norm=0.3,
    vf_coef=0.19816,
  ),
  "Pusher-v5": PpoParameters(
    gamma=0.98,
    gae_lambda=1.0,
    ent_coef=0.01,
    hidden_dims=(256, 256),
    activation="relu",
  ),
  "Reacher-v5": PpoParameters(
    learning_rate=0.000104019,
    gamma=0.98,
    gae_lambda=1.0,
    ent_coef=0.001,
    clip_range=0.3,
    max_grad_norm=0.9,
    vf_coef=0.950368,
  ),
  "Swimmer-v5": PpoParameters(
    learning_rate=6e-4,
    gamma=0.9999,
    gae_lambda=0.98,
    ent_coef=0.01,
  ),
  "Walker2d-v5": PpoParameters(
    learning_rate=5.05041e-5,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.000585045,
    clip_range=0.1,
    max_grad_norm=1,
    vf_coef=0.871923,
  ),
}


def gym_mujoco_ppo_cfg(task_id: str) -> RslRlOnPolicyRunnerCfg:
  """Batched Warp PPO with task-specific training parameters.

  Network/scalar settings are explicit v5 task parameters. Batching and the
  iteration budget are chosen for mjlab. Task overrides are below; evaluated
  recipes and warm-start qualifications are documented separately.
  Changing environment count changes minibatch size and total transitions; the
  native CLI exposes num_steps_per_env, num_mini_batches and max_iterations too.
  """
  gym_id = task_id.removeprefix("Mjlab-Gym-")
  params = TASK_PARAMETERS[gym_id]
  cfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=params.hidden_dims,
      activation=params.activation,
      obs_normalization=params.normalize_observations,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": exp(params.log_std_init),
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=params.hidden_dims,
      activation=params.activation,
      obs_normalization=params.normalize_observations,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=8,
      learning_rate=params.learning_rate,
      schedule="fixed",
      gamma=params.gamma,
      lam=params.gae_lambda,
      entropy_coef=params.ent_coef,
      max_grad_norm=params.max_grad_norm,
      value_loss_coef=params.vf_coef,
      use_clipped_value_loss=False,
      clip_param=params.clip_range,
      normalize_advantage_per_mini_batch=True,
    ),
    num_steps_per_env=32,
    max_iterations=3000 if gym_id.startswith("Humanoid") else 1000,
    experiment_name=f"gym_{gym_id.lower().replace('-', '_')}",
    logger="tensorboard",
  )
  if gym_id in ("Hopper-v5", "Humanoid-v5", "Swimmer-v5", "Walker2d-v5"):
    # Longer trajectories and more optimization work avoid the short-rollout
    # plateau. Hopper/Humanoid/Walker2d use 2048 x 128 samples, 32 minibatches
    # of 8192 and 10 epochs; Swimmer overrides the optimization batch below.
    cfg.num_steps_per_env = 128
    cfg.algorithm.num_mini_batches = 32
    cfg.algorithm.num_learning_epochs = 10
    cfg.algorithm.schedule = "adaptive"
    cfg.max_iterations = 1000 if gym_id == "Humanoid-v5" else 500
    if gym_id == "Swimmer-v5":
      # Validated from scratch with 4096 environments and long rollouts.
      cfg.algorithm.num_learning_epochs = 5
      cfg.algorithm.num_mini_batches = 4
      cfg.max_iterations = 300
  elif gym_id == "InvertedDoublePendulum-v5":
    cfg.algorithm.schedule = "adaptive"
  elif gym_id in ("Reacher-v5", "Pusher-v5"):
    # Task-specific exploration profiles evaluated with the native PPO runner.
    cfg.num_steps_per_env = 128
    cfg.algorithm.num_mini_batches = 4
    cfg.algorithm.schedule = "adaptive"
    if gym_id == "Reacher-v5":
      cfg.clip_actions = 1.0
      cfg.max_iterations = 300
    else:
      cfg.clip_actions = 2.0
      cfg.max_iterations = 600
  return cfg
