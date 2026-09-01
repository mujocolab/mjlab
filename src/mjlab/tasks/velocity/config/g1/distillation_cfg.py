"""Distillation and RL fine-tuning configs for Unitree G1 velocity task.

Stage 1 (``Mjlab-Velocity-Rough-Unitree-G1-Distill``): distill the rough
teacher (trained with a terrain height scan) into a blind recurrent student
via DAgger-style teacher-student distillation.

Stage 2 (``Mjlab-Velocity-Rough-Unitree-G1-Distill-Finetune``): RL
fine-tune the distilled student with asymmetric PPO (privileged critic),
using a critic warmup phase and a reduced initial action std, following
arXiv:2505.11164.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import (
  RslRlCriticWarmupPpoAlgorithmCfg,
  RslRlDistillationAlgorithmCfg,
  RslRlDistillationRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
)
from mjlab.tasks.velocity.distillation_env_cfg import (
  to_distillation_env_cfg,
  to_finetune_env_cfg,
)

from .env_cfgs import unitree_g1_rough_env_cfg
from .rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_distillation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return to_distillation_env_cfg(unitree_g1_rough_env_cfg(play=play))


def unitree_g1_finetune_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return to_finetune_env_cfg(unitree_g1_rough_env_cfg(play=play))


def _student_model_cfg(init_std: float) -> RslRlModelCfg:
  """Recurrent student: memory is needed to infer terrain without the scan."""
  return RslRlModelCfg(
    class_name="RNNModel",
    rnn_type="lstm",
    rnn_hidden_dim=256,
    rnn_num_layers=1,
    hidden_dims=(256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": init_std,
      "std_type": "scalar",
    },
  )


def unitree_g1_distillation_runner_cfg() -> RslRlDistillationRunnerCfg:
  """Distillation runner config for the G1 velocity task.

  The teacher config must mirror the actor of
  :func:`unitree_g1_ppo_runner_cfg` so rough-task checkpoints load strictly.
  Pass the teacher checkpoint via ``--agent.teacher-checkpoints <path>``.
  """
  teacher = unitree_g1_ppo_runner_cfg().actor
  return RslRlDistillationRunnerCfg(
    # init_std sets the fixed zero-mean exploration noise added to student
    # actions during data collection (it receives no gradient).
    student=_student_model_cfg(init_std=0.5),
    teacher=teacher,
    algorithm=RslRlDistillationAlgorithmCfg(
      num_learning_epochs=1,
      # One BPTT window per rollout: 24 steps at dt=0.02*decimation.
      gradient_length=24,
      learning_rate=1e-3,
      max_grad_norm=1.0,
      loss_type="mse",
    ),
    experiment_name="g1_velocity_distill",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )


def unitree_g1_finetune_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO fine-tuning config for the distilled G1 student.

  Point ``--agent.init-checkpoint`` at a distillation checkpoint. The actor
  must match the student architecture; the critic trains from scratch on
  the privileged observations, so the actor is frozen for the first
  ``critic_warmup_updates`` updates and starts with a reduced action std.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=_student_model_cfg(init_std=0.2),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlCriticWarmupPpoAlgorithmCfg(
      critic_warmup_updates=50,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      # Conservative fine-tuning: fixed, small learning rate so the
      # distilled behavior is not destroyed early on.
      learning_rate=1.0e-4,
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("student",), "critic": ("critic",)},
    experiment_name="g1_velocity_distill_finetune",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
