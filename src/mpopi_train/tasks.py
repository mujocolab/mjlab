"""Tasks of the MPOPI experiments, registered in mjlab's task registry.

``Mpopi-G1-2k-<method>`` is the flat G1 velocity task with a command curriculum
that reaches 1.5 m/s within 2000 iterations, trained with one of the methods in
:mod:`mpopi_train.presets`. Importing this module registers the tasks.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mpopi_train.algorithms.config import MpopiCfg
from mpopi_train.config import MpopiRunnerCfg, with_mpopi
from mpopi_train.presets import METHODS
from mpopi_train.runner import MpopiVelocityRunner

G1_ITERATIONS = 2_000
"""PPO iterations of the G1 2k tasks."""
G1_MAX_SPEED = 1.5
"""Top forward speed (m/s) commanded by the last curriculum stage."""
G1_STAGE_ITERATION = 500
"""Iteration at which the curriculum starts commanding ``G1_MAX_SPEED``."""
G1_EXPERIMENT = "g1_velocity_2k"


def g1_2k_env_cfg(
  play: bool = False, steps_per_iteration: int | None = None
) -> ManagerBasedRlEnvCfg:
  """Flat G1 velocity task that reaches its final speed range in 2000 iterations.

  mjlab's curriculum only adds speeds above 1 m/s after 5000 iterations and
  targets 3 m/s. Here the forward range grows from (-1, 1) to (-1, 1.5) m/s at
  iteration 500, so a 2000-iteration run spends 1500 iterations on the target.

  The curriculum counts env steps, so the stage is placed at
  ``G1_STAGE_ITERATION * steps_per_iteration``; by default the steps per
  iteration of the G1 runner. Overriding ``num_steps_per_env`` on the command
  line does not move it: set the stage ``step`` too.
  """
  if steps_per_iteration is None:
    steps_per_iteration = unitree_g1_ppo_runner_cfg().num_steps_per_env
  cfg = unitree_g1_flat_env_cfg(play=play)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  if play:
    twist_cmd.ranges.lin_vel_x = (-1.0, G1_MAX_SPEED)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)
    return cfg
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
    {
      "step": G1_STAGE_ITERATION * steps_per_iteration,
      "lin_vel_x": (-1.0, G1_MAX_SPEED),
      "ang_vel_z": (-0.7, 0.7),
    },
  ]
  return cfg


def g1_2k_runner_cfg(method: str, mpopi: MpopiCfg) -> MpopiRunnerCfg:
  """mjlab's G1 PPO settings for 2000 iterations, with ``mpopi`` added."""
  cfg = with_mpopi(unitree_g1_ppo_runner_cfg(), mpopi)
  cfg.experiment_name = G1_EXPERIMENT
  cfg.run_name = method
  cfg.max_iterations = G1_ITERATIONS
  return cfg


def g1_2k_task_id(method: str) -> str:
  return f"Mpopi-G1-2k-{method}"


for _method, _preset in METHODS.items():
  register_mjlab_task(
    task_id=g1_2k_task_id(_method),
    env_cfg=g1_2k_env_cfg(),
    play_env_cfg=g1_2k_env_cfg(play=True),
    rl_cfg=g1_2k_runner_cfg(_method, _preset()),
    runner_cls=MpopiVelocityRunner,
  )
