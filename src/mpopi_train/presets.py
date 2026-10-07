"""MPOPI settings of the compared training methods.

These are the settings of the G1 experiments (Kaggle notebooks). Every field
can still be overridden on the command line (``--agent.algorithm.mpopi.*``).
"""

from collections.abc import Callable
from dataclasses import replace

from mpopi_train.algorithms.config import MpcDataCfg, MpopiCfg
from mpopi_train.mpc.config import SamplingMpcCfg

MIN_ACTION_STD = 0.05
"""Lower bound on the policy std, shared by every method (prevents NaNs)."""

LABEL_ITERATIONS = 110
"""DAgger labels states every 5 iterations during the first 110 iterations."""


def ppo() -> MpopiCfg:
  """Plain RSL-RL PPO."""
  return MpopiCfg(mode="ppo", min_action_std=MIN_ACTION_STD)


def replay_is() -> MpopiCfg:
  """PPO plus the last 4 rollouts, importance-corrected (weights and V-trace)."""
  return MpopiCfg(mode="mpopi_ppo", min_action_std=MIN_ACTION_STD)


def dagger() -> MpopiCfg:
  """PPO plus behavior cloning toward an MPOPI planner on the policy's states.

  64 extra envs are driven by the policy; every 5 iterations of the first 110
  the planner labels 24 steps of them. The cloning weight decays from 1 to 0
  over 150 iterations and labels are kept for 50 iterations.
  """
  teacher = MpcDataCfg(
    num_envs=64,
    num_steps=24,
    collect_every=5,
    collect_iterations=LABEL_ITERATIONS,
    execution_std=0.0,
    driver="policy",
    buffer_segments=10,
    max_age=50,
    use_in_ppo=False,
    bc_coef=1.0,
    bc_iterations=LABEL_ITERATIONS + 40,
    planner=SamplingMpcCfg(
      num_samples=16, iterations=2, horizon=16, noise_std=0.2, num_knots=4
    ),
  )
  return MpopiCfg(mode="mpc_ppo", min_action_std=MIN_ACTION_STD, mpc=teacher)


def replay_is_dagger() -> MpopiCfg:
  """DAgger plus importance-corrected replay of PPO's own rollouts."""
  cfg = dagger()
  return replace(cfg, mpc=replace(cfg.mpc, replay_own_rollouts=True))


METHODS: dict[str, Callable[[], MpopiCfg]] = {
  "PPO": ppo,
  "Replay-IS": replay_is,
  "DAgger": dagger,
  "Replay-IS-DAgger": replay_is_dagger,
}
"""Method name (used in task ids and run names) -> MPOPI settings."""
