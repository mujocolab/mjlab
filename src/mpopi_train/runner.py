"""Runners that set up MPOPI and the MPC teacher on top of mjlab's runners."""

import math
from typing import Any, cast

from rsl_rl.env import VecEnv
from rsl_rl.modules import GaussianDistribution

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mpopi_train.algorithms.algorithm import MpopiPpo
from mpopi_train.mpc.collector import MpcCollector

MPOPI_PPO_CLASS_NAME = "mpopi_train.algorithms:MpopiPpo"


class MpopiRunnerMixin:
  """Resolves ``algorithm.mpopi`` before RSL-RL builds the algorithm.

  Mixed in front of an mjlab runner class. In mode ``"ppo"`` the ``mpopi`` key
  is removed so RSL-RL's ``PPO`` gets exactly its usual arguments; otherwise
  the algorithm becomes :class:`MpopiPpo`, and in mode ``"mpc_ppo"`` it gets an
  :class:`MpcCollector` on the training task.
  """

  alg: Any
  env: Any
  cfg: dict
  device: str

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs,
  ) -> None:
    mpopi_cfg = train_cfg.get("algorithm", {}).get("mpopi") or {}
    min_action_std = mpopi_cfg.get("min_action_std")
    _resolve_mpopi_mode(train_cfg)
    # The mixin precedes an mjlab runner, whose __init__ takes these arguments.
    cast(Any, super()).__init__(env, train_cfg, log_dir, device, **kwargs)
    if min_action_std is not None:
      set_min_action_std(self.alg.actor, min_action_std)
    self._attach_mpc_collector()

  def _attach_mpc_collector(self) -> None:
    """In mode ``"mpc_ppo"``, give the algorithm an MPC collector on this task."""
    alg = self.alg
    if not isinstance(alg, MpopiPpo) or alg.mpc_cfg is None:
      return
    if not isinstance(self.env, RslRlVecEnvWrapper):
      raise ValueError("mpc_ppo requires an mjlab environment.")
    cfg = alg.mpc_cfg
    collector = MpcCollector(
      self.env.unwrapped.cfg,
      num_envs=cfg.num_envs,
      num_steps=cfg.num_steps,
      planner_cfg=cfg.planner,
      execution_std=cfg.execution_std,
      clip_actions=self.env.clip_actions,
      device=self.device,
      seed=int(self.cfg.get("seed", 0)) + 1,  # Different starts from training.
      teacher_gap_every=cfg.teacher_gap_every,
    )
    alg.attach_mpc_collector(collector)


class MpopiOnPolicyRunner(MpopiRunnerMixin, MjlabOnPolicyRunner):
  """:class:`MjlabOnPolicyRunner` with MPOPI."""


class MpopiVelocityRunner(MpopiRunnerMixin, VelocityOnPolicyRunner):
  """mjlab's velocity-task runner (ONNX export on save) with MPOPI."""


def set_min_action_std(actor, min_std: float) -> None:
  """Raise the lower bound of a Gaussian actor's std to ``min_std``."""
  dist = getattr(actor, "distribution", None)
  if not isinstance(dist, GaussianDistribution):
    raise ValueError("min_action_std requires a Gaussian actor distribution.")
  dist.std_range[0] = max(dist.std_range[0], min_std)
  dist.log_std_range[0] = math.log(dist.std_range[0])


def _resolve_mpopi_mode(train_cfg: dict) -> None:
  """Translate ``algorithm.mpopi.mode`` into the RSL-RL algorithm class."""
  alg_cfg = train_cfg.get("algorithm")
  if alg_cfg is None or "mpopi" not in alg_cfg:
    return
  mpopi_cfg = alg_cfg.pop("mpopi")
  if mpopi_cfg is None or mpopi_cfg.get("mode", "ppo") == "ppo":
    return
  if alg_cfg.get("class_name", "PPO") not in ("PPO", MPOPI_PPO_CLASS_NAME):
    raise ValueError(
      f"MPOPI mode '{mpopi_cfg['mode']}' requires the PPO algorithm, got "
      f"class_name='{alg_cfg['class_name']}'."
    )
  alg_cfg["class_name"] = MPOPI_PPO_CLASS_NAME
  alg_cfg["mpopi"] = mpopi_cfg
