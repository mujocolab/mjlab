"""Runner configs that add MPOPI settings to mjlab's RSL-RL configs."""

from dataclasses import dataclass, field, fields

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mpopi_train.algorithms.config import MpopiCfg


@dataclass
class MpopiPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO settings plus the MPOPI stage (``--agent.algorithm.mpopi.*``)."""

  mpopi: MpopiCfg = field(default_factory=MpopiCfg)
  """MPOPI replay correction and MPC teacher. With ``mode="ppo"`` RSL-RL's
  ``PPO`` is constructed exactly as without it."""


@dataclass
class MpopiRunnerCfg(RslRlOnPolicyRunnerCfg):
  """On-policy runner config whose algorithm has MPOPI settings."""

  algorithm: MpopiPpoAlgorithmCfg = field(  # type: ignore[assignment]
    default_factory=MpopiPpoAlgorithmCfg
  )


def with_mpopi(cfg: RslRlOnPolicyRunnerCfg, mpopi: MpopiCfg) -> MpopiRunnerCfg:
  """Copy of an mjlab on-policy runner config with MPOPI settings added."""
  algorithm = MpopiPpoAlgorithmCfg(
    **{
      f.name: getattr(cfg.algorithm, f.name)
      for f in fields(cfg.algorithm)
      if f.name != "mpopi"
    },
    mpopi=mpopi,
  )
  return MpopiRunnerCfg(
    **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name != "algorithm"},
    algorithm=algorithm,
  )
