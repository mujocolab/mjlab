"""Teacher-student distillation variants of the velocity task.

Transforms a fully-configured velocity env config (with "actor"/"critic"
observation groups) into the observation layout used for the two stages of
the distill-then-finetune pipeline of arXiv:2505.11164:

- Distillation: a "teacher" group (the teacher's training observations,
  without corruption so labels are deterministic) and a "student" group
  (noisy, and by default blind, i.e. without the height scan, so the
  recurrent student must infer the terrain from proprioceptive history).
- RL fine-tuning: the same "student" group as the policy observation plus
  the privileged "critic" group for asymmetric PPO.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity import mdp

STUDENT_EXCLUDED_TERMS = ("height_scan",)
"""Actor terms hidden from the student (privileged exteroception)."""


def _make_student_group(actor_group: ObservationGroupCfg) -> ObservationGroupCfg:
  student_terms = {
    name: term
    for name, term in actor_group.terms.items()
    if name not in STUDENT_EXCLUDED_TERMS
  }
  # Corruption follows the actor group: enabled for training configs,
  # disabled for play configs.
  return replace(actor_group, terms=student_terms)


def to_distillation_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
  teacher_assignment: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Rewrite observation groups for the distillation stage.

  Args:
    cfg: A fully-configured velocity env config with "actor" and "critic"
      observation groups. Modified in place and returned.
    teacher_assignment: Add a "teacher_assignment" group exposing the
      per-env terrain type, for multi-teacher distillation.
  """
  actor_group = cfg.observations["actor"]
  observations = {
    "student": _make_student_group(actor_group),
    # The teacher sees exactly what it was trained on, but without noise so
    # its action labels are deterministic.
    "teacher": replace(actor_group, enable_corruption=False),
  }
  if teacher_assignment:
    observations["teacher_assignment"] = ObservationGroupCfg(
      terms={"terrain_type": ObservationTermCfg(func=mdp.terrain_type)},
      concatenate_terms=True,
      enable_corruption=False,
    )
  cfg.observations = observations
  return cfg


def to_finetune_env_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Rewrite observation groups for RL fine-tuning of a distilled student.

  The policy keeps the student's observation layout (so distilled weights
  and normalizer stats transfer), while the critic keeps the privileged
  observations for asymmetric PPO.
  """
  actor_group = cfg.observations["actor"]
  cfg.observations = {
    "student": _make_student_group(actor_group),
    "critic": cfg.observations["critic"],
  }
  return cfg
