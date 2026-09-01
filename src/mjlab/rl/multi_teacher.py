"""Multi-expert teacher model for DAgger-style distillation.

Implements the multi-expert distillation setup of "Parkour in the Wild"
(arXiv:2505.11164): several frozen terrain- or skill-specific experts label
the student's on-policy rollouts, with each environment assigned to one
expert via an integer observation (e.g. the terrain type).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_class
from tensordict import TensorDict

from mjlab.rl.utils import clean_model_cfg


class MultiTeacherModel(nn.Module):
  """Dispatches each environment to one of several frozen expert models.

  Exposes the subset of the rsl-rl model interface that
  ``rsl_rl.algorithms.Distillation`` uses for the teacher. Expert selection
  reads the ``assignment_group`` entry of the observation TensorDict, which
  must hold the per-env expert index with shape ``(num_envs, 1)``. Because
  the index travels with the observations, dispatch works both during
  rollouts and when replaying stored (possibly trajectory-padded) batches.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    teachers: list[dict] | tuple[dict, ...] = (),
    assignment_group: str = "teacher_assignment",
  ) -> None:
    super().__init__()
    if not teachers:
      raise ValueError("MultiTeacherModel requires at least one teacher config.")
    if assignment_group not in obs.keys():
      raise ValueError(
        f"Observation group '{assignment_group}' not found in environment "
        f"observations {list(obs.keys())}. Add an observation group holding "
        "the per-env expert index."
      )
    models: list[MLPModel] = []
    for teacher_cfg in teachers:
      model_class, model_kwargs = resolve_class(clean_model_cfg(dict(teacher_cfg)))
      models.append(model_class(obs, obs_groups, obs_set, output_dim, **model_kwargs))
    self.teachers = nn.ModuleList(models)
    # Typed handle onto the same modules (ModuleList iteration is untyped).
    self._models = models
    self.assignment_group = assignment_group
    self.is_recurrent = any(m.is_recurrent for m in models)

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: list[HiddenState] | None = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    del stochastic_output  # Teacher labels are always deterministic.
    hidden_states = (
      hidden_state if hidden_state is not None else [None] * len(self.teachers)
    )
    outputs = [
      m(obs, masks=masks, hidden_state=h)
      for m, h in zip(self._models, hidden_states, strict=True)
    ]
    # Index shape (..., 1) broadcasts against action outputs (..., A).
    index = obs[self.assignment_group].round().long()
    result = outputs[0]
    for i in range(1, len(outputs)):
      result = torch.where(index == i, outputs[i], result)
    return result

  def reset(
    self,
    dones: torch.Tensor | None = None,
    hidden_state: list[HiddenState] | None = None,
  ) -> None:
    hidden_states = (
      hidden_state if hidden_state is not None else [None] * len(self._models)
    )
    for m, h in zip(self._models, hidden_states, strict=True):
      m.reset(dones, hidden_state=h)

  def get_hidden_state(self) -> list[HiddenState] | None:
    states = [m.get_hidden_state() for m in self._models]
    return None if all(s is None for s in states) else states

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    for m in self._models:
      m.detach_hidden_state(dones)

  def update_normalization(self, obs: TensorDict) -> None:
    del obs  # Experts are frozen; their normalizer stats must not drift.
