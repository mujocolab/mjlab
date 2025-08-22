from typing import Dict, TypeVar
import torch

VecEnvObs = Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]

T_observations = TypeVar("T_observations")
T_actions = TypeVar("T_actions")
T_events = TypeVar("T_events")
T_rewards = TypeVar("T_rewards")
T_terminations = TypeVar("T_terminations")
T_commands = TypeVar("T_commands")
T_curriculum = TypeVar("T_curriculum")

__all__ = (
  "VecEnvObs",
  "VecEnvStepReturn",
)
