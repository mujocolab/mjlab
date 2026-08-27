"""Left-right symmetry mapping copied from Klavier's G1 walk PPO config."""

from __future__ import annotations

import torch

_NUM_ACTIONS = 29
_FRAME_STACK = 5
_SWAP_PAIRS = (
  (0, 1),
  (3, 4),
  (6, 7),
  (9, 10),
  (11, 12),
  (13, 14),
  (15, 16),
  (17, 18),
  (19, 20),
  (21, 22),
  (23, 24),
  (25, 26),
  (27, 28),
)
_INVERT = (2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 23, 24, 27, 28)


def _mirror_joint(value: torch.Tensor) -> torch.Tensor:
  index = torch.arange(_NUM_ACTIONS, device=value.device)
  for left, right in _SWAP_PAIRS:
    index[left], index[right] = index[right].clone(), index[left].clone()
  result = value[..., index].clone()
  result[..., list(_INVERT)] *= -1
  return result


def _mirror_actor_observation(value: torch.Tensor) -> torch.Tensor:
  if value.shape[-1] != 490:
    raise ValueError(f"Klavier mirror expects 490 Actor observations, got {value.shape}")
  result = value.clone()
  for frame in range(_FRAME_STACK):
    start = frame * 3
    result[..., start] *= -1
    result[..., start + 2] *= -1
    start = 3 * _FRAME_STACK + frame * 3
    result[..., start + 1] *= -1
    start = 6 * _FRAME_STACK + frame * 3
    result[..., start + 1] *= -1
    result[..., start + 2] *= -1
  for base in (11, 11 + _NUM_ACTIONS, 11 + 2 * _NUM_ACTIONS):
    for frame in range(_FRAME_STACK):
      start = base * _FRAME_STACK + frame * _NUM_ACTIONS
      result[..., start : start + _NUM_ACTIONS] = _mirror_joint(
        value[..., start : start + _NUM_ACTIONS]
      )
  return result


def _mirror_ball_history(value: torch.Tensor) -> torch.Tensor:
  """Mirror [..., history, (ball_xy, left_xy, right_xy, visible)]."""
  if value.shape[-1] != 7:
    raise ValueError(f"Klavier mirror expects 7-D ball history, got {value.shape}")
  result = value.clone()
  result[..., 1] *= -1
  result[..., 2] = value[..., 4]
  result[..., 3] = -value[..., 5]
  result[..., 4] = value[..., 2]
  result[..., 5] = -value[..., 3]
  return result


def data_augmentation_func(env, obs, actions):
  """Return original+mirrored batches in the RSL-RL 5.x extension format."""
  del env
  augmented_obs = None
  if obs is not None:
    mirrored = obs.clone()
    mirrored["actor"] = _mirror_actor_observation(obs["actor"])
    if "actor_history" in obs.keys():
      mirrored["actor_history"] = _mirror_ball_history(obs["actor_history"])
    augmented_obs = torch.cat((obs, mirrored), dim=0)
  augmented_actions = None
  if actions is not None:
    augmented_actions = torch.cat((actions, _mirror_joint(actions)), dim=0)
  return augmented_obs, augmented_actions
