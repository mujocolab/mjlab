"""Copy the full per-env state of an mjlab env into a batched planning env.

Sampling MPC plans on a second copy of the task with ``num_real * repeats``
worlds. Reproducing the real env's rewards needs more than the simulator
state for tasks with stateful managers: the current command and its timer,
the previous actions (action-rate penalties, observations), sensor histories
(foot air time), stateful reward terms, per-env entity data (encoder bias)
and the model fields that domain randomization changed per world.

Manager, sensor and entity state is copied generically: two envs built from
the same config have the same object structure, so every tensor whose leading
dimension is the number of real envs is copied into the matching tensor of the
planning env, repeated ``repeats`` times. ``tests/test_mpc_state_sync.py``
checks that this reproduces the real env's rewards exactly.
"""

import dataclasses

import torch

from mjlab.envs import ManagerBasedRlEnv

SIM_STATE_FIELDS = ("qpos", "qvel", "act", "qacc_warmstart", "ctrl")

# Attributes that point back to shared or global objects, not per-env state.
_SKIP = frozenset({"_env", "env", "_sim", "sim", "_scene", "scene", "_data", "_model"})
_MAX_DEPTH = 4


def sync_env_state(
  src: ManagerBasedRlEnv, dst: ManagerBasedRlEnv, repeats: int
) -> None:
  """Make each block of ``repeats`` worlds of ``dst`` a copy of one ``src`` env.

  Must be called inside ``torch.inference_mode()`` when ``dst`` is stepped in
  inference mode (its tensors are then inference tensors).
  """
  n = src.num_envs
  if dst.num_envs != n * repeats:
    raise ValueError(f"dst has {dst.num_envs} envs, expected {n} x {repeats}.")
  src_data, dst_data = src.sim.data, dst.sim.data
  for name in SIM_STATE_FIELDS:
    value = getattr(src_data, name)
    if value.shape[-1] == 0:
      continue
    getattr(dst_data, name)[:] = value.repeat_interleave(repeats, dim=0)
  for field in src.event_manager.domain_randomization_fields:
    value = getattr(src.sim.model, field)
    getattr(dst.sim.model, field)[:] = value.repeat_interleave(repeats, dim=0)
  pairs = [
    (src.action_manager, dst.action_manager),
    (src.command_manager, dst.command_manager),
    (src.reward_manager, dst.reward_manager),
    (src.scene.sensors, dst.scene.sensors),
    (
      {k: e.data for k, e in src.scene.entities.items()},
      {k: e.data for k, e in dst.scene.entities.items()},
    ),
  ]
  seen: set[int] = set()
  for a, b in pairs:
    _copy_tree(a, b, n, repeats, _MAX_DEPTH, seen)
  dst.sim.forward()


def _children(obj) -> dict:
  if isinstance(obj, dict):
    return obj
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
  return getattr(obj, "__dict__", {})


def _copy_tree(src, dst, n: int, repeats: int, depth: int, seen: set[int]) -> None:
  if isinstance(src, torch.Tensor):
    if (
      isinstance(dst, torch.Tensor)
      and src.dim() >= 1
      and src.shape[0] == n
      and dst.shape[0] == n * repeats
      and src.shape[1:] == dst.shape[1:]
    ):
      dst.copy_(src.repeat_interleave(repeats, dim=0))
    return
  if depth == 0 or id(src) in seen or isinstance(src, (str, bytes, int, float, bool)):
    return
  seen.add(id(src))
  src_children, dst_children = _children(src), _children(dst)
  for key, value in src_children.items():
    if key in _SKIP or key not in dst_children:
      continue
    _copy_tree(value, dst_children[key], n, repeats, depth - 1, seen)


def freeze_commands(env: ManagerBasedRlEnv) -> None:
  """Stop command resampling (the planning horizon keeps the real command)."""
  for term in env.command_manager._terms.values():
    if hasattr(term, "time_left"):
      term.time_left.fill_(float("inf"))
