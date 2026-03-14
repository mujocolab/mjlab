from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap

from mjlab.managers.scene_entity_config import SceneEntityCfg

from .objects import object_names_to_mesh_names, resolve_mesh_id

_DEFAULT_OBJECT_CFG = SceneEntityCfg("object", geom_names=("object_geom",))


def _resolve_object_geom_id(env, object_cfg: SceneEntityCfg) -> int:
  cache_key = f"_dex_manip_object_geom_id::{object_cfg.name}::{object_cfg.geom_names}"
  cached = getattr(env, cache_key, None)
  if cached is not None:
    return int(cached)

  object_cfg.resolve(env.scene)
  geom_ids = env.scene[object_cfg.name].indexing.geom_ids[object_cfg.geom_ids]
  geom_id = int(geom_ids[0].item())
  setattr(env, cache_key, geom_id)
  return geom_id


def object_policy_ids_from_env(
  env,
  object_names: Sequence[str],
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  """Return per-env policy ids by reading the active object mesh assignment."""
  geom_id = _resolve_object_geom_id(env, object_cfg)
  dataid = env.sim.model.geom_dataid
  if dataid.ndim != 2:
    raise ValueError(
      "Expected per-world geom_dataid with shape (num_envs, ngeom). "
      "Install MuJoCo-Warp with per-world geom_dataid support."
    )

  per_env_dataid = dataid[:, geom_id].to(dtype=torch.int64, device=env.device)
  mesh_ids = torch.tensor(
    [resolve_mesh_id(env, mesh_name) for mesh_name in object_names_to_mesh_names(tuple(object_names))],
    device=env.device,
    dtype=torch.int64,
  )

  matches = per_env_dataid.unsqueeze(-1) == mesh_ids.unsqueeze(0)
  if not torch.all(matches.any(dim=-1)):
    unknown = torch.unique(per_env_dataid[~matches.any(dim=-1)]).tolist()
    raise ValueError(
      f"Found envs assigned to mesh ids {unknown}, which are not covered by {tuple(object_names)}."
    )
  return matches.to(dtype=torch.int64).argmax(dim=-1)


def _index_obs(obs: Mapping[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
  return {key: value[index] for key, value in obs.items()}


class FrozenPolicyBank(nn.Module):
  """Batched frozen inference over multiple policies with equal env counts.

  The intended use is a single vectorized environment whose worlds are assigned
  to different objects. Each policy sees only the worlds for its own object, but
  all policy forwards are evaluated together inside one batched GPU call.
  """

  def __init__(self, policy_names: Sequence[str], policies: Sequence[nn.Module]):
    super().__init__()
    if not policies:
      raise ValueError("FrozenPolicyBank requires at least one policy.")
    if len(policy_names) != len(policies):
      raise ValueError(
        f"policy_names ({len(policy_names)}) and policies ({len(policies)}) must match."
      )

    self.policy_names = tuple(policy_names)
    self.num_policies = len(self.policy_names)

    modules = [policy.eval() for policy in policies]
    self._template = copy.deepcopy(modules[0]).to("meta")
    self._params, self._buffers = stack_module_state(modules)

  def _call_single(
    self,
    params: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor],
    obs: Mapping[str, torch.Tensor],
  ) -> torch.Tensor:
    return functional_call(self._template, (params, buffers), (obs,))

  def forward(self, obs: Mapping[str, torch.Tensor], env_policy_ids: torch.Tensor) -> torch.Tensor:
    if env_policy_ids.ndim != 1:
      raise ValueError(
        f"env_policy_ids must be 1-D with shape (num_envs,), got {tuple(env_policy_ids.shape)}."
      )
    if env_policy_ids.numel() == 0:
      raise ValueError("env_policy_ids is empty.")

    env_policy_ids = env_policy_ids.to(dtype=torch.long)
    order = torch.argsort(env_policy_ids, stable=True)
    sorted_policy_ids = env_policy_ids[order]

    unique_ids, counts = torch.unique_consecutive(sorted_policy_ids, return_counts=True)
    expected_ids = torch.arange(self.num_policies, device=env_policy_ids.device)
    if unique_ids.numel() != self.num_policies or not torch.equal(unique_ids, expected_ids):
      raise ValueError(
        "FrozenPolicyBank currently expects every policy id 0..N-1 to appear at least once. "
        f"Got unique ids {unique_ids.tolist()}."
      )
    if not torch.all(counts == counts[0]):
      raise ValueError(
        "FrozenPolicyBank currently expects equal env counts per policy. "
        f"Got counts {counts.tolist()}."
      )

    envs_per_policy = int(counts[0].item())
    grouped_obs = {
      key: value[order].reshape(self.num_policies, envs_per_policy, *value.shape[1:])
      for key, value in obs.items()
    }

    grouped_actions = vmap(self._call_single, in_dims=(0, 0, 0))(
      self._params,
      self._buffers,
      grouped_obs,
    )
    sorted_actions = grouped_actions.reshape(-1, *grouped_actions.shape[2:])

    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(order.numel(), device=order.device)
    return sorted_actions[inv_order]
