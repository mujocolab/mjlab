from __future__ import annotations

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.dex_manip.env_cfg import apply_dex_manip_overrides
from mjlab.tasks.dex_manip.inference import FrozenPolicyBank, object_policy_ids_from_env
from mjlab.tasks.registry import load_env_cfg


class _AffinePolicy(torch.nn.Module):
  def __init__(self, scale: float, bias: float):
    super().__init__()
    self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float32))
    self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))

  def forward(self, obs):
    return obs["actor"] * self.scale + self.bias


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


def _import_tasks() -> None:
  import mjlab.tasks  # noqa: F401


def test_frozen_policy_bank_matches_per_policy_dispatch() -> None:
  policies = [
    _AffinePolicy(1.0, 0.5),
    _AffinePolicy(2.0, -1.0),
    _AffinePolicy(-3.0, 2.5),
  ]
  bank = FrozenPolicyBank(("water-bottle", "orange", "tuna-fish-can"), policies)

  obs = {
    "actor": torch.tensor(
      [
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
        [6.0, 60.0],
      ]
    )
  }
  env_policy_ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

  actual = bank(obs, env_policy_ids)
  expected = torch.empty_like(obs["actor"])
  for policy_id, policy in enumerate(policies):
    env_ids = torch.nonzero(env_policy_ids == policy_id, as_tuple=False).squeeze(-1)
    expected[env_ids] = policy({"actor": obs["actor"][env_ids]})

  torch.testing.assert_close(actual, expected)


def test_object_policy_ids_from_env_cycle_assignment(device: str) -> None:
  _import_tasks()

  env_cfg = load_env_cfg("Mjlab-Dex-Manip")
  apply_dex_manip_overrides(
    env_cfg,
    objects="water-bottle,orange,tuna-fish-can",
    envs_per_object=2,
    assignment_mode="cycle",
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    if env.sim.model.geom_dataid.ndim != 2:
      pytest.skip(
        "Requires per-world geom_dataid support (MuJoCo-Warp PR #1191 or newer)."
      )

    env.reset()
    policy_ids = object_policy_ids_from_env(
      env,
      ("water-bottle", "orange", "tuna-fish-can"),
    )

    assert policy_ids.tolist() == [0, 1, 2, 0, 1, 2]
  finally:
    env.close()


def test_single_env_object_assignment_reset(device: str) -> None:
  _import_tasks()

  env_cfg = load_env_cfg("Mjlab-Dex-Manip")
  apply_dex_manip_overrides(
    env_cfg,
    objects="water-bottle",
    envs_per_object=1,
    assignment_mode="cycle",
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    env.reset()
  finally:
    env.close()
