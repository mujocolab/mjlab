from __future__ import annotations

import tempfile
from dataclasses import asdict

import pytest
import torch
from conftest import get_test_device

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.dex_manip.env_cfg import apply_dex_manip_overrides
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


def _import_tasks() -> None:
  import mjlab.tasks  # noqa: F401


def _resolve_object_geom_id(env: ManagerBasedRlEnv) -> int:
  geom_cfg = SceneEntityCfg("object", geom_names=("object_geom",))
  geom_cfg.resolve(env.scene)
  geom_ids = env.scene["object"].indexing.geom_ids[geom_cfg.geom_ids]
  return int(geom_ids[0].item())


def test_dex_manip_task_registered() -> None:
  _import_tasks()

  assert "Mjlab-Dex-Manip" in list_tasks()
  cfg = load_env_cfg("Mjlab-Dex-Manip")

  assert "randomize_object_mesh" in cfg.events
  mesh_ids = cfg.events["randomize_object_mesh"].params["mesh_ids"]
  assert mesh_ids == (
    "water_bottle_mesh",
    "orange_mesh",
    "tuna_fish_can_mesh",
  )
  assert cfg.metrics is not None
  assert "reward_mean" in cfg.metrics
  assert "reward_water_bottle" in cfg.metrics
  assert "reward_orange" in cfg.metrics
  assert "reward_tuna_fish_can" in cfg.metrics


def test_dex_manip_apply_overrides() -> None:
  _import_tasks()

  cfg = load_env_cfg("Mjlab-Dex-Manip")
  selected = apply_dex_manip_overrides(
    cfg,
    objects="water-bottle,orange,tuna-fish-can",
    envs_per_object=4,
    assignment_mode="cycle",
  )

  assert selected == ("water-bottle", "orange", "tuna-fish-can")
  assert cfg.scene.num_envs == 12
  assert cfg.events["randomize_object_mesh"].params["mesh_ids"] == (
    "water_bottle_mesh",
    "orange_mesh",
    "tuna_fish_can_mesh",
  )


def test_dex_manip_task_uses_task_specific_grasp_and_spawn() -> None:
  _import_tasks()

  cfg = load_env_cfg("Mjlab-Dex-Manip")

  robot_cfg = cfg.scene.entities["robot"]
  object_cfg = cfg.scene.entities["object"]

  assert robot_cfg.init_state.joint_pos["if_rot"] == pytest.approx(0.4)
  assert robot_cfg.init_state.joint_pos["rf_rot"] == pytest.approx(-0.4)
  assert robot_cfg.init_state.joint_pos["th_cmc"] == pytest.approx(1.45)
  assert object_cfg.init_state.pos == pytest.approx((-0.092, 0.055, 0.27))


@pytest.mark.slow
def test_dex_manip_training_smoke(device: str) -> None:
  _import_tasks()

  env_cfg = load_env_cfg("Mjlab-Dex-Manip")
  apply_dex_manip_overrides(
    env_cfg,
    objects="water-bottle,orange,tuna-fish-can",
    envs_per_object=1,
    assignment_mode="cycle",
  )
  env_cfg.episode_length_s = 0.2

  rl_cfg = asdict(load_rl_cfg("Mjlab-Dex-Manip"))
  rl_cfg["max_iterations"] = 1
  rl_cfg["num_steps_per_env"] = 4
  rl_cfg["save_interval"] = 100

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  try:
    geom_id = _resolve_object_geom_id(env)
    if env.sim.model.geom_dataid.ndim != 2:
      pytest.skip(
        "Requires per-world geom_dataid support (MuJoCo-Warp PR #1191 or newer)."
      )

    env.reset()
    values_before = env.sim.model.geom_dataid[:, geom_id].clone()
    assert torch.unique(values_before).numel() >= 2

    with tempfile.TemporaryDirectory() as tmpdir:
      wrapped = RslRlVecEnvWrapper(env)
      runner = MjlabOnPolicyRunner(wrapped, rl_cfg, log_dir=tmpdir, device=device)
      runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    values_after = env.sim.model.geom_dataid[:, geom_id]
    assert torch.unique(values_after).numel() >= 2
  finally:
    env.close()
