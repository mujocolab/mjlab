"""The MPC planner must reproduce the real env's rewards on a stateful task."""

import pytest
import torch
from conftest import get_test_device

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.mpc import SamplingMpc, SamplingMpcCfg
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Velocity-Flat-Unitree-G1"
NUM_REAL, NUM_SAMPLES = 2, 2


@pytest.mark.slow
def test_planner_reproduces_g1_velocity_rewards():
  device = get_test_device()
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = NUM_REAL
  cfg.seed = 0
  real = ManagerBasedRlEnv(cfg=cfg, device=device)
  real.reset()
  planner = SamplingMpc(
    load_env_cfg(TASK),
    NUM_REAL,
    SamplingMpcCfg(num_samples=NUM_SAMPLES, horizon=1),
    device=device,
  )
  try:
    g = torch.Generator(device=device).manual_seed(1)
    dim = real.action_manager.total_action_dim

    def action() -> torch.Tensor:
      return 0.3 * torch.randn(NUM_REAL, dim, device=device, generator=g)

    # Let commands, action history, foot air time and swing heights evolve.
    for _ in range(30):
      real.step(action())
    planner._copy_state(real)
    for _ in range(10):
      a = action()
      real.step(a)
      with torch.inference_mode():
        planner.env.step(a.repeat_interleave(NUM_SAMPLES, dim=0))
      # Per-term rewards match up to float32 noise from the contact solver.
      # Copying only the simulator state gives errors of order 1 here.
      torch.testing.assert_close(
        planner.env.reward_manager._step_reward.view(NUM_REAL, NUM_SAMPLES, -1),
        real.reward_manager._step_reward[:, None].expand(-1, NUM_SAMPLES, -1),
        atol=1e-3,
        rtol=0,
      )
  finally:
    planner.close()
    real.close()
