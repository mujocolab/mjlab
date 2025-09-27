"""Smoke test for mjlab package."""

import io
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout


def test_basic_functionality() -> None:
  """Test that mjlab can create and close an environment."""
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.tasks.velocity.config.g1.flat_env_cfg import UNITREE_G1_FLAT_ENV_CFG
  from mjlab.tasks.velocity.config.g1.rough_env_cfg import UNITREE_G1_ROUGH_ENV_CFG
  from mjlab.tasks.velocity.config.go1.flat_env_cfg import UNITREE_GO1_FLAT_ENV_CFG
  from mjlab.tasks.velocity.config.go1.rough_env_cfg import UNITREE_GO1_ROUGH_ENV_CFG

  configs = [
    ("UNITREE_GO1_FLAT_ENV_CFG", UNITREE_GO1_FLAT_ENV_CFG),
    ("UNITREE_GO1_ROUGH_ENV_CFG", UNITREE_GO1_ROUGH_ENV_CFG),
    ("UNITREE_G1_FLAT_ENV_CFG", UNITREE_G1_FLAT_ENV_CFG),
    ("UNITREE_G1_ROUGH_ENV_CFG", UNITREE_G1_ROUGH_ENV_CFG),
  ]

  # Note: G1_FLAT_ENV_CFG from tracking task requires a motion file
  # and is not included in basic smoke test

  # Suppress env spam.
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      for config_name, cfg in configs:
        env = ManagerBasedRlEnv(cfg, device="cpu")
        assert env.sim.data.time == 0.0
        env.close()


if __name__ == "__main__":
  try:
    test_basic_functionality()
    print("✓ Smoke test passed!")
    sys.exit(0)
  except Exception as e:
    print(f"✗ Smoke test failed: {e}")
    sys.exit(1)
