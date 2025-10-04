"""Smoke test for mjlab package."""

import io
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout


def test_basic_functionality() -> None:
  """Test that mjlab can create and close an environment."""
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.tasks.velocity.config.go1.flat_env_cfg import UNITREE_GO1_FLAT_ENV_CFG

  # Suppress env spam.
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(UNITREE_GO1_FLAT_ENV_CFG, device="cpu")
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
