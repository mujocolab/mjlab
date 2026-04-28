"""Tests for mjlab.utils.random."""

import subprocess
import sys
import textwrap


def test_seed_rng_cpu_device_does_not_initialize_warp_cuda() -> None:
  """seed_rng(device="cpu") must not initialize Warp's CUDA runtime.

  Runs in a subprocess so that Warp is guaranteed uninitialized before the
  call.
  """
  script = textwrap.dedent("""
    import warp as wp
    from mjlab.utils.random import seed_rng

    assert wp._src.context.runtime is None, "Warp must not be initialized yet"
    seed_rng(42, device="cpu")
    rt = wp._src.context.runtime
    if rt is not None:
        cuda = [d for d in wp.get_devices() if "cuda" in str(d)]
        assert not cuda, f"seed_rng(device='cpu') initialized CUDA devices {cuda}"
  """)
  result = subprocess.run(
    [sys.executable, "-c", script], capture_output=True, text=True
  )
  assert result.returncode == 0, (
    f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
  )
