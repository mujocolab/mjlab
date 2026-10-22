"""Tests for torch backend configuration."""

import pytest
import torch

from mjlab.utils.torch import configure_torch_backends


@pytest.fixture(autouse=True)
def restore_torch_backends():
  yield
  configure_torch_backends()


@pytest.mark.parametrize(
  ("allow_tf32", "expected"), [(True, "high"), (False, "highest")]
)
def test_float32_matmul_precision_stays_readable(allow_tf32: bool, expected: str):
  """torch.compile's max-autotune modes read the generic matmul precision.

  With only the per-backend ``fp32_precision`` API set, this getter raises, which
  fails max-autotune compilation of any graph containing a matmul.
  """
  configure_torch_backends(allow_tf32=allow_tf32)
  assert torch.get_float32_matmul_precision() == expected
  assert torch.backends.cuda.matmul.allow_tf32 is allow_tf32
  assert torch.backends.cudnn.allow_tf32 is allow_tf32
