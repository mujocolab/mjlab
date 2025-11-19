"""Tests for GPU selection utilities."""

import os

import pytest

from mjlab.utils.gpu import select_gpus


@pytest.fixture(autouse=True)
def clean_cuda_env():
  """Clean CUDA_VISIBLE_DEVICES before each test."""
  original = os.environ.get("CUDA_VISIBLE_DEVICES")
  yield
  # Restore original value after test.
  if original is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = original


def test_select_gpus_with_visible_devices_set():
  """Respects CUDA_VISIBLE_DEVICES when selecting GPUs."""
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

  # Select first two GPUs (indices 0, 1 -> physical GPUs 0, 1).
  selected, num = select_gpus([0, 1])
  assert selected == [0, 1]
  assert num == 2


def test_select_gpus_with_non_contiguous_devices():
  """Handles non-contiguous CUDA_VISIBLE_DEVICES correctly."""
  os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,5"

  # Index 0 -> physical GPU 1.
  selected, num = select_gpus([0])
  assert selected == [1]
  assert num == 1

  # Index 1 -> physical GPU 3.
  selected, num = select_gpus([1])
  assert selected == [3]
  assert num == 1

  # Indices 0, 2 -> physical GPUs 1, 5.
  selected, num = select_gpus([0, 2])
  assert selected == [1, 5]
  assert num == 2


def test_select_gpus_all_option():
  """Selects all GPUs when 'all' is specified."""
  os.environ["CUDA_VISIBLE_DEVICES"] = "2,4,6"

  selected, num = select_gpus("all")
  assert selected == [2, 4, 6]
  assert num == 3


def test_select_gpus_single_gpu():
  """Correctly selects a single GPU."""
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

  selected, num = select_gpus([0])
  assert selected == [0]
  assert num == 1


def test_select_gpus_with_spaces():
  """Handles CUDA_VISIBLE_DEVICES with spaces correctly."""
  os.environ["CUDA_VISIBLE_DEVICES"] = " 0 , 1 , 2 "

  selected, num = select_gpus([1, 2])
  assert selected == [1, 2]
  assert num == 2


def test_select_gpus_empty_segments():
  """Handles empty segments in CUDA_VISIBLE_DEVICES."""
  # This can happen with malformed environment variables.
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,,2"

  selected, num = select_gpus([0, 1])
  assert selected == [0, 2]
  assert num == 2


def test_select_gpus_respects_user_order():
  """Preserves the order specified by the user."""
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

  # User requests in reverse order.
  selected, num = select_gpus([3, 2, 1, 0])
  assert selected == [3, 2, 1, 0]
  assert num == 4


def test_select_gpus_cpu_mode_explicit():
  """Selects CPU mode when None is specified."""
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

  selected, num = select_gpus(None)
  assert selected is None
  assert num == 0


def test_select_gpus_cpu_mode_empty_cuda_visible_devices():
  """Selects CPU mode when CUDA_VISIBLE_DEVICES is empty."""
  os.environ["CUDA_VISIBLE_DEVICES"] = ""

  # Should return CPU mode (None, 0) since no GPUs are visible.
  selected, num = select_gpus([0])
  assert selected is None
  assert num == 0
