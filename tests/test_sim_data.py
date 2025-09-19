from dataclasses import dataclass

import pytest
import torch
import warp as wp

from mjlab.sim.sim_data import TorchArray, WarpBridge

wp.config.quiet = True


@dataclass
class MockData:
  arr: wp.array
  val: float = 1.0


def get_test_device():
  """Get device for testing, preferring CUDA if available."""
  if torch.cuda.is_available():
    return "cuda:0"
  return "cpu"


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def mock_data(device):
  with wp.ScopedDevice(device):
    return MockData(arr=wp.array([[1.0, 2.0], [3.0, 4.0]], dtype=wp.float32))


def test_torch_array_shares_memory(device):
  """TorchArray modifications affect underlying warp array."""
  with wp.ScopedDevice(device):
    wp_arr = wp.array([1.0, 2.0], dtype=wp.float32)
  torch_arr = TorchArray(wp_arr)
  torch_arr[0] = 99.0
  assert wp_arr.numpy()[0] == 99.0


def test_torch_array_ops_work(device):
  """TorchArray supports PyTorch operations."""
  with wp.ScopedDevice(device):
    wp_arr = wp.array([1.0, 2.0], dtype=wp.float32)
  torch_arr = TorchArray(wp_arr)
  assert torch.allclose(torch_arr * 2, torch.tensor([2.0, 4.0], device=device))
  assert torch.sum(torch_arr).item() == 3.0  # type: ignore


def test_bridge_wraps_arrays(mock_data):
  """WarpBridge wraps warp arrays as TorchArray."""
  bridge = WarpBridge(mock_data)
  assert isinstance(bridge.arr, TorchArray)
  assert bridge.val == 1.0  # Non-arrays pass through.


def test_bridge_preserves_memory_on_slice_assign(mock_data, device):
  """Slice assignment preserves memory addresses (CUDA graph safe)."""
  bridge = WarpBridge(mock_data)
  initial_ptr = bridge.arr._tensor.data_ptr()

  bridge.arr[:] = torch.zeros((2, 2), device=device)

  assert bridge.arr._tensor.data_ptr() == initial_ptr
  assert torch.all(bridge.arr._tensor == 0)


def test_bridge_raises_on_setattr(mock_data, device):
  """Direct assignment raises AttributeError with helpful message."""
  bridge = WarpBridge(mock_data)

  with pytest.raises(AttributeError, match="Cannot set attribute 'arr' on WarpBridge"):
    bridge.arr = torch.zeros((2, 2), device=device)

  with pytest.raises(AttributeError, match="Use in-place operations instead"):
    bridge.val = 42.0


def test_bridge_caches_wrappers(mock_data):
  """WarpBridge caches wrapper objects."""
  bridge = WarpBridge(mock_data)
  arr1 = bridge.arr
  arr2 = bridge.arr
  assert arr1 is arr2  # Same cached object.
