from dataclasses import dataclass

import torch
import warp as wp
from absl.testing import absltest

from mjlab.sim.sim_data import TorchArray, WarpBridge

wp.config.quiet = True


@dataclass
class MockMjwarpData:
  qpos: wp.array
  qvel: wp.array
  ctrl: wp.array
  non_array_field: float = 1.0


class TensorProxyTest(absltest.TestCase):
  def setUp(self):
    self.num_envs = 4
    self.pos_dim = 7
    self.vel_dim = 6
    self.device = "cpu"

    with wp.ScopedDevice(self.device):
      self.wp_array = wp.array([[1.0, 2.0], [3.0, 4.0]], dtype=wp.float32)
    self.proxy = TorchArray(self.wp_array)

  def test_memory_sharing(self):
    """Test that proxy shares memory with warp array."""
    self.proxy[0, 0] = 99.0
    self.assertEqual(self.wp_array.numpy()[0, 0], 99.0)

  def test_arithmetic_ops(self):
    """Test essential arithmetic operations."""
    result = self.proxy**2
    expected = self.proxy._tensor**2
    self.assertTrue(torch.allclose(result, expected))

    self.assertTrue(torch.allclose(2 * self.proxy + 1, 2 * self.proxy._tensor + 1))

  def test_torch_func_interception(self):
    """Test that torch functions work with TensorProxy."""
    result = torch.sum(self.proxy)  # type: ignore
    expected = torch.sum(self.proxy._tensor)
    self.assertTrue(torch.allclose(result, expected))


class WarpTensorTest(absltest.TestCase):
  def setUp(self):
    self.device = "cpu"
    with wp.ScopedDevice(self.device):
      self.mock_data = MockMjwarpData(
        qpos=wp.zeros((2, 3), dtype=wp.float32),
        qvel=wp.ones((2, 3), dtype=wp.float32),
        ctrl=wp.zeros((2, 1), dtype=wp.float32),
        non_array_field=42.0,
      )
    self.warp_tensor = WarpBridge(self.mock_data)

  def test_array_wrapping(self):
    """Test that warp arrays are wrapped as TensorProxy."""
    qpos = self.warp_tensor.qpos
    self.assertIsInstance(qpos, TorchArray)
    self.assertIs(qpos._wp_array, self.mock_data.qpos)

  def test_non_array_passthrough(self):
    """Test that non-array fields are returned as is."""
    self.assertEqual(self.warp_tensor.non_array_field, 42.0)
    self.assertNotIsInstance(self.warp_tensor.non_array_field, TorchArray)

  def test_setting_from_tensor(self):
    """Test setting array from PyTorch tensor."""
    new_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=self.device)
    self.warp_tensor.qpos = new_tensor
    self.assertTrue(torch.allclose(self.warp_tensor.qpos._tensor, new_tensor))  # type: ignore

  def test_end_to_end_workflow(self):
    """Test a realistic usage workflow."""
    qpos = self.warp_tensor.qpos
    qvel = self.warp_tensor.qvel

    kinetic_energy = 0.5 * torch.sum(qvel**2, dim=1)
    potential_energy = torch.sum(qpos**2, dim=1)
    total_energy = kinetic_energy + potential_energy

    self.assertEqual(kinetic_energy.shape, (2,))
    self.assertEqual(total_energy.shape, (2,))

    qpos[0, 0] = 100.0
    self.assertEqual(self.warp_tensor.struct.qpos.numpy()[0, 0], 100.0)


if __name__ == "__main__":
  absltest.main()
