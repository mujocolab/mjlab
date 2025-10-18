# tests/test_imu_observations.py
from unittest.mock import Mock

import pytest
import torch

from mjlab.envs.mdp.observations import (
  imu_ang_vel,
  imu_lin_acc,
  imu_orientation,
  imu_projected_gravity,
)
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse


class TestImuState:
  @pytest.fixture
  def mock_env(self):
    """Create basic mock environment."""
    env = Mock()
    env.num_envs = 4
    env.device = "cpu"
    env.step_dt = 0.01

    # Setup robot with three sites, with "imu" at local index 1
    robot = Mock()
    robot.site_names = ["left_foot_sensor", "imu", "right_foot_sensor"]
    num_sites = len(robot.site_names)

    robot.indexing = Mock()
    robot.indexing.site_ids = torch.tensor([0, 1, 2], dtype=torch.int64)

    site_quat_w = torch.zeros((env.num_envs, num_sites, 4), dtype=torch.float32)
    site_ang_vel_w = torch.zeros((env.num_envs, num_sites, 3), dtype=torch.float32)
    site_lin_acc_w = torch.zeros((env.num_envs, num_sites, 3), dtype=torch.float32)
    gravity_vec_w = torch.tensor([[0.0, 0.0, -9.81]], dtype=torch.float32).repeat(
      env.num_envs, 1
    )

    site_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])

    site_quat_w[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # identity
    site_quat_w[1, 1] = torch.tensor([0.70710678, 0.70710678, 0.0, 0.0])  # +90° about x
    site_quat_w[2, 1] = torch.tensor([0.70710678, 0.0, 0.70710678, 0.0])  # +90° about y
    site_quat_w[3, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # +180° about z

    site_ang_vel_w[:, 1, :] = torch.tensor([1.0, 2.0, 3.0])
    site_lin_acc_w[:, 1, :] = torch.tensor([0.5, -1.0, 2.0])

    robot.data = Mock()
    robot.data.site_quat_w = site_quat_w
    robot.data.site_ang_vel_w = site_ang_vel_w
    robot.data.site_lin_acc_w = site_lin_acc_w
    robot.data.gravity_vec_w = gravity_vec_w

    def _find_sites(name, preserve_order=True):
      if isinstance(name, (list, tuple)):
        idxs = [robot.site_names.index(n) for n in name]
        names = list(name)
      else:
        idxs = [robot.site_names.index(name)]
        names = [name]
      return idxs, names

    robot.find_sites = Mock(side_effect=_find_sites)

    env.scene = {"robot": robot}

    return env

  @pytest.fixture
  def base_cfg(self):
    """Create base config."""
    cfg = Mock()
    cfg.params = {
      "site_name": "imu",
    }
    return cfg

  def test_orientation_returns_quaternion(self, base_cfg, mock_env):
    term = imu_orientation(base_cfg, mock_env)
    out = term(mock_env)  # [num_envs, 4]

    asset = mock_env.scene["robot"]

    idxs, _ = asset.find_sites("imu")
    imu_local = idxs[0]
    imu_global = int(asset.indexing.site_ids[imu_local].item())

    expected = asset.data.site_quat_w[:, imu_global, :]

    assert out.shape == (mock_env.num_envs, 4)
    assert torch.allclose(out, expected, atol=1e-6)

  def test_ang_vel_world_to_body(self, base_cfg, mock_env):
    term = imu_ang_vel(base_cfg, mock_env)
    out = term(mock_env)  # [num_envs, 3]

    asset = mock_env.scene["robot"]

    idxs, _ = asset.find_sites("imu")
    imu_local = idxs[0]
    imu_global = int(asset.indexing.site_ids[imu_local].item())

    ang_vel_w = asset.data.site_ang_vel_w[:, imu_global, :]
    quat_w = asset.data.site_quat_w[:, imu_global, :]
    expected = quat_apply_inverse(quat_w, ang_vel_w)

    assert out.shape == (mock_env.num_envs, 3)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(out[0], ang_vel_w[0], atol=1e-6)

  def test_lin_acc_world_to_body(self, base_cfg, mock_env):
    term = imu_lin_acc(base_cfg, mock_env)
    out = term(mock_env)  # [num_envs, 3]

    asset = mock_env.scene["robot"]

    idxs, _ = asset.find_sites("imu")
    imu_local = idxs[0]
    imu_global = int(asset.indexing.site_ids[imu_local].item())

    lin_acc_w = asset.data.site_lin_acc_w[:, imu_global, :]
    quat_w = asset.data.site_quat_w[:, imu_global, :]
    expected = quat_apply_inverse(quat_w, lin_acc_w)

    assert out.shape == (mock_env.num_envs, 3)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(out[0], lin_acc_w[0], atol=1e-6)

  def test_projected_gravity_world_to_body(self, base_cfg, mock_env):
    term = imu_projected_gravity(base_cfg, mock_env)
    out = term(mock_env)  # [num_envs, 3]

    asset = mock_env.scene["robot"]

    idxs, _ = asset.find_sites("imu")
    imu_local = idxs[0]
    imu_global = int(asset.indexing.site_ids[imu_local].item())

    gravity_w = asset.data.gravity_vec_w
    quat_w = asset.data.site_quat_w[:, imu_global, :]
    expected = quat_apply_inverse(quat_w, gravity_w)

    assert out.shape == (mock_env.num_envs, 3)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(out[0], gravity_w[0], atol=1e-6)

  def test_custom_site_name_is_used(self, base_cfg, mock_env):
    asset = mock_env.scene["robot"]

    asset.indexing.site_ids = torch.tensor([0, 2, 1], dtype=torch.int64)
    asset.find_sites = Mock(return_value=([1], ["imu_alt"]))
    base_cfg.params["site_name"] = "imu_alt"

    asset.data.site_quat_w[:, 2, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    asset.data.site_ang_vel_w[:, 2, :] = (
      torch.tensor([9.0, 8.0, 7.0], dtype=torch.float32)
      .unsqueeze(0)
      .expand(mock_env.num_envs, -1)
    )
    asset.data.site_lin_acc_w[:, 2, :] = (
      torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32)
      .unsqueeze(0)
      .expand(mock_env.num_envs, -1)
    )

    q_term = imu_orientation(base_cfg, mock_env)
    w_term = imu_ang_vel(base_cfg, mock_env)
    a_term = imu_lin_acc(base_cfg, mock_env)

    q_out = q_term(mock_env)
    w_out = w_term(mock_env)
    a_out = a_term(mock_env)

    assert torch.allclose(q_out, asset.data.site_quat_w[:, 2, :], atol=1e-6)
    assert torch.allclose(
      w_out,
      quat_apply_inverse(
        asset.data.site_quat_w[:, 2, :], asset.data.site_ang_vel_w[:, 2, :]
      ),
      atol=1e-6,
    )
    assert torch.allclose(
      a_out,
      quat_apply_inverse(
        asset.data.site_quat_w[:, 2, :], asset.data.site_lin_acc_w[:, 2, :]
      ),
      atol=1e-6,
    )


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
