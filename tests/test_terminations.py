from unittest.mock import Mock

import pytest
import torch

from mjlab.envs.mdp.terminations import illegal_contacts


class TestIllegalContacts:
  @pytest.fixture
  def mock_env(self):
    """Create basic mock environment."""
    env = Mock()
    env.num_envs = 4
    env.device = "cpu"

    # Setup robot with some sensors.
    robot = Mock()
    robot.data = Mock()
    robot.data.sensor_data = {
      "torso_link_contact": torch.zeros((4, 1)),
      "left_arm_1_contact": torch.zeros((4, 1)),
      "right_arm_1_contact": torch.zeros((4, 1)),
    }
    env.scene = {"robot": robot}
    return env

  @pytest.fixture
  def base_cfg(self):
    """Create base config."""
    cfg = Mock()
    cfg.params = {
      # rely on default asset_cfg ("robot")
      "sensor_names": ["torso_link_contact"],
      "threshold": 1.0,
    }
    return cfg

  def test_init_requires_sensor_names(self, base_cfg, mock_env):
    """Test missing sensor."""
    base_cfg.params.update({"sensor_names": []})  # missing
    with pytest.raises(ValueError):
      illegal_contacts(base_cfg, mock_env)

  def test_init_unknown_sensor_raises(self, base_cfg, mock_env):
    """Test unknown sensor."""
    base_cfg.params.update({"sensor_names": ["does_not_exist"], "threshold": 1.0})
    with pytest.raises(KeyError):
      illegal_contacts(base_cfg, mock_env)

  def test_no_termination_below_or_equal_threshold(self, base_cfg, mock_env):
    """Test threshold respected."""
    term = illegal_contacts(base_cfg, mock_env)

    # Below threshold -> no terminate
    mock_env.scene["robot"].data.sensor_data["torso_link_contact"] = torch.zeros((4, 1))
    out = term(mock_env, sensor_names=["torso_link_contact"])
    assert out.dtype == torch.bool and out.shape == (mock_env.num_envs,)
    assert not out.any()

    # Exactly at threshold (strict ">"): still no terminate
    mock_env.scene["robot"].data.sensor_data["torso_link_contact"] = torch.full(
      (4, 1), 1.0
    )
    out = term(mock_env, sensor_names=["torso_link_contact"])
    assert not out.any()

  def test_termination_strictly_above_threshold(self, base_cfg, mock_env):
    """Test threshold not respected."""
    term = illegal_contacts(base_cfg, mock_env)

    # Strictly above threshold -> terminate
    mock_env.scene["robot"].data.sensor_data["torso_link_contact"] = torch.full(
      (4, 1), 1.0001
    )
    out = term(mock_env, sensor_names=["torso_link_contact"])
    assert out.all()

  def test_vectorized_mixed_batch(self, base_cfg, mock_env):
    """Test mixed force values."""
    term = illegal_contacts(base_cfg, mock_env)

    # Mix values across envs; expect strict ">" behavior
    forces = torch.tensor([[0.0], [2.0], [1.0], [1.1]])  # >1.0 only at idx 1 and 3
    mock_env.scene["robot"].data.sensor_data["torso_link_contact"] = forces
    out = term(mock_env, sensor_names=["torso_link_contact"])
    expected = torch.tensor([False, True, False, True])
    assert torch.equal(out, expected)

  def test_multiple_sensors_or_logic(self, base_cfg, mock_env):
    """Test multiple sensors/logics."""
    base_cfg.params.update(
      {
        "sensor_names": ["torso_link_contact", "left_arm_1_contact"],
        "threshold": 1.0,
      }
    )
    term = illegal_contacts(base_cfg, mock_env)

    # First sensor below, second above -> OR across sensors => terminate
    mock_env.scene["robot"].data.sensor_data["torso_link_contact"] = torch.zeros((4, 1))
    mock_env.scene["robot"].data.sensor_data["left_arm_1_contact"] = torch.full(
      (4, 1), 2.0
    )
    out = term(mock_env, sensor_names=["torso_link_contact", "left_arm_1_contact"])
    assert out.all()


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
