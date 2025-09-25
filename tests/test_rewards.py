from unittest.mock import Mock

import pytest
import torch

from mjlab.tasks.velocity.mdp.rewards import feet_air_time


class TestFeetAirTime:
  @pytest.fixture
  def mock_env(self):
    """Create basic mock environment."""
    env = Mock()
    env.num_envs = 4
    env.device = "cpu"
    env.step_dt = 0.01

    # Setup robot with foot sensors.
    robot = Mock()
    robot.sensor_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    robot.data.sensor_data = {name: torch.ones((4, 1)) for name in robot.sensor_names}
    env.scene = {"robot": robot}

    # Setup command manager.
    env.command_manager.get_command = Mock(
      return_value=torch.tensor([[1.0, 0.0, 0.0]] * 4)
    )
    return env

  @pytest.fixture
  def base_cfg(self):
    """Create base config."""
    cfg = Mock()
    cfg.params = {
      "threshold": 0.1,
      "asset_name": "robot",
      "sensor_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
      "command_name": "base_velocity",
      "command_threshold": 0.5,
      "reward_mode": "continuous",
      "command_scale_type": "smooth",
      "command_scale_width": 0.2,
    }
    return cfg

  def test_continuous_mode_threshold_behavior(self, base_cfg, mock_env):
    """Test continuous mode respects threshold."""
    reward_term = feet_air_time(base_cfg, mock_env)
    robot = mock_env.scene["robot"]

    # Lift foot.
    robot.data.sensor_data["FL_foot"] = torch.zeros((4, 1))

    # Below threshold (0.05s < 0.1s) - no reward.
    for _ in range(5):
      assert (reward_term(mock_env) == 0).all()

    # Above threshold (0.15s > 0.1s) - gives reward.
    reward = torch.zeros(4)
    for _ in range(10):
      reward = reward_term(mock_env)
    assert (reward > 0.9).all()  # Should be ~1.0 * scale

  def test_on_landing_mode(self, base_cfg, mock_env):
    """Test on_landing mode gives reward only on landing."""
    base_cfg.params.update({"reward_mode": "on_landing", "command_scale_type": "hard"})
    reward_term = feet_air_time(base_cfg, mock_env)
    robot = mock_env.scene["robot"]

    # Keep foot in air for 0.15s.
    robot.data.sensor_data["FL_foot"] = torch.zeros((4, 1))
    for _ in range(15):
      assert (reward_term(mock_env) == 0).all()  # No reward while in air.

    # Land - should get (0.15 - 0.1) / 0.01 = 5.0.
    robot.data.sensor_data["FL_foot"] = torch.ones((4, 1))
    assert torch.allclose(reward_term(mock_env), torch.full((4,), 5.0))

    # Next step - no reward.
    assert (reward_term(mock_env) == 0).all()

  def test_command_scaling(self, base_cfg, mock_env):
    """Test smooth vs hard command scaling."""
    reward_term = feet_air_time(base_cfg, mock_env)
    robot = mock_env.scene["robot"]

    def get_reward_with_command(cmd_norm):
      mock_env.command_manager.get_command.return_value = torch.tensor(
        [[cmd_norm, 0.0, 0.0]] * 4
      )
      reward_term.reset()
      robot.data.sensor_data["FL_foot"] = torch.zeros((4, 1))
      reward = torch.zeros(4)
      for _ in range(15):  # Exceed threshold.
        reward = reward_term(mock_env)
      return reward[0].item()

    # Smooth scaling - gradual transition.
    scales = [get_reward_with_command(x) for x in [0.0, 0.5, 1.0]]
    assert scales[0] < 0.01  # Near zero for low command.
    assert abs(scales[1] - 0.5) < 0.01  # ~0.5 at threshold.
    assert scales[2] > 0.99  # Near 1.0 for high command.

    # Hard scaling - binary.
    base_cfg.params["command_scale_type"] = "hard"
    reward_term = feet_air_time(base_cfg, mock_env)
    assert get_reward_with_command(0.3) == 0  # Below threshold.
    assert get_reward_with_command(0.7) > 0  # Above threshold.

  def test_continuous_vs_landing_total_reward(self, base_cfg, mock_env):
    """Test both modes give same total reward."""
    base_cfg.params["command_scale_type"] = "hard"  # Avoid scaling differences

    def simulate_jump(reward_term, robot):
      total = 0.0
      robot.data.sensor_data["FL_foot"] = torch.zeros((4, 1))
      for _ in range(15):  # 0.15s in air.
        total += reward_term(mock_env)[0].item() * mock_env.step_dt
      robot.data.sensor_data["FL_foot"] = torch.ones((4, 1))
      total += reward_term(mock_env)[0].item() * mock_env.step_dt
      return total

    # Test continuous mode.
    reward_cont = feet_air_time(base_cfg, mock_env)
    total_cont = simulate_jump(reward_cont, mock_env.scene["robot"])

    # Test landing mode.
    base_cfg.params["reward_mode"] = "on_landing"
    reward_land = feet_air_time(base_cfg, mock_env)
    total_land = simulate_jump(reward_land, mock_env.scene["robot"])

    assert abs(total_cont - total_land) < 0.001
    assert abs(total_cont - 0.05) < 0.001  # Expected: 5 steps * 0.01.

  def test_multiple_feet_rewards(self, base_cfg, mock_env):
    """Test multiple feet rewards sum correctly."""
    reward_term = feet_air_time(base_cfg, mock_env)
    robot = mock_env.scene["robot"]

    # Lift two feet and wait past threshold.
    robot.data.sensor_data["FL_foot"] = torch.zeros((4, 1))
    robot.data.sensor_data["FR_foot"] = torch.zeros((4, 1))
    two_feet_reward = torch.zeros(4)
    for _ in range(15):
      two_feet_reward = reward_term(mock_env)

    # Put one foot down.
    robot.data.sensor_data["FR_foot"] = torch.ones((4, 1))
    one_foot_reward = reward_term(mock_env)

    # Two feet should give ~2x reward of one foot.
    assert abs(two_feet_reward[0] / one_foot_reward[0] - 2.0) < 0.1

  def test_reset(self, base_cfg, mock_env):
    """Test reset clears state correctly."""
    reward_term = feet_air_time(base_cfg, mock_env)
    robot = mock_env.scene["robot"]

    # Build up air time.
    robot.data.sensor_data["FL_foot"] = torch.zeros((4, 1))
    for _ in range(10):
      reward_term(mock_env)

    reward_term.reset(env_ids=torch.tensor([0, 2]))

    assert reward_term.current_air_time[0, 0] == 0
    assert reward_term.current_air_time[2, 0] == 0
    assert reward_term.current_air_time[1, 0] > 0
    assert reward_term.current_air_time[3, 0] > 0


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
