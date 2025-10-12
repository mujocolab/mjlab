from unittest.mock import Mock

import pytest
import torch

from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.reward_manager import RewardManager
from mjlab.tasks.velocity.mdp.rewards import feet_air_time


@pytest.fixture
def mock_env():
  env = Mock()
  env.num_envs = 4
  env.device = "cpu"
  env.step_dt = 0.01
  robot = Mock()
  robot.sensor_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
  robot.data.sensor_data = {name: torch.ones((4, 1)) for name in robot.sensor_names}
  env.scene = {"robot": robot}
  env.command_manager.get_command = Mock(
    return_value=torch.tensor([[1.0, 0.0, 0.0]] * 4)
  )
  return env


class TestFeetAirTime:
  @pytest.fixture
  def base_cfg(self):
    """Create base config."""
    cfg = Mock()
    cfg.params = {
      "threshold_min": 0.1,
      "threshold_max": 0.4,  # 300ms window.
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
    """Test continuous mode respects both min and max thresholds."""
    reward_term = feet_air_time(base_cfg, mock_env)
    robot = mock_env.scene["robot"]

    # Lift foot.
    robot.data.sensor_data["FL_foot"] = torch.zeros((4, 1))

    # Below min threshold (0.05s < 0.1s) - no reward.
    for _ in range(5):
      assert (reward_term(mock_env) == 0).all()

    # Between thresholds (0.15s, within 0.1-0.4s) - gives reward.
    reward = torch.zeros(4)
    for _ in range(10):
      reward = reward_term(mock_env)
    assert (reward > 0.9).all()  # Should be ~1.0 * scale.

    # Continue past max threshold (0.45s > 0.4s) - no reward.
    for _ in range(30):
      reward = reward_term(mock_env)
    assert (reward == 0).all()  # Should stop giving reward.

  def test_on_landing_mode_with_clamping(self, base_cfg, mock_env):
    """Test on_landing mode clamps reward at max threshold."""
    base_cfg.params.update({"reward_mode": "on_landing", "command_scale_type": "hard"})
    reward_term = feet_air_time(base_cfg, mock_env)
    robot = mock_env.scene["robot"]

    # Test normal air time (0.2s, within window).
    robot.data.sensor_data["FL_foot"] = torch.zeros((4, 1))
    for _ in range(20):
      assert (reward_term(mock_env) == 0).all()

    # Land - should get (0.2 - 0.1) / 0.01 = 10.0.
    robot.data.sensor_data["FL_foot"] = torch.ones((4, 1))
    assert torch.allclose(reward_term(mock_env), torch.full((4,), 10.0))

    # Test excessive air time (0.6s, beyond max).
    robot.data.sensor_data["FL_foot"] = torch.zeros((4, 1))
    for _ in range(60):
      reward_term(mock_env)

    # Land - should get clamped to (0.4 - 0.1) / 0.01 = 30.0, not 50.0.
    robot.data.sensor_data["FL_foot"] = torch.ones((4, 1))
    assert torch.allclose(reward_term(mock_env), torch.full((4,), 30.0))

  def test_reward_window_sweet_spot(self, base_cfg, mock_env):
    """Test that reward window creates optimal stepping behavior."""
    base_cfg.params["command_scale_type"] = "hard"  # Simplify test
    reward_term = feet_air_time(base_cfg, mock_env)
    robot = mock_env.scene["robot"]

    def total_reward_for_air_time(air_time_steps):
      reward_term.reset()
      robot.data.sensor_data["FL_foot"] = torch.zeros((4, 1))
      total = 0.0

      if base_cfg.params["reward_mode"] == "continuous":
        for _ in range(air_time_steps):
          total += reward_term(mock_env)[0].item() * mock_env.step_dt
      else:
        for _ in range(air_time_steps):
          reward_term(mock_env)
        robot.data.sensor_data["FL_foot"] = torch.ones((4, 1))
        total = reward_term(mock_env)[0].item() * mock_env.step_dt

      return total

    # Test continuous mode.
    short_step = total_reward_for_air_time(5)  # 0.05s - too short.
    good_step = total_reward_for_air_time(25)  # 0.25s - optimal.
    long_step = total_reward_for_air_time(50)  # 0.50s - too long.

    assert short_step == 0  # Below min threshold.
    assert good_step > 0  # In sweet spot.
    # Long step should give same total as max (0.3s of reward).
    assert abs(long_step - 0.3) < 0.01

    # Test on_landing mode.
    base_cfg.params["reward_mode"] = "on_landing"
    reward_term = feet_air_time(base_cfg, mock_env)

    short_step = total_reward_for_air_time(5)  # 0.05s - too short.
    good_step = total_reward_for_air_time(25)  # 0.25s - optimal.
    long_step = total_reward_for_air_time(50)  # 0.50s - clamped.

    assert short_step == 0
    assert abs(good_step - 0.15) < 0.001  # (0.25 - 0.1) = 0.15.
    assert abs(long_step - 0.3) < 0.001  # Clamped to (0.4 - 0.1) = 0.3.

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
      for _ in range(15):  # Exceed min threshold but stay below max.
        reward = reward_term(mock_env)
      return reward[0].item()

    # Smooth scaling - gradual transition.
    scales = [get_reward_with_command(x) for x in [0.0, 0.5, 1.0]]
    assert scales[0] < 0.01  # Near zero for low command.
    assert abs(scales[1] - 0.5) < 0.01  # ~0.5 at threshold.
    assert scales[2] > 0.99  # Near 1.0 for high command.

    # Hard scaling - binary
    base_cfg.params["command_scale_type"] = "hard"
    reward_term = feet_air_time(base_cfg, mock_env)
    assert get_reward_with_command(0.3) == 0  # Below threshold.
    assert get_reward_with_command(0.7) > 0  # Above threshold.

  def test_default_threshold_max(self, base_cfg, mock_env):
    """Test default max threshold is min + 0.3."""
    del base_cfg.params["threshold_max"]  # Remove explicit max.
    reward_term = feet_air_time(base_cfg, mock_env)

    assert reward_term.threshold_min == 0.1
    assert reward_term.threshold_max == 0.4  # Should default to 0.1 + 0.3.

  def test_multiple_feet_rewards(self, base_cfg, mock_env):
    """Test multiple feet rewards sum correctly."""
    reward_term = feet_air_time(base_cfg, mock_env)
    robot = mock_env.scene["robot"]

    # Lift two feet and wait past min threshold but before max.
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


class TestRewardManagerClassReset:
  """Test RewardManager correctly handles class-based vs function-based terms."""

  def test_class_based_reward_reset(self, mock_env):
    """Test that class-based reward terms are tracked and have reset called."""
    from dataclasses import dataclass, field

    @dataclass
    class Cfg:
      term: RewardTermCfg = field(
        default_factory=lambda: RewardTermCfg(
          func=feet_air_time,
          weight=1.0,
          params={
            "threshold_min": 0.1,
            "asset_name": "robot",
            "sensor_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            "command_name": "base_velocity",
            "command_threshold": 0.5,
          },
        )
      )

    mock_env.max_episode_length_s = 10.0
    manager = RewardManager(Cfg(), mock_env)
    term = manager._class_term_cfgs[0].func

    mock_env.scene["robot"].data.sensor_data["FL_foot"] = torch.zeros((4, 1))
    for _ in range(10):
      manager.compute(dt=0.01)

    assert (term.current_air_time > 0).any()
    manager.reset(env_ids=torch.tensor([0, 2]))
    assert term.current_air_time[0, 0] == 0
    assert term.current_air_time[1, 0] > 0

  def test_function_based_reward_not_tracked(self, mock_env):
    """Test that function-based reward terms are not tracked as class terms."""
    from dataclasses import dataclass, field

    @dataclass
    class Cfg:
      term: RewardTermCfg = field(
        default_factory=lambda: RewardTermCfg(
          func=lambda env: torch.ones(env.num_envs), weight=1.0, params={}
        )
      )

    mock_env.max_episode_length_s = 10.0
    manager = RewardManager(Cfg(), mock_env)
    assert len(manager._class_term_cfgs) == 0


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
