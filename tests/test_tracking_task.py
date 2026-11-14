"""Tests specific to motion tracking tasks."""

from unittest.mock import Mock

import pytest
import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand


@pytest.fixture(scope="module")
def tracking_task_ids() -> list[str]:
  """Get all tracking task IDs."""
  return [t for t in list_tasks() if "Tracking" in t]


@pytest.fixture(scope="module")
def g1_tracking_task_ids(tracking_task_ids: list[str]) -> list[str]:
  """Get all G1 tracking task IDs."""
  return [t for t in tracking_task_ids if "G1" in t]


def test_tracking_tasks_have_motion_command(tracking_task_ids: list[str]) -> None:
  """All tracking tasks should have a 'motion' command of type MotionCommandCfg."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "motion" in cfg.commands, f"Task {task_id} missing 'motion' command"

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} motion command is not MotionCommandCfg"
    )


def test_tracking_tasks_have_self_collision_sensor(
  tracking_task_ids: list[str],
) -> None:
  """All tracking tasks should have a self_collision sensor."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_tracking_no_state_estimation_observations() -> None:
  """No-state-estimation tasks remove observations that depend on state estimation."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

  # Test both training and play modes
  for play_mode in [False, True]:
    cfg = load_env_cfg(task_id, play=play_mode)
    mode_str = "play mode" if play_mode else "training mode"

    assert "actor" in cfg.observations, (
      f"Task {task_id} ({mode_str}) missing policy observations"
    )
    actor_terms = cfg.observations["actor"].terms

    assert "motion_anchor_pos_b" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has motion_anchor_pos_b in policy, "
      "expected it to be removed for no-state-estimation variant"
    )
    assert "base_lin_vel" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has base_lin_vel in policy, "
      "expected it to be removed for no-state-estimation variant"
    )


def test_tracking_play_disables_rsi_randomization() -> None:
  """Tracking play tasks should disable RSI randomization."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.pose_range == {}, (
      f"Task {task_id} (play mode) has non-empty pose_range={motion_cmd.pose_range}, "
      "expected empty dict for disabled RSI"
    )
    assert motion_cmd.velocity_range == {}, (
      f"Task {task_id} (play mode) has non-empty velocity_range={motion_cmd.velocity_range}, "
      "expected empty dict for disabled RSI"
    )


def test_tracking_play_uses_start_sampling_mode() -> None:
  """Tracking play tasks should use sampling_mode='start'."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.sampling_mode == "start", (
      f"Task {task_id} (play mode) sampling_mode={motion_cmd.sampling_mode}, expected 'start'"
    )


def test_g1_tracking_has_correct_action_scale(g1_tracking_task_ids: list[str]) -> None:
  """G1 tracking tasks should use G1_ACTION_SCALE."""
  for task_id in g1_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def _make_stub_motion_command(
  *,
  bin_count: int = 100,
  adaptive_uniform_ratio: float = 0.1,
  adaptive_kernel_size: int = 1,
  adaptive_lambda: float = 0.8,
) -> MotionCommand:
  cmd = object.__new__(MotionCommand)
  cmd.cfg = Mock()
  cmd.cfg.adaptive_uniform_ratio = adaptive_uniform_ratio
  cmd.cfg.adaptive_kernel_size = adaptive_kernel_size
  cmd.cfg.adaptive_lambda = adaptive_lambda
  cmd.bin_count = bin_count
  cmd.bin_failed_count = torch.zeros(bin_count, dtype=torch.float32)
  cmd._current_bin_failed = torch.zeros(bin_count, dtype=torch.float32)
  kernel = torch.tensor(
    [adaptive_lambda**i for i in range(adaptive_kernel_size)], dtype=torch.float32
  )
  cmd.kernel = kernel / kernel.sum()
  return cmd


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0, 50.0, 500.0])
def test_adaptive_sampling_uniform_floor_scale_invariant(scale: float) -> None:
  """Uniform prior share stays invariant to failure-count scale (#1153)."""
  cmd = _make_stub_motion_command(bin_count=100, adaptive_uniform_ratio=0.1)
  cmd.bin_failed_count[0] = scale

  probs = cmd._compute_adaptive_sampling_probs()

  assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
  assert torch.isclose(probs[0], torch.tensor(0.901), atol=1e-6)
  assert torch.allclose(probs[1:], torch.full((99,), 0.001), atol=1e-6)


def test_adaptive_sampling_zero_failures_falls_back_to_uniform() -> None:
  """Zero accumulated failures produce a valid uniform distribution even at eps=0."""
  cmd = _make_stub_motion_command(bin_count=20, adaptive_uniform_ratio=0.0)
  probs = cmd._compute_adaptive_sampling_probs()

  assert torch.allclose(probs, torch.full((20,), 1.0 / 20.0), atol=1e-6)


def test_motion_command_cfg_validates_adaptive_uniform_ratio() -> None:
  """MotionCommandCfg rejects adaptive_uniform_ratio outside [0, 1]."""
  for bad_ratio in (-0.1, 1.5):
    with pytest.raises(ValueError, match="adaptive_uniform_ratio"):
      MotionCommandCfg(
        motion_file="dummy.npz",
        anchor_body_name="pelvis",
        body_names=("pelvis",),
        entity_name="robot",
        resampling_time_range=(10.0, 10.0),
        adaptive_uniform_ratio=bad_ratio,
      )
