"""Tests for video recording with mediapy."""

from pathlib import Path
from unittest.mock import Mock

import mediapy as media
import numpy as np
import torch


def _make_mock_env(num_envs: int = 1):
  """Create a mock environment that produces random RGB frames."""
  env = Mock()
  env.render_mode = "rgb_array"
  env.metadata = {"render_fps": 30}
  env.render.return_value = np.random.default_rng().integers(
    0, 255, (num_envs, 64, 64, 3), dtype=np.uint8
  )
  env.step.return_value = (
    torch.zeros(num_envs),  # obs
    torch.zeros(num_envs),  # reward
    torch.zeros(num_envs, dtype=torch.bool),  # terminated
    torch.zeros(num_envs, dtype=torch.bool),  # truncated
    {},  # info
  )
  env.close.return_value = None
  env.unwrapped = env
  return env


def test_step_trigger_writes_video(tmp_path: Path):
  """VideoRecorder writes a readable mp4 when the step trigger fires."""
  from mjlab.utils.wrappers.video_recorder import VideoRecorder

  env = _make_mock_env()
  recorder = VideoRecorder(
    env,
    video_folder=tmp_path,
    step_trigger=lambda step: step == 0,
    video_length=5,
    disable_logger=True,
  )

  action = torch.zeros(1)
  for _ in range(6):
    recorder.step(action)

  recorder.close()

  videos = list(tmp_path.glob("*.mp4"))
  assert len(videos) == 1

  # Verify the file is a valid video readable by mediapy.
  frames = media.read_video(str(videos[0]))
  assert len(frames) == 5
  assert frames[0].shape == (64, 64, 3)


def test_accepts_string_path(tmp_path: Path):
  """VideoRecorder accepts a string path for video_folder."""
  from mjlab.utils.wrappers.video_recorder import VideoRecorder

  env = _make_mock_env()
  folder = str(tmp_path / "vids")
  recorder = VideoRecorder(
    env,
    video_folder=folder,
    step_trigger=lambda step: step == 0,
    video_length=3,
    disable_logger=True,
  )

  action = torch.zeros(1)
  for _ in range(4):
    recorder.step(action)

  recorder.close()

  assert list(Path(folder).glob("*.mp4"))


def test_episode_trigger_records_episode_once(tmp_path: Path):
  """An episode trigger records one video per triggered episode.

  With video_length shorter than the episode, the recording must not restart
  mid-episode and overwrite the episode's video with a later chunk.
  """
  from mjlab.utils.wrappers.video_recorder import VideoRecorder

  env = _make_mock_env()
  recorder = VideoRecorder(
    env,
    video_folder=tmp_path,
    episode_trigger=lambda episode: episode == 0,
    video_length=5,
    disable_logger=True,
  )

  # 20 steps inside a single episode (the mock never terminates).
  action = torch.zeros(1)
  for _ in range(20):
    recorder.step(action)

  recorder.close()

  assert recorder.video_count == 1
  videos = list(tmp_path.glob("*.mp4"))
  assert len(videos) == 1
  assert len(media.read_video(str(videos[0]))) == 5


def test_episode_trigger_fires_at_each_triggered_episode_start(tmp_path: Path):
  """The episode trigger is re-evaluated at the start of every new episode."""
  from mjlab.utils.wrappers.video_recorder import VideoRecorder

  env = _make_mock_env()
  # Episodes last 4 steps: env 0 terminates on every 4th step.
  env.step.side_effect = [
    (
      torch.zeros(1),
      torch.zeros(1),
      torch.tensor([i % 4 == 3]),
      torch.zeros(1, dtype=torch.bool),
      {},
    )
    for i in range(12)
  ]
  recorder = VideoRecorder(
    env,
    video_folder=tmp_path,
    episode_trigger=lambda episode: episode in (0, 2),
    video_length=2,
    disable_logger=True,
  )

  action = torch.zeros(1)
  for _ in range(12):
    recorder.step(action)

  recorder.close()

  assert recorder.video_count == 2
  videos = sorted(p.name for p in tmp_path.glob("*.mp4"))
  assert videos == ["rl-video-episode-0.mp4", "rl-video-episode-2.mp4"]
