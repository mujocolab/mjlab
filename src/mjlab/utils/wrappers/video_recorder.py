"""Video recording wrapper for environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from typing_extensions import assert_never

from mjlab.envs import ManagerBasedRlEnv


class VideoRecorder(ManagerBasedRlEnv):
  """Wraps an environment to record video during interaction.

  A minimal wrapper that records frames as the environment steps.
  Delegates all attribute access and method calls to the wrapped environment.

  For vectorized environments, only records the first environment (index 0) and
  tracks its episode boundaries. This matches gymnasium's RecordVideo behavior.

  Note: Unlike gymnasium's RecordVideo, this wrapper allows both episode_trigger
  and step_trigger to be used simultaneously. If both are provided, recording will
  start when either trigger fires. The filename will reflect which trigger started
  the recording (e.g., "rl-video-step-1000.mp4" or "rl-video-episode-5.mp4").

  Args:
      env: The environment to wrap and record.
      video_folder: Directory to save videos to.
      episode_trigger: Callable that returns True if should record this episode.
          Receives the actual episode count (increments when env[0] episodes end).
      step_trigger: Callable that returns True if should record this step.
          Receives the global step count.
      video_length: Maximum frames per video. If None, records until env[0] episode ends.
          If set, records exactly that many frames regardless of episode boundaries.
      name_prefix: Prefix for video filenames.
      disable_logger: Whether to disable logging.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    video_folder: Path,
    episode_trigger: Callable[[int], bool] | None = None,
    step_trigger: Callable[[int], bool] | None = None,
    video_length: int | None = None,
    name_prefix: str = "rl-video",
    disable_logger: bool = False,
  ):
    # Don't call super().__init__() - we're wrapping an existing env.
    self._wrapped_env = env
    self.video_folder = video_folder
    self.video_folder.mkdir(parents=True, exist_ok=True)

    self.episode_trigger = episode_trigger
    self.step_trigger = step_trigger
    self.video_length = video_length
    self.name_prefix = name_prefix
    self.disable_logger = disable_logger

    self.step_count: int = 0
    self.episode_count: int = 0  # Tracks actual episodes
    self.video_count: int = 0  # Tracks completed videos
    self.is_recording: bool = False
    self.current_video_frames: list[np.ndarray] = []
    self.current_video_path: Path | None = None
    self.trigger_type: Literal["step", "episode"] | None = None

  def __getattr__(self, name: str) -> Any:
    """Delegate attribute access to wrapped environment."""
    return getattr(self._wrapped_env, name)

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    """Get the unwrapped environment."""
    return self._wrapped_env.unwrapped

  def reset(self, **kwargs: Any) -> Any:
    """Reset the environment."""
    return self._wrapped_env.reset(**kwargs)

  def step(self, action: torch.Tensor) -> Any:
    """Step the environment and optionally record video.

    Args:
        action: Action tensor.

    Returns:
        Tuple of (obs, reward, terminated, truncated, info) from env.step().
    """
    # Check if we should start recording.
    step_triggered = self.step_trigger is not None and self.step_trigger(
      self.step_count
    )
    episode_triggered = self.episode_trigger is not None and self.episode_trigger(
      self.episode_count
    )

    if (step_triggered or episode_triggered) and not self.is_recording:
      # Track which trigger started the recording for filename generation
      if step_triggered:
        self.trigger_type = "step"
      else:
        self.trigger_type = "episode"
      self._start_recording()

    # Step the environment.
    obs, reward, terminated, truncated, info = self._wrapped_env.step(action)

    # Track episode boundaries (only for the first environment, which we're recording)
    # This matches gymnasium's behavior for vectorized environments.
    if terminated[0] or truncated[0]:
      self.episode_count += 1

    # Record frame if recording.
    if self.is_recording:
      self._record_frame()

      # Check if we should stop recording.
      # If video_length is set, stop only when reaching that length.
      # If video_length is None, stop when the first environment (being recorded) terminates.
      if self.video_length is not None:
        should_stop = len(self.current_video_frames) >= self.video_length
      else:
        should_stop = terminated[0] or truncated[0]

      if should_stop:
        self._finish_recording()

    self.step_count += 1

    return obs, reward, terminated, truncated, info

  def render(self) -> np.ndarray | None:
    """Render the environment."""
    return self._wrapped_env.render()

  def close(self) -> None:
    """Close the environment and finalize any open videos."""
    if self.is_recording:
      self._finish_recording()
    self._wrapped_env.close()

  def _start_recording(self) -> None:
    """Start recording a new video."""
    self.is_recording = True
    self.current_video_frames = []

    # Generate video filename based on which trigger started recording.
    assert self.trigger_type is not None, "trigger_type must be set before recording"

    if self.trigger_type == "step":
      video_filename = f"{self.name_prefix}-step-{self.step_count}.mp4"
    elif self.trigger_type == "episode":
      video_filename = f"{self.name_prefix}-episode-{self.episode_count}.mp4"
    else:
      assert_never(self.trigger_type)

    self.current_video_path = self.video_folder / video_filename

    if not self.disable_logger:
      print(f"[INFO] Recording video to {self.current_video_path}")

  def _record_frame(self) -> None:
    """Record a frame from the environment.

    For vectorized environments, only records env[0].
    """
    if self._wrapped_env.render_mode == "rgb_array":
      frame = self._wrapped_env.render()
      if frame is not None:
        # For vectorized envs: frame shape is (num_envs, height, width, 3).
        # Extract the first environment's frame.
        rgb_frame = (
          frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
        )
        self.current_video_frames.append(rgb_frame)

  def _finish_recording(self) -> None:
    """Finish recording and save the video."""
    if self.current_video_frames:
      from moviepy import ImageSequenceClip

      # Convert frames to uint8 format.
      video_frames = []
      for frame in self.current_video_frames:
        frame = np.asarray(frame) if not isinstance(frame, np.ndarray) else frame
        if frame.dtype != np.uint8:
          frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        video_frames.append(frame)

      # Write video using moviepy.
      fps = self._wrapped_env.metadata.get("render_fps", 30)
      clip = ImageSequenceClip(video_frames, fps=fps)
      clip.write_videofile(str(self.current_video_path))

      if not self.disable_logger:
        print(f"[INFO] Saved video to {self.current_video_path}")

    self.is_recording = False
    self.current_video_frames = []
    self.current_video_path = None
    self.video_count += 1
    self.trigger_type = None  # Reset trigger type after recording
