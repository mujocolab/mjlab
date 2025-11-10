"""Video recording wrapper for environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv


class VideoRecorder(ManagerBasedRlEnv):
  """Wraps an environment to record video during interaction.

  A minimal wrapper that records frames as the environment steps.
  Delegates all attribute access and method calls to the wrapped environment.

  Args:
      env: The environment to wrap and record.
      video_folder: Directory to save videos to.
      episode_trigger: Callable that returns True if should record this episode.
      step_trigger: Callable that returns True if should record this step.
      video_length: Maximum frames per video (None = unlimited).
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

    self.step_count = 0
    self.episode_count = 0
    self.is_recording = False
    self.current_video_frames = []
    self.current_video_path = None

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
    should_record = False
    if self.step_trigger is not None and self.step_trigger(self.step_count):
      should_record = True
    elif self.episode_trigger is not None and self.episode_trigger(self.episode_count):
      should_record = True

    if should_record and not self.is_recording:
      self._start_recording()

    # Step the environment.
    obs, reward, terminated, truncated, info = self._wrapped_env.step(action)

    # Record frame if recording.
    if self.is_recording:
      self._record_frame()

      # Check if we should stop recording.
      should_stop = False
      if (
        self.video_length is not None
        and len(self.current_video_frames) >= self.video_length
      ):
        should_stop = True
      # Also stop if any environment was reset.
      if terminated.any() or truncated.any():
        should_stop = True

      if should_stop:
        self._finish_recording()

    self.step_count += 1

    return obs, reward, terminated, truncated, info

  def render(self) -> Any:
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

    # Generate video filename.
    video_filename = f"{self.name_prefix}-episode-{self.episode_count}.mp4"
    self.current_video_path = self.video_folder / video_filename

    if not self.disable_logger:
      print(f"[INFO] Recording video to {self.current_video_path}")

  def _record_frame(self) -> None:
    """Record a frame from the environment."""
    # Get the current frame from rendering.
    if self._wrapped_env.render_mode == "rgb_array":
      frame = self._wrapped_env.render()
      if frame is not None:
        # Frame shape: (num_envs, height, width, 3).
        # Record the first environment.
        rgb_frame = (
          frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
        )
        self.current_video_frames.append(rgb_frame)

  def _finish_recording(self) -> None:
    """Finish recording and save the video."""
    if self.current_video_frames:
      from moviepy import ImageSequenceClip

      # Convert frames to proper format.
      video_frames = []
      for frame in self.current_video_frames:
        # Convert to numpy if needed.
        if not isinstance(frame, np.ndarray):
          frame = np.asarray(frame)

        # Ensure uint8.
        if frame.dtype != np.uint8:
          frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

        video_frames.append(frame)

      # Write video using moviepy.
      fps = self.env.metadata.get("render_fps", 30)
      clip = ImageSequenceClip(video_frames, fps=fps)
      clip.write_videofile(str(self.current_video_path), verbose=False, logger=None)

      if not self.disable_logger:
        print(f"[INFO] Saved video to {self.current_video_path}")

    self.is_recording = False
    self.current_video_frames = []
    self.current_video_path = None
    self.episode_count += 1
