"""Motion imitation task suite."""

from mjlab import PROCESSED_DATA_DIR
from mjlab.envs.registry import _REGISTRY
from mjlab.envs.motion_imitation.reference_motion import ReferenceMotion
from mjlab.envs.motion_imitation.timeseries import TimeSeries
from mjlab.envs.motion_imitation.motion_tracking import (
  G1MotionTrackingConfig,
  MotionTrackingEnv,
)

__all__ = (
  "ReferenceMotion",
  "TimeSeries",
  "MOTION_DATA_DIR",
  "LAFAN1_DATA_DIR",
  "PROCESSED_DATA_DIR",
)


def _get_episode_length(motion_name: str) -> int:
  return len(ReferenceMotion.from_npz(PROCESSED_DATA_DIR / motion_name))


class G1MotionTrackingDanceConfig(G1MotionTrackingConfig):
  motion_name: str = "dance1_subject2_50hz.npz"
  max_episode_length: int = _get_episode_length(motion_name)


_REGISTRY.register_task(
  "Motion-Tracking-Dance-G1-v0",
  MotionTrackingEnv,
  G1MotionTrackingDanceConfig,
)
