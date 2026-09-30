"""Tests for resampling in the csv_to_npz motion loader."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.scripts.csv_to_npz import MotionLoader


def _pose_at(s: float) -> np.ndarray:
  """Pose at fractional input frame ``s``: base xyz, base quat xyzw, 3 dofs.

  Every channel changes linearly with ``s`` (the quaternion is a yaw that grows
  linearly), so lerp and slerp between two rows land exactly on this pose.
  """
  yaw = 0.02 * s
  return np.array(
    [
      0.01 * s,
      -0.005 * s,
      0.79,
      0.0,
      0.0,
      math.sin(yaw / 2),
      math.cos(yaw / 2),
      0.01 * s,
      -0.02 * s,
      0.03 * s,
    ]
  )


@pytest.mark.parametrize(
  ("input_frames", "input_fps", "output_fps", "expected_frames"),
  [
    (91, 30, 30, 91),
    (91, 30, 50, 151),
    # Here duration / output_dt is 114.99999999999999 in float64, so this case
    # needs the rounding tolerance to get 116 frames.
    (70, 30, 50, 116),
    # The clip ends between two output steps, so no frame goes past the end.
    (92, 30, 50, 152),
  ],
)
def test_motion_loader_keeps_last_frame(
  tmp_path: Path,
  input_frames: int,
  input_fps: int,
  output_fps: int,
  expected_frames: int,
) -> None:
  """The output reaches the end of the clip and never goes past it."""
  csv_path = tmp_path / "motion.csv"
  rows = np.stack([_pose_at(i) for i in range(input_frames)])
  np.savetxt(csv_path, rows, delimiter=",")

  loader = MotionLoader(
    motion_file=str(csv_path),
    input_fps=input_fps,
    output_fps=output_fps,
    device="cpu",
  )

  assert loader.output_frames == expected_frames
  assert loader.motion_dof_poss.shape[0] == expected_frames

  # Pose at the time of the last output frame. This is the last CSV row, except
  # in the 92-frame case.
  expected = _pose_at((expected_frames - 1) * input_fps / output_fps)
  expected = torch.tensor(expected, dtype=torch.float32)
  torch.testing.assert_close(
    loader.motion_base_poss[-1], expected[:3], rtol=0, atol=1e-5
  )
  torch.testing.assert_close(
    loader.motion_base_rots[-1], expected[[6, 3, 4, 5]], rtol=0, atol=1e-5
  )
  torch.testing.assert_close(
    loader.motion_dof_poss[-1], expected[7:], rtol=0, atol=1e-5
  )
