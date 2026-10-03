"""Tests for resampling in the csv_to_npz motion loader."""

from pathlib import Path

import numpy as np
import pytest

from mjlab.scripts.csv_to_npz import MotionLoader


@pytest.mark.parametrize(
  ("input_frames", "output_fps", "expected_frames"),
  [
    (91, 30, 91),
    (91, 50, 151),
    # duration / output_dt is 114.99999999999999 here, so this needs the tolerance.
    (70, 50, 116),
    # The clip ends between two output steps, so no frame goes past the end.
    (92, 50, 152),
  ],
)
def test_motion_loader_keeps_last_frame(
  tmp_path: Path, input_frames: int, output_fps: int, expected_frames: int
) -> None:
  input_fps = 30
  # Base xyz, identity quat (xyzw), and one dof equal to the input frame index.
  rows = np.zeros((input_frames, 8))
  rows[:, 6] = 1.0
  rows[:, 7] = np.arange(input_frames)
  csv_path = tmp_path / "motion.csv"
  np.savetxt(csv_path, rows, delimiter=",")

  loader = MotionLoader(
    motion_file=str(csv_path),
    input_fps=input_fps,
    output_fps=output_fps,
    device="cpu",
  )

  assert loader.output_frames == expected_frames
  last_input_frame = (expected_frames - 1) * input_fps / output_fps
  assert loader.motion_dof_poss[-1].item() == pytest.approx(last_input_frame, abs=1e-4)
