"""Upsamples a motion to a specified frequency."""

import dataclasses
from pathlib import Path

import numpy as np
import tyro

from mjlab.envs.motion_imitation import timeseries
from mjlab import MOTION_DATA_DIR

_FPS = 30.0


@dataclasses.dataclass(frozen=True)
class Args:
  npz_file: Path
  """Path to the augmented motion data, in `.npz` format."""
  target_dt: float
  """Desired timestep."""
  slice_times: tuple[float, float] | None = None
  """Time range to slice."""
  output_dir: Path = MOTION_DATA_DIR / "processed"
  """Directory to save the rendered motion."""
  repeat_last_frame: float = 0.0
  """Seconds to repeat the last frame."""


def main(args: Args) -> None:
  ts = timeseries.TimeSeries.from_npz(args.npz_file, dt=(1.0 / _FPS))

  # Upsample and optionally slice.
  if args.slice_times is not None:
    t0, t1 = args.slice_times
    new_nsteps = int(np.ceil((t1 - t0) / args.target_dt)) + 1
    new_times = np.linspace(t0, t1, new_nsteps, endpoint=True)
    ts = ts.resample(new_times=new_times)
  else:
    ts = ts.resample(target_dt=args.target_dt)

  if args.repeat_last_frame > 0.0:
    ts = ts.repeat_last_frame(args.repeat_last_frame)
  print(f"Length: {len(ts)}")

  # Save.
  if not args.output_dir.exists():
    args.output_dir.mkdir(parents=True, exist_ok=True)
  fps = int(round(1.0 / args.target_dt))
  filename = args.output_dir / f"{args.npz_file.stem}_{fps}hz.npz"
  ts.save_as_npz(filename)
  print(f"Saved to {filename}")


if __name__ == "__main__":
  main(tyro.cli(Args, description=__doc__))
