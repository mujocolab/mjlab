"""Pack per-clip motion ``.npz`` files into one directory of ``.npy`` arrays."""

import json
from pathlib import Path

import numpy as np
import tyro

import mjlab
from mjlab.tasks.tracking.mdp import MOTION_FIELDS, load_clip, resolve_motion_file


def main(input: str, output: str) -> None:
  """Concatenate motion clips into a packed directory that ``MotionLoader`` loads.

  The directory holds one ``<field>.npy`` per field plus ``motion_ends.npy``,
  ``fps.npy``, ``joint_names.json`` and ``motion_names.json``.

  Args:
    input: Glob of per-clip ``.npz`` files that share ``joint_names`` and ``fps``.
    output: Directory to write.
  """
  files = resolve_motion_file(input)
  clips = [load_clip(path) for path in files]
  joint_names = clips[0]["joint_names"].tolist()
  if any(clip["joint_names"].tolist() != joint_names for clip in clips):
    raise ValueError("the clips have different joint_names")
  fps = float(clips[0]["fps"].item())
  if any(float(clip["fps"].item()) != fps for clip in clips):
    raise ValueError("the clips have different fps")

  out = Path(output).expanduser()
  out.mkdir(parents=True, exist_ok=True)
  (out / "joint_names.json").write_text(json.dumps(joint_names))
  (out / "motion_names.json").write_text(json.dumps([Path(f).stem for f in files]))
  np.save(out / "fps.npy", np.array([fps]))
  lengths = [len(clip["joint_pos"]) for clip in clips]
  np.save(out / "motion_ends.npy", np.cumsum(lengths))
  for field in MOTION_FIELDS:
    np.save(out / f"{field}.npy", np.concatenate([clip.pop(field) for clip in clips]))
  print(f"[INFO] Packed {len(files)} clips ({sum(lengths)} frames) into {out}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
