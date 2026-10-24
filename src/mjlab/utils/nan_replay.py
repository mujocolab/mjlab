"""Read historical and model-revision-aware NaN diagnostic archives."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


class NanDump:
  """Load captured integration states and their corresponding native models.

  Diagnostic NPZ files contain pickle metadata, including historical archives;
  only open files from trusted sources. New archives verify each referenced MJB.
  """

  def __init__(self, path: str | Path) -> None:
    self.path = Path(path)
    with np.load(self.path, allow_pickle=True) as archive:
      self.metadata: dict[str, Any] = archive["_metadata"].item()
      self.states = {
        key: archive[key] for key in archive.files if key.startswith("states_step_")
      }
    self.state_keys = sorted(self.states, key=lambda key: int(key.split("_")[-1]))
    if self.metadata.get("format_version", 1) not in (1, 2):
      raise ValueError("Unsupported NaN dump format version")
    if not self.state_keys:
      raise ValueError("NaN dump contains no captured states")
    self.state_spec = self.metadata.get(
      "state_spec", mujoco.mjtState.mjSTATE_PHYSICS.value
    )
    self._model_name: str | None = None
    self._model: mujoco.MjModel | None = None
    self._data: mujoco.MjData | None = None

  def restore(
    self, frame: int, environment: int, *, forward: bool = False
  ) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Restore a frame/world; optionally compute visualization quantities.

    No model constants are recomputed. ``forward`` also restores integration
    inputs afterward, so viewing a state does not overwrite its solver warmstart.
    """
    if not 0 <= frame < len(self.state_keys):
      raise IndexError("NaN dump frame index is out of range")
    key = self.state_keys[frame]
    states = self.states[key]
    if not 0 <= environment < states.shape[0]:
      raise IndexError("NaN dump environment index is out of range")
    if self.metadata.get("format_version", 1) >= 2:
      try:
        name = self.metadata["model_files"][key][environment]
      except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
          "NaN dump is missing this frame/world's model revision"
        ) from error
    else:
      name = self.metadata["model_file"]
    if not isinstance(name, str) or Path(name).name != name:
      raise ValueError("NaN dump model must be a filename beside the archive")
    if name != self._model_name:
      model_path = self.path.parent / name
      if self.metadata.get("format_version", 1) >= 2:
        expected = self.metadata.get("model_hashes", {}).get(name)
        if (
          not expected
          or hashlib.sha256(model_path.read_bytes()).hexdigest() != expected
        ):
          raise ValueError(f"NaN dump model checksum mismatch: {name}")
      self._model = mujoco.MjModel.from_binary_path(str(model_path))
      self._data = mujoco.MjData(self._model)
      self._model_name = name
    assert self._model is not None and self._data is not None
    mujoco.mj_setState(self._model, self._data, states[environment], self.state_spec)
    if forward:
      mujoco.mj_forward(self._model, self._data)
      mujoco.mj_setState(self._model, self._data, states[environment], self.state_spec)
    return self._model, self._data
