"""Lightweight NaN guard for capturing simulation states when NaN/Inf detected."""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

if TYPE_CHECKING:
  pass


@dataclass
class NanGuardCfg:
  """Configuration for NaN guard."""

  enabled: bool = False
  buffer_size: int = 100
  output_dir: str = "/tmp/mjlab/nan_dumps"
  max_envs_to_capture: int = 5  # Max number of NaN envs to save.


class NanGuard:
  """Guards against NaN/Inf by buffering states and dumping on detection.

  When enabled, maintains a rolling buffer of simulation states and writes
  them to disk when NaN or Inf is detected. When disabled, all operations
  are no-ops with minimal overhead.
  """

  def __init__(self, cfg: NanGuardCfg, num_envs: int, mj_model: mujoco.MjModel) -> None:
    self.enabled = cfg.enabled
    self.num_envs = num_envs

    if not self.enabled:
      return

    self.buffer_size = cfg.buffer_size
    self.output_dir = Path(cfg.output_dir)
    self.max_envs_to_capture = cfg.max_envs_to_capture
    self.num_to_capture = min(self.num_envs, self.max_envs_to_capture)
    self.buffer: deque = deque(maxlen=self.buffer_size)
    self.step_counter = 0
    self._dumped = False  # Only dump once per training run.

    if self.num_to_capture < self.num_envs:
      print(
        f"[NanGuard] Capturing only {self.num_to_capture}/{self.num_envs} envs "
        f"(limited by nan_guard_max_envs={self.max_envs_to_capture})"
      )

    self.state_size = mujoco.mj_stateSize(mj_model, mujoco.mjtState.mjSTATE_PHYSICS)
    self.mj_model = mj_model
    self.mj_data = mujoco.MjData(mj_model)

  def capture(self, wp_data: mjwarp.Data) -> None:
    """Capture current simulation state to buffer (mjSTATE_PHYSICS)."""
    if not self.enabled:
      return

    states = np.empty((self.num_to_capture, self.state_size))

    for i in range(self.num_to_capture):
      self.mj_data.qpos[:] = wp_data.qpos[i].cpu().numpy()
      self.mj_data.qvel[:] = wp_data.qvel[i].cpu().numpy()
      if self.mj_model.na > 0:
        self.mj_data.act[:] = wp_data.act[i].cpu().numpy()

      mujoco.mj_getState(
        self.mj_model, self.mj_data, states[i], mujoco.mjtState.mjSTATE_PHYSICS
      )

    state = {"step": self.step_counter, "states": states.copy()}
    self.buffer.append(state)
    self.step_counter += 1

  @contextmanager
  def watch(self, wp_data: mjwarp.Data) -> Iterator[None]:
    """Context manager that captures state before and checks for NaN/Inf after.

    Usage:
      with nan_guard.watch(wp_data):
        mjwarp.step(wp_model, wp_data)
    """
    self.capture(wp_data)
    yield
    self.check_and_dump(wp_data)

  def check_and_dump(self, data: mjwarp.Data) -> bool:
    """Check for NaN/Inf and dump buffer if detected.

    Returns:
      True if NaN/Inf detected and dump occurred, False otherwise.
    """
    if not self.enabled or self._dumped:
      return False

    tensors_to_check = [data.qpos, data.qvel, data.qacc, data.qacc_warmstart]

    # Build per-env NaN mask (True if env has NaN/Inf in any tensor).
    nan_mask = torch.zeros(
      data.qpos.shape[0], dtype=torch.bool, device=data.qpos.device
    )
    for t in tensors_to_check:
      nan_mask |= torch.isnan(t).any(dim=-1) | torch.isinf(t).any(dim=-1)

    if nan_mask.any():
      nan_env_ids = torch.where(nan_mask)[0].cpu().numpy().tolist()
      self._dump_buffer(nan_env_ids)
      self._dumped = True
      return True

    return False

  def _dump_buffer(self, nan_env_ids: list[int]) -> None:
    """Write buffered states to disk."""
    self.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = self.output_dir / f"nan_dump_{timestamp}.npz"
    model_filename = self.output_dir / f"model_{timestamp}.mjb"

    # Save model to MJB (binary) format for easy reloading.
    mujoco.mj_saveModel(self.mj_model, str(model_filename), None)

    # Convert buffer to arrays indexed by step.
    data = {}
    for item in self.buffer:
      step = item["step"]
      data[f"states_step_{step:06d}"] = item["states"]

    # Add metadata.
    data["_metadata"] = np.array(
      {
        "num_envs_total": self.num_envs,
        "num_envs_captured": self.num_to_capture,
        "nan_env_ids": nan_env_ids[: self.max_envs_to_capture],
        "state_size": self.state_size,
        "buffer_size": len(self.buffer),
        "detection_step": self.step_counter,
        "timestamp": timestamp,
        "model_file": model_filename.name,
        "note": "States captured using mj_getState(mjSTATE_PHYSICS). "
        "Use mj_setState to restore. Model saved as MJB for easy reloading.",
      },
      dtype=object,
    )

    np.savez_compressed(filename, **data)
    print(f"[NanGuard] Detected NaN/Inf at step {self.step_counter}")
    print(f"[NanGuard] NaN/Inf found in envs: {nan_env_ids[:10]}...")
    print(f"[NanGuard] Dumped {len(self.buffer)} states to: {filename}")
    print(f"[NanGuard] Saved model to: {model_filename}")
