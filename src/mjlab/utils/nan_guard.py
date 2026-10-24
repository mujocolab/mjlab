"""Capture per-world model and integration-state history when NaN/Inf is detected."""

import enum
import errno
import hashlib
import os
import shutil
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from filelock import FileLock

ModelProvider = Callable[[], tuple[Any, set[str]]]

# mjSTATE_INTEGRATION, including controls, forces, solver warmstart and history.
_STATE_FIELDS = (
  "time",
  "qpos",
  "qvel",
  "act",
  "history",
  "qacc_warmstart",
  "ctrl",
  "qfrc_applied",
  "xfrc_applied",
  "eq_active",
  "mocap_pos",
  "mocap_quat",
  "userdata",
  "plugin_state",
)


def _link_or_copy(source: Path, latest: Path) -> None:
  """Publish a latest artifact without requiring symlink privileges."""
  temporary = latest.with_name(f".{latest.name}.{uuid4().hex}.tmp")
  try:
    try:
      temporary.symlink_to(source.name)
    except (NotImplementedError, OSError) as error:
      if isinstance(error, OSError) and (
        error.errno not in (errno.EPERM, errno.ENOSYS, errno.ENOTSUP)
        and getattr(error, "winerror", None) not in (50, 1314)
      ):
        raise
      shutil.copy2(source, temporary)
    os.replace(temporary, latest)
  finally:
    temporary.unlink(missing_ok=True)


def _atomic_save(path: Path, writer: Callable[[Any], None]) -> None:
  temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
  try:
    with temporary.open("wb") as stream:
      writer(stream)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


@contextmanager
def _simulation_stream(data: Any) -> Iterator[None]:
  """Order diagnostic reads/clones with the Warp step, including custom streams."""
  if data.qpos.is_cuda:
    stream = torch.cuda.ExternalStream(
      wp.get_stream(str(data.qpos.device)).cuda_stream, device=data.qpos.device
    )
    with torch.cuda.stream(stream):
      yield
  else:
    yield


@dataclass
class NanGuardCfg:
  """Configuration for NaN guard."""

  enabled: bool = False
  buffer_size: int = 100
  output_dir: str = "/tmp/mjlab/nan_dumps"
  max_envs_to_dump: int = 5

  def __post_init__(self) -> None:
    if self.buffer_size < 1 or self.max_envs_to_dump < 1:
      raise ValueError("NaN guard buffer_size and max_envs_to_dump must be positive")


class NanGuard:
  """Guards against NaN/Inf by buffering states and dumping on detection.

  When enabled, maintains a rolling buffer of simulation states and writes
  them to disk when NaN or Inf is detected. When disabled, all operations
  are no-ops with minimal overhead.
  """

  def __init__(
    self,
    cfg: NanGuardCfg,
    num_envs: int,
    mj_model: mujoco.MjModel,
    *,
    model_provider: ModelProvider | None = None,
  ) -> None:
    self.enabled = cfg.enabled
    self.num_envs = num_envs

    if not self.enabled:
      return

    self.buffer_size = cfg.buffer_size
    self.output_dir = Path(cfg.output_dir)
    self.max_envs_to_dump = cfg.max_envs_to_dump
    self.buffer: deque = deque(maxlen=self.buffer_size)
    self.step_counter = 0
    self._dumped = False

    self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION.value
    self.state_size = mujoco.mj_stateSize(mj_model, self.state_spec)
    # Viewers may sync a selected world's fields into the public host model.
    # Keep a private immutable asset/template model for historical revisions.
    embed_assets = getattr(mujoco, "mj_embedAssets", None)
    if embed_assets is not None:
      self.mj_model = embed_assets(mj_model)
    else:
      # Before MuJoCo 3.17 assets are embedded by the compiler itself.
      if np.any(mj_model.tex_adr < 0):
        raise ValueError("NaN replay needs mj_embedAssets for by-reference textures")
      self.mj_model = copy(mj_model)
    self.mj_data = mujoco.MjData(self.mj_model)
    self._model_provider = model_provider

  def capture(self, wp_data: mjwarp.Data) -> None:
    """Capture current simulation state to buffer."""
    if not self.enabled or self._dumped:
      return
    with _simulation_stream(wp_data):
      self._capture(wp_data)

  def _capture(self, wp_data: mjwarp.Data) -> None:
    state: dict[str, Any] = {"step": self.step_counter}
    for name in _STATE_FIELDS:
      if np.size(getattr(self.mj_data, name)) == 0:
        continue
      if not hasattr(wp_data, name):
        raise ValueError(f"NaN guard cannot capture integration field {name!r}")
      state[name] = getattr(wp_data, name).clone()
    state["model_fields"] = {}
    state["model_options"] = {}
    state["warp_options"] = {}
    if self._model_provider is not None:
      model, fields = self._model_provider()
      for name in sorted(fields):
        target = getattr(self.mj_model, name, None)
        if not isinstance(target, np.ndarray):
          raise ValueError(f"Cannot replay expanded model field {name!r}")
        if not target.size:
          continue
        state["model_fields"][name] = getattr(model, name)[:].clone()
      for name in model.opt.struct.__dataclass_fields__:
        value = getattr(model.opt, name)
        if hasattr(value, "clone"):
          # Keep shared options shared instead of cloning a broadcast view.
          value = value[: value.wp_array.shape[0]].clone()
        elif isinstance(value, enum.Enum):
          value = value.value
        state["warp_options"][name] = value
        if hasattr(self.mj_model.opt, name):
          state["model_options"][name] = value
      # Warp stores the reciprocal square root, rather than MuJoCo's impratio.
      value = state["warp_options"]["impratio_invsqrt"]
      state["model_options"]["impratio"] = value.square().reciprocal()

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

  @staticmethod
  def detect_nans(data: mjwarp.Data) -> torch.Tensor:
    """Detect NaN/Inf values in physics state (qpos, qvel, qacc, qacc_warmstart).

    Args:
      data: MuJoCo simulation data containing physics state.

    Returns:
      Boolean tensor where True indicates environments with NaN/Inf values.
    """
    tensors_to_check = [
      data.qpos,
      data.qvel,
      data.qacc,
      data.qacc_warmstart,
      data.sensordata,
    ]

    # Build per-env NaN mask (True if env has NaN/Inf in any tensor).
    nan_mask = torch.zeros(
      data.qpos.shape[0], dtype=torch.bool, device=data.qpos.device
    )
    for t in tensors_to_check:
      nan_mask |= torch.isnan(t).any(dim=-1) | torch.isinf(t).any(dim=-1)

    return nan_mask

  def check_and_dump(self, data: mjwarp.Data) -> bool:
    """Check for NaN/Inf and dump buffer if detected.

    Returns:
      True if NaN/Inf detected and dump occurred, False otherwise.
    """
    if not self.enabled or self._dumped:
      return False

    with _simulation_stream(data):
      nan_mask = self.detect_nans(data)
      if nan_mask.any():
        nan_env_ids = torch.where(nan_mask)[0].cpu().numpy().tolist()
        self._dump_buffer(nan_env_ids)
        self._dumped = True
        self.buffer.clear()
        return True

    return False

  def _dump_buffer(self, nan_env_ids: list[int]) -> None:
    """Write buffered states to disk."""
    # The viewer package imports Simulation, which imports NanGuard.
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts

    self.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (
      datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}_{uuid4().hex}"
    )
    filename = self.output_dir / f"nan_dump_{timestamp}.npz"
    model_filename = self.output_dir / f"model_{timestamp}.mjb"

    envs_to_dump = nan_env_ids[: self.max_envs_to_dump]
    if not self.buffer:
      raise ValueError("NaN guard has no captured frames to dump")
    data: dict[str, Any] = {}
    model_files: dict[str, list[str]] = {}
    model_hashes: dict[str, str] = {}
    backend_options: dict[str, Any] = {}
    revisions: dict[str, str] = {}
    for item in self.buffer:
      step = item["step"]
      key = f"states_step_{step:06d}"
      states = np.empty((len(envs_to_dump), self.state_size))
      model_files[key] = []
      for idx, env_id in enumerate(envs_to_dump):
        fields = {
          name: value[env_id].cpu().numpy()
          for name, value in item["model_fields"].items()
        }
        option_groups: dict[str, dict[str, np.ndarray]] = {}
        for group in ("model_options", "warp_options"):
          option_groups[group] = {}
          for name, value in item[group].items():
            if isinstance(value, torch.Tensor):
              value = value[env_id % value.shape[0]].cpu().numpy()
            option_groups[group][name] = np.asarray(value)
        options = option_groups["model_options"]
        digest = hashlib.sha256()
        for group, values in {"fields": fields, **option_groups}.items():
          digest.update(group.encode())
          for name, value in sorted(values.items()):
            digest.update(name.encode())
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
        revision = digest.hexdigest()
        if revision not in revisions:
          native_model = copy(self.mj_model)
          for name, value in fields.items():
            target = getattr(native_model, name)
            target[:] = value.reshape(target.shape)
          for name, value in options.items():
            target = getattr(native_model.opt, name)
            if isinstance(target, np.ndarray):
              target[:] = value.reshape(target.shape)
            else:
              setattr(native_model.opt, name, value.item())
          disable_model_sameframe_shortcuts(native_model)
          name = (
            model_filename.name
            if not revisions
            else f"model_{timestamp}_r{len(revisions):06d}.mjb"
          )
          binary = np.empty(mujoco.mj_sizeModel(native_model), dtype=np.uint8)
          mujoco.mj_saveModel(native_model, None, binary)
          _atomic_save(
            self.output_dir / name,
            lambda stream, binary=binary: stream.write(binary.data),
          )
          model_hashes[name] = hashlib.sha256(binary.data).hexdigest()
          backend_options[name] = {
            key: value.tolist() for key, value in option_groups["warp_options"].items()
          }
          revisions[revision] = name
        model_files[key].append(revisions[revision])
        for name in _STATE_FIELDS:
          if name not in item:
            continue
          value = item[name][env_id].cpu().numpy()
          target = getattr(self.mj_data, name)
          if isinstance(target, np.ndarray):
            target[:] = value.reshape(target.shape)
          else:
            setattr(self.mj_data, name, value.item())
        mujoco.mj_getState(self.mj_model, self.mj_data, states[idx], self.state_spec)
      data[key] = states

    data["_metadata"] = np.array(
      {
        "format_version": 2,
        "num_envs_total": self.num_envs,
        "num_envs_dumped": len(envs_to_dump),
        "nan_env_ids": nan_env_ids,
        "dumped_env_ids": list(envs_to_dump),
        "state_spec": self.state_spec,
        "state_size": self.state_size,
        "buffer_size": len(self.buffer),
        "detection_step": self.step_counter,
        "timestamp": timestamp,
        "model_file": model_filename.name,
        "model_files": model_files,
        "model_hashes": model_hashes,
        "warp_options": backend_options,
        "mujoco_version": mujoco.__version__,
        "note": "Pre-step integration states. model_files selects the immutable "
        "model revision for every frame/world. Restore using mj_setState; "
        "do not recompute captured model constants with mj_setConst.",
      },
      dtype=object,
    )

    # The unique NPZ is the commit point: every referenced MJB is complete first.
    _atomic_save(filename, lambda stream: np.savez_compressed(stream, **data))

    # Report the authoritative archive even if publishing a latest alias fails.
    print(f"[NanGuard] Detected NaN/Inf at step {self.step_counter}")
    print(f"[NanGuard] NaN/Inf found in envs: {nan_env_ids[:10]}...")
    print(f"[NanGuard] Dumping {len(envs_to_dump)} envs: {envs_to_dump}")
    print(f"[NanGuard] Dumped {len(self.buffer)} states to: {filename}", flush=True)
    print(f"[NanGuard] Saved model to: {model_filename}")

    # Symlink where supported; otherwise expose the same artifacts as copies.
    latest_dump = self.output_dir / "nan_dump_latest.npz"
    latest_model = self.output_dir / "model_latest.mjb"
    with FileLock(str(self.output_dir / "nan_dump_latest.lock"), timeout=10):
      _link_or_copy(model_filename, latest_model)
      _link_or_copy(filename, latest_dump)

    print(f"[NanGuard] Latest dump available at: {latest_dump}")
