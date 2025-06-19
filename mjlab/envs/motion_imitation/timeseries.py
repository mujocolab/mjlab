"""Time series utilities."""

import pathlib
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline

_INTERP_MAP = {
  "qpos": "special",
  "qvel": "linear",
  "xpos": "linear",
  "xquat": "slerp",
  "linvel": "linear",
  "angvel": "linear",
  "com": "linear",
}


@dataclass(frozen=True)
class TimeSeries:
  """A utility class for working with time-series data.

  Attributes:
    times: 1D array of timestamps.
    xpos: 2D array of body positions. Shape is (t, nbody, 3).
    xquat: 2D array of body orientations. Shape is (t, nbody, 4).
    qpos: 2D array of joint positions. Shape is (t, nq).
    qvel: 2D array of joint velocities. Shape is (t, nv).
    linvel: 2D array of linear velocities. Shape is (t, nbody, 3).
    angvel: 2D array of angular velocities. Shape is (t, nbody, 3).
    com: 2D array of center of mass positions. Shape is (t, 3).
  """

  times: np.ndarray
  xpos: np.ndarray
  xquat: np.ndarray
  qpos: np.ndarray
  qvel: np.ndarray
  linvel: np.ndarray
  angvel: np.ndarray
  com: np.ndarray

  def __post_init__(self):
    if self.times.ndim != 1:
      raise ValueError(f"times must be a 1D array, got {self.times.ndim}D array")
    if not np.all(np.diff(self.times) > 0):
      raise ValueError("times must be monotonically increasing")

  def __len__(self) -> int:
    return len(self.times)

  @classmethod
  def from_npz(cls, path: pathlib.Path, dt: Optional[float] = None) -> "TimeSeries":
    with np.load(path) as f:
      qpos = f["qpos"]
      xpos = f["xpos"]
      xquat = f["xquat"]
      qvel = f["qvel"]
      linvel = f["linvel"]
      angvel = f["angvel"]
      com = f["com"]
      times = f.get("times", None)
    if times is None:
      if dt is None:
        raise ValueError("dt must be provided if times is not in the npz file")
      n_steps = len(qpos)
      times = np.linspace(0.0, n_steps * dt, n_steps, endpoint=True)
    return cls(
      times=times,
      qpos=qpos,
      qvel=qvel,
      xpos=xpos,
      xquat=xquat,
      linvel=linvel,
      angvel=angvel,
      com=com,
    )

  def save_as_npz(self, path: pathlib.Path) -> None:
    np.savez(
      path,
      times=self.times,
      qpos=self.qpos,
      qvel=self.qvel,
      xpos=self.xpos,
      xquat=self.xquat,
      linvel=self.linvel,
      angvel=self.angvel,
      com=self.com,
    )

  def interpolate(self, t: Union[float, np.ndarray], data: np.ndarray) -> np.ndarray:
    t = np.atleast_1d(np.asarray(t))
    return scipy.interpolate.interp1d(
      self.times,
      data,
      kind="linear",
      axis=0,
      bounds_error=False,
      fill_value=(data[0], data[-1]),
      assume_sorted=True,
    )(t)

  def slerp(self, t: Union[float, np.ndarray], quats: np.ndarray) -> np.ndarray:
    t = np.atleast_1d(np.asarray(t))
    _, N, _ = quats.shape
    result = np.zeros((t.shape[0], N, 4))
    for i in range(N):
      rotations = R.from_quat(quats[:, i], scalar_first=True)
      slerp = RotationSpline(self.times, rotations)
      result[:, i] = slerp(t).as_quat(scalar_first=True)
    return result

  def special_interpolate(
    self, t: Union[float, np.ndarray], data: np.ndarray
  ) -> np.ndarray:
    result = np.zeros((t.shape[0], data.shape[1]))
    result[:, :3] = self.interpolate(t, data[:, :3])
    result[:, 3:7] = self.slerp(t, data[:, 3:7][:, None, :])[:, 0, :]
    result[:, 7:] = self.interpolate(t, data[:, 7:])
    return result

  def resample(
    self, new_times: Optional[np.ndarray] = None, target_dt: Optional[float] = None
  ) -> "TimeSeries":
    # Generate new times if target_dt is provided.
    if new_times is None:
      if target_dt is None:
        raise ValueError("Either new_times or target_dt must be provided")
      if target_dt <= 0:
        raise ValueError("target_dt must be a positive float")

      # Create evenly spaced timestamps.
      new_nsteps = int(np.ceil((self.times[-1] - self.times[0]) / target_dt)) + 1
      new_times = np.linspace(self.times[0], self.times[-1], new_nsteps, endpoint=True)
    else:
      # Make sure new_times is valid.
      if new_times.ndim != 1:
        raise ValueError("new_times must be a 1D array")
      if not np.all(np.diff(new_times) > 0):
        raise ValueError("new_times must be strictly increasing")

    data = {}
    for key, interp_method in _INTERP_MAP.items():
      arr = getattr(self, key)  # (t, n, d)
      if interp_method == "linear":
        data[key] = self.interpolate(new_times, arr)
      elif interp_method == "special":
        data[key] = self.special_interpolate(new_times, arr)
      else:
        data[key] = self.slerp(new_times, arr)

    return TimeSeries(times=new_times, **data)

  def scale_time(self, scale: float) -> "TimeSeries":
    """Scales the time series by a factor.

    Args:
      scale: The time scale factor. Values > 1 make the motion slower, < 1 make it faster.

    Returns:
      A new TimeSeries with scaled timestamps and adjusted velocities.
    """
    if scale <= 0:
      raise ValueError("Time scale must be positive")

    new_times = self.times * scale

    # Scale velocities inversely (since v = dx/dt).
    data = {
      "times": new_times,
      "qpos": self.qpos.copy(),
      "xpos": self.xpos.copy(),
      "xquat": self.xquat.copy(),
      "qvel": self.qvel / scale,
      "linvel": self.linvel / scale,
      "angvel": self.angvel / scale,
      "com": self.com.copy(),
    }

    return TimeSeries(**data)

  def repeat_last_frame(self, t: float) -> "TimeSeries":
    """Repeats the last frame for t seconds.

    Args:
      t: Duration in seconds to repeat the last frame.

    Returns:
      A new TimeSeries that extends the original one by repeating the last frame.
    """
    if t <= 0:
      raise ValueError("Duration must be positive")

    # Calculate number of frames needed based on the average dt.
    avg_dt = np.mean(np.diff(self.times))
    n_frames = int(np.ceil(t / avg_dt))

    new_times = np.concatenate(
      [self.times, self.times[-1] + np.linspace(avg_dt, t, n_frames)]
    )

    data = {
      "times": new_times,
      "qpos": np.concatenate([self.qpos, np.tile(self.qpos[-1], (n_frames, 1))]),
      "xpos": np.concatenate([self.xpos, np.tile(self.xpos[-1], (n_frames, 1, 1))]),
      "xquat": np.concatenate([self.xquat, np.tile(self.xquat[-1], (n_frames, 1, 1))]),
      "qvel": np.concatenate([self.qvel, np.zeros((n_frames, self.qvel.shape[1]))]),
      "linvel": np.concatenate(
        [self.linvel, np.zeros((n_frames, self.linvel.shape[1], 3))]
      ),
      "angvel": np.concatenate(
        [self.angvel, np.zeros((n_frames, self.angvel.shape[1], 3))]
      ),
      "com": np.concatenate([self.com, np.tile(self.com[-1], (n_frames, 1))]),
    }

    return TimeSeries(**data)
