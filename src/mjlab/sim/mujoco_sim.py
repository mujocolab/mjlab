"""CPU simulation backend: C MuJoCo stepped across a thread pool by mjbatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import mjbatch
import mujoco
import numpy as np
import torch

from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim.sim import SimulationCfg
from mjlab.utils.nan_guard import NanGuard

if TYPE_CHECKING:
  import mujoco_warp as mjwarp


class _Data:
  """mjData fields as (num_envs, ...) float32 tensors sharing memory with the batch.

  A field is bound on first access, so a step only copies out what has been read.
  Layouts match the MJWarp bridge.
  """

  def __init__(self, batch: mjbatch.Batch, mj_data: mujoco.MjData) -> None:
    self.nworld = batch.num_sims
    self._batch = batch
    self._mj_data = mj_data

  def __getattr__(self, name: str) -> torch.Tensor:
    if name.startswith("_"):
      raise AttributeError(name)
    is_mjtnum = np.asarray(getattr(self._mj_data, name)).dtype == np.float64
    field = torch.from_numpy(self._batch.bind(name, np.float32 if is_mjtnum else None))
    if name.endswith("xmat"):
      field = field.unflatten(-1, (3, 3))
    # A field bound between physics calls is empty until the next one.
    self._batch.forward()
    self.__dict__[name] = field
    return field


class _Model:
  """mjModel fields laid out like the MJWarp bridge.

  Floating-point fields are (num_envs, ...) float32 tensors: a read-only broadcast of
  the compiled model, or per-env storage that the physics reads once expanded.
  Everything else comes from the compiled model as is.
  """

  def __init__(self, batch: mjbatch.Batch, mj_model: mujoco.MjModel) -> None:
    self._batch = batch
    self._mj_model = mj_model

  def __getattr__(self, name: str) -> Any:
    if name.startswith("_"):
      raise AttributeError(name)
    value = getattr(self._mj_model, name)
    if not isinstance(value, np.ndarray):
      return value
    field = torch.as_tensor(value)
    if field.is_floating_point():
      field = field.float().expand(self._batch.num_sims, *field.shape)
    return self._cache(name, field)

  def expand(self, name: str) -> None:
    is_mjtnum = getattr(self._mj_model, name).dtype == np.float64
    values = self._batch.expand(name, np.float32 if is_mjtnum else None)
    self._cache(name, torch.from_numpy(values))

  def _cache(self, name: str, field: torch.Tensor) -> torch.Tensor:
    if name == "geom_aabb":
      field = field.unflatten(-1, (2, 3))
    self.__dict__[name] = field
    return field


class MujocoSimulation:
  """CPU simulation: C MuJoCo stepped across a thread pool by mjbatch.

  Presents the same model and data tensors as the MJWarp :class:`Simulation`, on the
  CPU. Cameras, raycast sensors, and mesh variants need MJWarp.
  """

  def __init__(self, num_envs: int, cfg: SimulationCfg, model: mujoco.MjModel) -> None:
    self.cfg = cfg
    self.num_envs = num_envs
    self.device = "cpu"

    cfg.mujoco.apply(model)
    # MJWarp lets a diverged world carry its NaNs; C MuJoCo would reset it silently.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
    self._mj_model = model
    self._mj_data = mujoco.MjData(model)
    mujoco.mj_forward(model, self._mj_data)

    self._batch = mjbatch.Batch(model, num_envs, cfg.nthread or 0)
    self._data = _Data(self._batch, self._mj_data)
    self._model = _Model(self._batch, model)
    self._reset_mask = np.zeros(num_envs, dtype=bool)
    self._default_model_fields: dict[str, torch.Tensor] = {}
    self._expanded_fields: set[str] = set()
    self.nan_guard = NanGuard(cfg.nan_guard, num_envs, model)

  # Properties.

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mj_data(self) -> mujoco.MjData:
    return self._mj_data

  @property
  def data(self) -> mjwarp.Data:
    return cast("mjwarp.Data", self._data)

  @property
  def model(self) -> mjwarp.Model:
    return cast("mjwarp.Model", self._model)

  @property
  def expanded_fields(self) -> set[str]:
    """Names of model fields that have been expanded for per-env DR."""
    return self._expanded_fields

  @property
  def per_world_default_fields(self) -> set[str]:
    """Fields with per-world defaults; none without mesh variants."""
    return set()

  # Methods.

  def expand_model_fields(self, fields: tuple[str, ...]) -> None:
    """Expand model fields to support per-environment parameters."""
    for field in fields:
      self._model.expand(field)
    self._expanded_fields.update(fields)

  def get_default_field(self, field: str) -> torch.Tensor:
    """Get the compiled model's value for a field, before any randomization."""
    if field not in self._default_model_fields:
      self._default_model_fields[field] = torch.as_tensor(
        getattr(self._mj_model, field), dtype=getattr(self._model, field).dtype
      ).clone()
    return self._default_model_fields[field]

  def recompute_constants(self, level: RecomputeLevel) -> None:
    """Recompute derived model constants after domain randomization.

    mj_setConst covers every level, so ``level`` does not select anything here.
    """
    del level
    self._batch.set_const()

  def forward(self) -> None:
    self._batch.forward()

  def step(self) -> None:
    with self.nan_guard.watch(self.data):
      self._batch.step()

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    ids = None
    if env_ids is not None:
      self._reset_mask[:] = False
      self._reset_mask[env_ids.numpy()] = True
      ids = self._reset_mask
    # mjbatch replays writes made since the last physics call on top of the reset;
    # consume them first so a reset env starts from the model defaults.
    self._batch.forward(ids)
    self._batch.reset(ids)

  def sense(self) -> None:
    """Nothing to do: the sensors that render need the MJWarp backend."""
