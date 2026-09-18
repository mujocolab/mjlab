"""MuJoCo simulation using mujoco.rollout."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import mujoco
import mujoco.rollout
import numpy as np
import psutil
import torch

from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim.sim import SimulationCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg


@dataclass
class MujocoSimData:
  """Simulation data container for the MuJoCo backend.

  Mirrors the mjwarp.Data interface for the subset of fields the MuJoCo path uses:
  nworld, time, qpos, qvel, ctrl, sensordata, mocap_pos/mocap_quat (initialized
  from model defaults), and zeroed xfrc/qfrc buffers that the native viewer reads
  when rendering.

  Derived body kinematics (xpos/xquat/subtree_com/cvel) are intentionally absent
  for efficiency as they are not produced by `mujoco.rollout`. Consumers that need
  them via entity.data have two options:
    1. Add sensors for all necesary derived body kinematics
    2. Use ``MujocoSimDataWithKinematics`` instead, see the
  ``mujoco_with_kinematics`` backend and the MujocoKinematics class.
  Option 1 is best for applications where speed is important
  """

  nworld: int
  time: torch.Tensor  # [N]
  qpos: torch.Tensor  # [N, nq]
  qvel: torch.Tensor  # [N, nv]
  ctrl: torch.Tensor  # [N, nu]
  sensordata: torch.Tensor  # [N, nsensordata]

  # mocap_pos and mocap_quat flow into mujoco.rollout via control_spec.
  mocap_pos: torch.Tensor  # [N, nmocap, 3]
  mocap_quat: torch.Tensor  # [N, nmocap, 4]

  # xfrc_applied and qfrc_applied are placeholder buffers for the native viewer
  # only — they do not flow into rollout.
  xfrc_applied: torch.Tensor  # [N, nbody, 6]
  qfrc_applied: torch.Tensor  # [N, nv]


@dataclass
class MujocoSimDataWithKinematics(MujocoSimData):
  """``MujocoSimData`` augmented with derived body kinematics.

  Consumed by entity.data accessors (body_link_*, root_*). mujoco.rollout does not
  write these back, so they are refreshed externally each substep by a
  ``MujocoKinematics`` instance which writes through these
  torch tensors.
  """

  xpos: torch.Tensor  # [N, nbody, 3]
  xquat: torch.Tensor  # [N, nbody, 4]
  subtree_com: torch.Tensor  # [N, nbody, 3]
  cvel: torch.Tensor  # [N, nbody, 6]
  site_xpos: torch.Tensor  # [N, nsite, 3]
  site_xmat: torch.Tensor  # [N, nsite, 3, 3]


class MujocoModelField:
  """Write-through bridge tensor for a single per-entity model field.

  Wraps a [N, nentity, ...] tensor. __setitem__ writes to the tensor and
  immediately copies the updated env rows to the per-env MjModel numpy arrays,
  so read-after-write is always consistent (no separate sync step needed).
  """

  def __init__(
    self,
    data: torch.Tensor,
    env_models: list[mujoco.MjModel],
    field: str,
  ) -> None:
    self._data = data
    self._env_models = env_models  # mutated in-place by expand_model_fields
    self._field = field

  def __getitem__(self, key: Any) -> torch.Tensor:
    return self._data[key]

  def __setitem__(self, key: Any, value: Any) -> None:
    env_key = key[0] if isinstance(key, tuple) else key
    # All DR write sites should use integer, tensor, or list keys — never a bare slice.
    if isinstance(env_key, slice):
      raise TypeError(
        f"MujocoModelField does not support slice indexing at the env "
        f"dimension; got key={key!r}. Use an explicit env_ids tensor."
      )
    num_envs = self._data.shape[0]
    if len(self._env_models) != num_envs:
      raise RuntimeError(
        f"MujocoModelField.__setitem__ called before expand_model_fields: "
        f"_env_models has {len(self._env_models)} entries but num_envs={num_envs}. "
        "Call sim.expand_model_fields() before any DR write."
      )
    self._data[key] = value
    env_ids = torch.as_tensor(env_key).flatten().unique()
    # Slice only the affected rows to avoid a full device-to-host transfer.
    rows = self._data[env_ids].cpu().numpy()
    for idx, i in enumerate(env_ids.tolist()):
      numpy_arr = getattr(self._env_models[i], self._field)
      numpy_arr[:] = rows[idx].reshape(numpy_arr.shape)

  def __len__(self) -> int:
    return len(self._data)

  def __repr__(self) -> str:
    return (
      f"MujocoModelField(field={self._field!r}, "
      f"shape={self._data.shape}, dtype={self._data.dtype})"
    )

  def __getattr__(self, name: str) -> Any:
    """All unknown attributes are passed through to _data."""
    if name == "_data":
      # Without this check, accesing self._data before it's initialized would
      # cause infinite recursion
      raise AttributeError(name)
    return getattr(self._data, name)

  @property
  def shape(self) -> torch.Size:
    return self._data.shape

  @property
  def dtype(self) -> torch.dtype:
    return self._data.dtype

  @property
  def device(self) -> torch.device:
    return self._data.device


class MujocoBridgeAuxStruct:
  """Struct-like wrapper for MuJoCo fields converting numpy arrays to torch tensors."""

  def __init__(self, aux_struct: Any, device: str):
    self._aux_struct = aux_struct
    self._device = device
    self._field_cache: dict[str, Any] = {}

  def __getattr__(self, name: str) -> Any:

    if name not in self._field_cache:
      attr = getattr(self._aux_struct, name)
      if isinstance(attr, np.ndarray):
        dtype = torch.float32 if np.issubdtype(attr.dtype, np.floating) else None
        self._field_cache[name] = torch.tensor(
          attr, dtype=dtype, device=self._device
        ).unsqueeze(0)
        return self._field_cache[name]
      self._field_cache[name] = attr
    return self._field_cache[name]


class MujocoModelBridge:
  """Model bridge for the MuJoCo backend.

  Lazily wraps per-entity MjModel fields as write-through MujocoModelField
  tensors. Writes to these tensors immediately sync the updated env rows to
  the per-env MjModel numpy arrays, keeping the tensor and MjModel consistent.

  Special cases:
  - ``geom_type``: read-only 1D plain tensor (DR never writes geometry types).
  """

  def __init__(
    self,
    num_envs: int,
    template: mujoco.MjModel,
    env_models: list[mujoco.MjModel],
    device: str,
  ) -> None:
    self._num_envs = num_envs
    self._template = template
    self._env_models = env_models  # same list as MujocoSimulation._env_models
    self._device = device
    self._field_cache: dict[str, MujocoModelField] = {}
    self.opt = MujocoBridgeAuxStruct(template.opt, device=device)
    self.vis = MujocoBridgeAuxStruct(template.vis, device=device)
    self.stat = MujocoBridgeAuxStruct(template.stat, device=device)

    # geom_type is (ngeom,) int, not per-world, so it doesn't need to be a MujocoModelField
    self.geom_type: torch.Tensor = torch.tensor(
      self._template.geom_type, dtype=torch.int32, device=self._device
    )

  def __getattr__(self, name: str) -> MujocoModelField:
    """Pass MujocoModelBridge.<field_name> access through to the environment models.

    Lazily creates a MujocoModelField for each field.
    """

    if name not in self._field_cache:
      attr = getattr(self._template, name)
      if type(attr) is np.ndarray:
        stacked = np.tile(attr, (self._num_envs,) + (1,) * attr.ndim)
        dtype = torch.float32 if np.issubdtype(attr.dtype, np.floating) else None
      elif isinstance(attr, bool):
        stacked = np.full((self._num_envs,), attr)
        dtype = torch.bool
      elif isinstance(attr, (int, float)):
        stacked = np.full((self._num_envs,), attr)
        dtype = torch.float32 if isinstance(attr, float) else None
      elif isinstance(attr, bytes):
        stacked = attr
        dtype = bytes
      else:
        raise TypeError(
          f"Unsupported type for MujocoModelBridge field '{name}': {type(attr)}"
        )
      # geom_aabb is stored flat (ngeom, 6) in numpy but warp/DR code expects
      # (ngeom, 2, 3) — reshape so the tensor layout matches.
      if name == "geom_aabb" and stacked.ndim == 3 and stacked.shape[-1] == 6:
        stacked = stacked.reshape(self._num_envs, stacked.shape[1], 2, 3)

      # Float MjModel fields are stored as float64 (mjtNum); downcast to
      # float32 to match the rest of the public tensor surface. Integer
      # fields (e.g., geom_contype) preserve their native dtype so the
      # write-through sync round-trips into the int MjModel arrays without
      # lossy float casting.
      if dtype is bytes:
        tensor = stacked
        self._field_cache[name] = MujocoModelField(tensor, self._env_models, name)
      else:
        tensor = torch.as_tensor(stacked.copy(), dtype=dtype).to(self._device)
        self._field_cache[name] = MujocoModelField(tensor, self._env_models, name)

    return self._field_cache[name]


class MujocoSimulation:
  """MuJoCo simulation using mujoco.rollout.

  Mirrors the Simulation interface for the subset of methods ManagerBasedRlEnv uses.
  The following features are not supported: heterogeneous worlds, domain randomization, and raycast
  sensors.

  Physics always runs on CPU via mujoco.rollout. The public tensors (data.qpos,
  data.qvel, etc.) live on ``device``, which may be a GPU. step() shuttles state
  between the device tensors and the float64 numpy rollout buffers; the float32↔
  float64 conversion already requires a copy on both paths, so there is no
  meaningful performance difference between CPU and GPU devices.
  """

  # cfg.backend must equal this value
  _BACKEND: ClassVar[str] = "mujoco"

  def __init__(
    self,
    num_envs: int,
    cfg: SimulationCfg,
    model: mujoco.MjModel | None = None,
    device: str = "cpu",
    *,
    spec: mujoco.MjSpec | None = None,
  ) -> None:
    """Initialize MujocoSimulation.

    Args:
        num_envs: Number of parallel simulation environments.
        cfg: Simulation configuration; ``cfg.mujoco`` is applied to the compiled
            model to set physics options (timestep, solver, etc.).
        model: Pre-compiled :class:`mujoco.MjModel`. Mutually exclusive with
            ``spec``.
        device: PyTorch device for public tensors, e.g. ``"cpu"`` or
            ``"cuda:0"``. Physics always runs on CPU regardless of this value.
        spec: :class:`mujoco.MjSpec` to compile into a model. Mutually exclusive
            with ``model``.

    Raises:
        ValueError: If neither ``model`` nor ``spec`` is provided.
    """
    if cfg.backend != self._BACKEND:
      msg = (
        f'For {type(self).__name__}, cfg.backend must be "{self._BACKEND}", '
        f'got "{cfg.backend}"'
      )
      raise ValueError(msg)

    if spec is not None:
      compiled = spec.compile()
      cfg.mujoco.apply(compiled)
      mj_model = compiled
    elif model is not None:
      cfg.mujoco.apply(model)
      mj_model = model
    else:
      raise ValueError("Either model or spec must be provided.")

    self._mj_model = mj_model
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    # Scratch MjData for the native viewer only (see mj_data property for details).
    self._mj_data = mj_data

    self.cfg = cfg
    self.num_envs = num_envs
    self.device = device

    nsensordata = mj_model.nsensordata
    ntime = mujoco.mj_stateSize(mj_model, mujoco.mjtState.mjSTATE_TIME)
    self._nstate = mujoco.mj_stateSize(mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    # FULLPHYSICS state layout: time(1) + qpos(nq) + qvel(nv) + act + ...
    self._slice_time = slice(0, ntime)
    self._slice_qpos = slice(self._slice_time.stop, self._slice_time.stop + mj_model.nq)
    self._slice_qvel = slice(self._slice_qpos.stop, self._slice_qpos.stop + mj_model.nv)

    if self._nstate < self._slice_qvel.stop:
      msg = f"FULLPHYSICS state size {self._nstate} < expected minimum {self._slice_qvel.stop}"
      raise ValueError(msg)

    # The control vector contains ctrl, mocap_pos, and mocap_quat
    self._ctrl_spec = int(
      mujoco.mjtState.mjSTATE_CTRL
      | mujoco.mjtState.mjSTATE_MOCAP_POS
      | mujoco.mjtState.mjSTATE_MOCAP_QUAT
    )
    _ncontrol = mujoco.mj_stateSize(mj_model, self._ctrl_spec)

    _ctrl_size = mujoco.mj_stateSize(mj_model, mujoco.mjtState.mjSTATE_CTRL)
    _mocap_pos_size = mujoco.mj_stateSize(mj_model, mujoco.mjtState.mjSTATE_MOCAP_POS)
    _mocap_quat_size = mujoco.mj_stateSize(mj_model, mujoco.mjtState.mjSTATE_MOCAP_QUAT)
    self._slice_ctrl = slice(0, _ctrl_size)
    self._slice_mocap_pos = slice(
      self._slice_ctrl.stop,
      self._slice_ctrl.stop + _mocap_pos_size,
    )
    self._slice_mocap_quat = slice(
      self._slice_mocap_pos.stop,
      self._slice_mocap_pos.stop + _mocap_quat_size,
    )
    if self._slice_mocap_quat.stop != _ncontrol:
      msg = (
        f"ctrl_buf layout mismatch: expected {_ncontrol}, "
        f"got {self._slice_mocap_quat.stop}"
      )
      raise ValueError(msg)

    default_state = np.zeros(self._nstate, dtype=np.float64)
    mujoco.mj_getState(
      mj_model, mj_data, default_state, mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    # Replicate the single default state across all envs.
    default_state_tiled = np.tile(default_state, (num_envs, 1))  # [N, nstate]

    # Pre-computed defaults on the target device; copy_ in reset() avoids
    # repeated float64→float32 conversions at runtime.
    self._default_time = torch.tensor(
      default_state_tiled[:, self._slice_time].squeeze(-1),
      dtype=torch.float32,
      device=device,
    )
    self._default_qpos = torch.tensor(
      default_state_tiled[:, self._slice_qpos], dtype=torch.float32, device=device
    )
    self._default_qvel = torch.tensor(
      default_state_tiled[:, self._slice_qvel], dtype=torch.float32, device=device
    )
    self._default_mocap_pos = torch.tensor(
      np.tile(mj_data.mocap_pos, (num_envs, 1, 1)),
      dtype=torch.float32,
      device=device,
    )
    self._default_mocap_quat = torch.tensor(
      np.tile(mj_data.mocap_quat, (num_envs, 1, 1)),
      dtype=torch.float32,
      device=device,
    )

    # mujoco.rollout only writes qpos/qvel/sensordata back; _build_data lets the
    # WithKinematics subclass attach the derived-kinematics buffers it needs.
    self._data = self._build_data(
      nworld=num_envs,
      time=self._default_time.clone(),
      qpos=self._default_qpos.clone(),
      qvel=self._default_qvel.clone(),
      ctrl=torch.zeros(num_envs, mj_model.nu, dtype=torch.float32, device=device),
      sensordata=torch.zeros(num_envs, nsensordata, dtype=torch.float32, device=device),
      mocap_pos=self._default_mocap_pos.clone(),
      mocap_quat=self._default_mocap_quat.clone(),
      xfrc_applied=torch.zeros(
        num_envs, mj_model.nbody, 6, dtype=torch.float32, device=device
      ),
      qfrc_applied=torch.zeros(
        num_envs, mj_model.nv, dtype=torch.float32, device=device
      ),
    )

    # WARNING: MujocoModelField instances hold a direct reference to self._env_models,
    # so it should never be reassigned, only modified in-place. To replace all the
    # models, use self._env_models.clear() and self._env_models.extend().
    self._env_models: list[mujoco.MjModel] = [mj_model]

    self._mj_models_expanded: bool = False
    self._default_fields: dict[str, torch.Tensor] = {}
    self._expanded_field_names: set[str] = set()
    self._model = MujocoModelBridge(
      num_envs=num_envs,
      template=mj_model,
      env_models=self._env_models,
      device=self.device,
    )
    nthread = min(
      num_envs, psutil.cpu_count(logical=True) or 1
    )  # psutil.cpu_count() returns None if undetermined
    self._rollout = mujoco.rollout.Rollout(nthread=nthread)
    self._thread_data = [mujoco.MjData(mj_model) for _ in range(nthread)]

    # Pre-allocated float64 numpy buffers for mujoco.rollout (always CPU).
    self._state_buf = np.zeros((num_envs, self._nstate), dtype=np.float64)
    self._ctrl_buf = np.zeros((num_envs, 1, _ncontrol), dtype=np.float64)
    self._result_state_buf = np.zeros((num_envs, 1, self._nstate), dtype=np.float64)
    self._result_sensordata_buf = np.zeros((num_envs, 1, nsensordata), dtype=np.float64)

  def _build_data(self, **kwargs: Any) -> MujocoSimData:
    """Construct the data container from the common fields.

    Overridden by MujocoSimulationWithKinematics to attach the derived-kinematics
    buffers. Called from __init__ after mj_model/mj_data/num_envs/device are set.
    """
    return MujocoSimData(**kwargs)

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mj_data(self) -> mujoco.MjData:
    """Scratch MjData for the native viewer.

    The base sim never writes to this buffer; mujoco.rollout advances its own
    per-thread MjData. The native viewer is responsible for syncing the
    current sim state (qpos, qvel, ctrl, mocap_pos/quat, xfrc_applied)
    into this buffer and calling mj_forward before each rendered frame.

    Exception: MujocoSimulationWithKinematics uses this buffer as the scratch for
    its kinematics engine, so it is forwarded on every step()/reset()/forward().
    """
    return self._mj_data

  @property
  def data(self) -> MujocoSimData:
    return self._data

  @property
  def model(self) -> MujocoModelBridge:
    return self._model

  @property
  def expanded_fields(self) -> set[str]:
    return self._expanded_field_names

  def _run_rollout(self) -> None:
    """Pack current state into numpy buffers and run mujoco.rollout for one step.

    Results are stored in self._result_state_buf and self._result_sensordata_buf.
    """
    # The float32->float64 upcast into _state_buf/_ctrl_buf copies on all paths.
    # .cpu() is a no-op on CPU tensors; on GPU tensors it does a device-to-host transfer.
    # _data.time is [N]; the _slice_time column is [N, 1], so reshape to match
    # (mirrors the .squeeze(-1) on the store-back path in _store_rollout_results).
    self._state_buf[:, self._slice_time] = (
      self._data.time.cpu().numpy().reshape(self.num_envs, -1)
    )
    self._state_buf[:, self._slice_qpos] = self._data.qpos.cpu().numpy()
    self._state_buf[:, self._slice_qvel] = self._data.qvel.cpu().numpy()
    self._ctrl_buf[:, 0, self._slice_ctrl] = self._data.ctrl.cpu().numpy()
    if self._mj_model.nmocap > 0:
      self._ctrl_buf[:, 0, self._slice_mocap_pos] = (
        self._data.mocap_pos.cpu().numpy().reshape(self.num_envs, -1)
      )
      self._ctrl_buf[:, 0, self._slice_mocap_quat] = (
        self._data.mocap_quat.cpu().numpy().reshape(self.num_envs, -1)
      )

    model = self._env_models if self._mj_models_expanded else self._mj_model
    self._rollout.rollout(
      model=model,
      data=self._thread_data,
      initial_state=self._state_buf,
      control=self._ctrl_buf,
      control_spec=self._ctrl_spec,
      nstep=1,
      state=self._result_state_buf,
      sensordata=self._result_sensordata_buf,
    )

  def _store_rollout_results(
    self, update_state: bool = True, update_sensordata: bool = True
  ) -> None:
    """Store the results from a rollout.

    Args:
        update_state (bool, optional): Whether to update the state (time,
          qpos, qvel). Defaults to True.
        update_sensordata (bool, optional): Whether to update sensordata.
          Defaults to True.
    """
    if update_state:
      # copy_() casts float64→float32 and handles host-to-device transfer in one call.
      self._data.time.copy_(
        torch.from_numpy(self._result_state_buf[:, 0, self._slice_time].squeeze(-1))
      )
      self._data.qpos.copy_(
        torch.from_numpy(self._result_state_buf[:, 0, self._slice_qpos])
      )
      self._data.qvel.copy_(
        torch.from_numpy(self._result_state_buf[:, 0, self._slice_qvel])
      )

    if update_sensordata and self._result_sensordata_buf.shape[-1] > 0:
      self._data.sensordata.copy_(
        torch.from_numpy(self._result_sensordata_buf[:, 0, :])
      )

  def step(self) -> None:
    """Advance physics by one step for all envs via mujoco.rollout."""
    self._run_rollout()
    self._store_rollout_results(update_state=True, update_sensordata=True)

  def forward(self) -> None:
    """Update sensordata for all envs via a one-step rollout.

    time, qpos, and qvel are not modified.
    """
    if self._data.sensordata.shape[-1] == 0:
      return
    self._run_rollout()
    self._store_rollout_results(update_state=False, update_sensordata=True)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Restore default time, qpos, qvel, ctrl, mocap_pos, and mocap_quat for the specified envs.

    Args:
        env_ids: 1D integer tensor of environment indices to reset. Resets all
            environments when ``None``.
    """
    if env_ids is None:
      self._data.time.copy_(self._default_time)
      self._data.qpos.copy_(self._default_qpos)
      self._data.qvel.copy_(self._default_qvel)
      self._data.ctrl.zero_()
      self._data.mocap_pos.copy_(self._default_mocap_pos)
      self._data.mocap_quat.copy_(self._default_mocap_quat)
    else:
      self._data.time[env_ids] = self._default_time[env_ids]
      self._data.qpos[env_ids] = self._default_qpos[env_ids]
      self._data.qvel[env_ids] = self._default_qvel[env_ids]
      self._data.ctrl[env_ids] = 0.0
      self._data.mocap_pos[env_ids] = self._default_mocap_pos[env_ids]
      self._data.mocap_quat[env_ids] = self._default_mocap_quat[env_ids]

  def sense(self) -> None:
    """No-op: sensordata is populated by rollout during step()."""

  def set_sensor_context(self, ctx: object) -> None:
    """No-op: no GPU sensor context for the MuJoCo backend."""

  def expand_model_fields(self, fields: tuple[str, ...]) -> None:
    """Expand model to per-env copies, enabling per-env domain randomization.

    Args:
        fields: Names of model fields that will be written per-env. Used only
            to update ``expanded_fields``; all fields are available for write
            after expansion regardless.
    """
    if not fields:
      return
    self._expanded_field_names.update(fields)
    if self._mj_models_expanded:
      return
    new_models = [copy.copy(self._mj_model) for _ in range(self.num_envs)]
    self._env_models.clear()
    self._env_models.extend(new_models)
    self._mj_models_expanded = True
    nthread = len(self._thread_data)
    # Thread data is only used for scratch, so it's safe just to use env_models[0].
    # All that models is the shape of the data, which must be the same for all envs.
    self._thread_data = [mujoco.MjData(self._env_models[0]) for _ in range(nthread)]

  def recompute_constants(self, level: RecomputeLevel) -> None:
    """Recompute derived model constants after domain randomization.

    Runs mujoco.mj_setConst on every per-env MjModel using a single shared
    MjData scratch.

    Args:
        level: Minimum recompute level required. ``RecomputeLevel.none`` is a no-op.
    """
    if level == RecomputeLevel.none:
      return
    scratch = mujoco.MjData(self._env_models[0])
    for mj_model in self._env_models:
      mujoco.mj_setConst(mj_model, scratch)

  def get_default_field(self, field: str) -> torch.Tensor:
    """Return the compile-time default value for a model field, cached for reuse.

    Returns the original values from the template MjModel before any
    domain randomization is applied.

    Args:
        field: Name of the model field (e.g. ``"body_mass"``).

    Returns:
        A ``[nentity, ...]`` float32 tensor on ``self.device``.
    """
    if field not in self._default_fields:
      arr = getattr(self._mj_model, field)
      self._default_fields[field] = torch.as_tensor(
        arr.copy(), dtype=torch.float32, device=self.device
      )
    return self._default_fields[field]

  @property
  def per_world_default_fields(self) -> set[str]:
    return set()

  # Environment and Model setup methods

  @classmethod
  def setup_cfg(cls, cfg: ManagerBasedRlEnvCfg) -> None:
    """Apply sim backend-specific fixups to the environment configuration."""
    del cfg  # unused

  def setup_model(self, model: mujoco.MjModel) -> None:
    """Apply sim backend-specific fixups to the model."""
    del model  # unused


class MujocoKinematics:
  """Class to compute kinematic data from MjData that MujocoSimulation does not compute.

  mujoco.rollout only writes qpos/qvel/sensordata back to the sim, so body/geom poses
  and com-based quantities must be recomputed from qpos/qvel. ``forward`` runs
  ``mj_kinematics -> mj_comPos -> mj_comVel`` on a scratch MjData for every env and
  fills body_xpos/body_xmat/body_xquat/subtree_com/cvel/site_xpos/site_xmat (plus
  mocap/ctrl snapshots). That chain is sufficient for these fields and avoids the
  collision/constraint work a full mj_forward would do.

  This class is intended for consumers that require kinematic data for all bodies.
  The viewer holds one directly; MujocoSimulationWithKinematics owns one internally
  and refreshes it from step()/reset()/forward(), so most callers never construct one.
  RL training with the plain MuJoCo backend should use sensors instead.

  When the sim's ``data`` is a MujocoSimDataWithKinematics (the
  ``mujoco_with_kinematics`` backend), ``forward`` additionally publishes the computed
  fields into ``data`` so entity.data accessors observe the current state; against a
  plain MujocoSimulation it only fills this object's body_* buffers.
  """

  def __init__(
    self, sim: MujocoSimulation, scratch_mjdata: mujoco.MjData | None = None
  ):
    num_envs = sim.num_envs
    mj_model = sim.mj_model
    # Caller may supply an existing MjData to forward in place (e.g. the sim's scratch
    # buffer the native viewer renders from); otherwise allocate a private one.
    self.scratch_mjdata: mujoco.MjData = (
      scratch_mjdata if scratch_mjdata is not None else mujoco.MjData(mj_model)
    )
    self.body_xpos: np.ndarray = np.empty((num_envs, mj_model.nbody, 3))
    self.body_xmat: np.ndarray = np.empty((num_envs, mj_model.nbody, 3, 3))
    self.body_xquat: np.ndarray = np.empty((num_envs, mj_model.nbody, 4))
    self.subtree_com: np.ndarray = np.empty((num_envs, mj_model.nbody, 3))
    self.cvel: np.ndarray = np.empty((num_envs, mj_model.nbody, 6))
    self.site_xpos: np.ndarray = np.empty((num_envs, mj_model.nsite, 3))
    self.site_xmat: np.ndarray = np.empty((num_envs, mj_model.nsite, 3, 3))
    self._on_cpu: bool = torch.device(sim.device).type == "cpu"
    if self._on_cpu:
      # On CPU, forward() assigns views into data.qpos/data.qvel; no buffer needed.
      # Placeholders satisfy the type annotation until the first forward() call.
      self.qpos: np.ndarray = np.empty((0,))
      self.qvel: np.ndarray = np.empty((0,))
    else:
      self.qpos = np.empty((num_envs, mj_model.nq))
      self.qvel = np.empty((num_envs, mj_model.nv))

    self.mocap_pos: np.ndarray | None
    self.mocap_quat: np.ndarray | None
    if mj_model.nmocap > 0:
      self.mocap_pos = np.empty((num_envs, mj_model.nmocap, 3))
      self.mocap_quat = np.empty((num_envs, mj_model.nmocap, 4))
    else:
      self.mocap_pos = None
      self.mocap_quat = None

    self.ctrl: np.ndarray | None
    if mj_model.nu > 0:
      self.ctrl = np.empty((num_envs, mj_model.nu))
    else:
      self.ctrl = None

  def forward(self, sim: MujocoSimulation) -> None:
    """Recompute derived kinematics from the current sim state, for every env.

    Fills this object's body_xpos/body_xmat/body_xquat/subtree_com/cvel/site_xpos/
    site_xmat buffers and, when ``sim.data`` is a MujocoSimDataWithKinematics, copies
    them into ``sim.data``.

    Args:
        sim (MujocoSimulation): The simulation.
    """
    mj_model = sim.mj_model
    data = sim.data
    if self._on_cpu:
      self.qpos = data.qpos.numpy()
      self.qvel = data.qvel.numpy()
    else:
      np.copyto(self.qpos, data.qpos.cpu().numpy())
      np.copyto(self.qvel, data.qvel.cpu().numpy())
    if self.mocap_pos is not None:
      np.copyto(self.mocap_pos, data.mocap_pos.cpu().numpy())
    if self.mocap_quat is not None:
      np.copyto(self.mocap_quat, data.mocap_quat.cpu().numpy())
    if self.ctrl is not None:
      np.copyto(self.ctrl, data.ctrl.cpu().numpy())

    mjdata = self.scratch_mjdata
    for i in range(sim.num_envs):
      mjdata.qpos[:] = self.qpos[i]
      mjdata.qvel[:] = self.qvel[i]
      # mocap drives the world pose of mocap-attached bodies, so it must be set
      # before mj_kinematics for those bodies' xpos/xquat to be correct.
      if self.mocap_pos is not None:
        mjdata.mocap_pos[:] = self.mocap_pos[i]
      if self.mocap_quat is not None:
        mjdata.mocap_quat[:] = self.mocap_quat[i]
      mujoco.mj_kinematics(mj_model, mjdata)
      mujoco.mj_comPos(mj_model, mjdata)  # subtree_com (+ cdof for mj_comVel)
      mujoco.mj_comVel(mj_model, mjdata)  # cvel
      self.body_xpos[i] = mjdata.xpos
      self.body_xmat[i] = mjdata.xmat.reshape(mj_model.nbody, 3, 3)
      self.body_xquat[i] = mjdata.xquat
      self.subtree_com[i] = mjdata.subtree_com
      self.cvel[i] = mjdata.cvel
      self.site_xpos[i] = mjdata.site_xpos
      self.site_xmat[i] = mjdata.site_xmat.reshape(mj_model.nsite, 3, 3)

    # Publish into sim.data only when it carries the kinematics fields. A viewer
    # attached to a plain MujocoSimulation reads body_xpos/body_xmat off this object
    # directly, so there is nothing to write back there.
    if isinstance(data, MujocoSimDataWithKinematics):
      data.xpos.copy_(torch.from_numpy(self.body_xpos))
      data.xquat.copy_(torch.from_numpy(self.body_xquat))
      data.subtree_com.copy_(torch.from_numpy(self.subtree_com))
      data.cvel.copy_(torch.from_numpy(self.cvel))
      data.site_xpos.copy_(torch.from_numpy(self.site_xpos))
      data.site_xmat.copy_(torch.from_numpy(self.site_xmat))


class MujocoSimulationWithKinematics(MujocoSimulation):
  """MujocoSimulation that also exposes derived body kinematics in ``data``.

  Selected via ``cfg.backend = "mujoco_with_kinematics"``.

  The base backend leaves the body_link_*/root_* entity.data accessors unsupported
  because mujoco.rollout does not produce xpos/xquat/subtree_com/cvel. This variant
  adds those buffers to ``data`` and owns a MujocoKinematics engine that refreshes
  them after every step(), reset(), and forward(). Consumers therefore just call
  step()/reset()/forward() as usual; there is no separate kinematics object to keep.

  The engine uses the sim's scratch mj_data (``self._mj_data``) as its working
  buffer, so each refresh also forwards that buffer (body/geom poses) See the ``mj_data`` property.
  """

  _BACKEND: ClassVar[str] = "mujoco_with_kinematics"

  def __init__(
    self,
    num_envs: int,
    cfg: SimulationCfg,
    model: mujoco.MjModel | None = None,
    device: str = "cpu",
    *,
    spec: mujoco.MjSpec | None = None,
  ) -> None:
    """See MujocoSimulation; also builds the kinematics engine and seeds data."""
    super().__init__(num_envs, cfg, model=model, device=device, spec=spec)
    # Own the kinematics engine; its scratch is the sim's mj_data so refreshes also
    # forward that buffer. Constructed after super().__init__ (needs mj_model etc.).
    self._kinematics = MujocoKinematics(self, scratch_mjdata=self._mj_data)
    # Seed data.xpos/... and mj_data from the initial state.
    self._refresh_kinematics()

  def _build_data(self, **kwargs: Any) -> MujocoSimDataWithKinematics:
    mj_data = self._mj_data
    num_envs = self.num_envs

    def _seed(field: np.ndarray) -> torch.Tensor:
      # [d...] -> [N, d...] float32, aliased to torch so MujocoKinematics writes
      # through to data readers.
      tiled = np.repeat(np.expand_dims(field.astype(np.float32), 0), num_envs, axis=0)
      return torch.from_numpy(tiled)

    return MujocoSimDataWithKinematics(
      **kwargs,
      xpos=_seed(mj_data.xpos),
      xquat=_seed(mj_data.xquat),
      subtree_com=_seed(mj_data.subtree_com),
      cvel=_seed(mj_data.cvel),
      site_xpos=_seed(mj_data.site_xpos),
      site_xmat=_seed(mj_data.site_xmat.reshape(-1, 3, 3)),
    )

  def _refresh_kinematics(self) -> None:
    """Recompute derived body kinematics into data (and the scratch mj_data)."""
    self._kinematics.forward(self)

  def step(self) -> None:
    """Refresh derived body kinematics for the current state, then advance physics.

    The refresh runs before the rollout, so after step() the kinematics in ``data``
    (and the scratch mj_data) reflect the pre-step qpos/qvel — the state callers
    observe before the physics advance — and lag the post-step qpos by one step.
    """
    self._refresh_kinematics()
    super().step()

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Reset the given envs, then refresh derived body kinematics."""
    super().reset(env_ids)
    self._refresh_kinematics()

  def forward(self) -> None:
    """Refresh sensordata (via rollout), then derived body kinematics."""
    super().forward()
    self._refresh_kinematics()
