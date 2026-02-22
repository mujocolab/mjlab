from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim.randomization import expand_model_fields
from mjlab.sim.sim_data import TorchArray, WarpBridge
from mjlab.utils.nan_guard import NanGuard, NanGuardCfg

if TYPE_CHECKING:
  from mjlab.sensor.sensor_context import SensorContext

# Type aliases for better IDE support while maintaining runtime compatibility
# At runtime, WarpBridge wraps the actual MJWarp objects.
if TYPE_CHECKING:
  ModelBridge = mjwarp.Model
  DataBridge = mjwarp.Data
else:
  ModelBridge = WarpBridge
  DataBridge = WarpBridge

# Minimum CUDA driver version supported for conditional CUDA graphs.
_GRAPH_CAPTURE_MIN_DRIVER = (12, 4)

_JACOBIAN_MAP = {
  "auto": mujoco.mjtJacobian.mjJAC_AUTO,
  "dense": mujoco.mjtJacobian.mjJAC_DENSE,
  "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
}
_CONE_MAP = {
  "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
  "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
}
_INTEGRATOR_MAP = {
  "euler": mujoco.mjtIntegrator.mjINT_EULER,
  "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
_SOLVER_MAP = {
  "newton": mujoco.mjtSolver.mjSOL_NEWTON,
  "cg": mujoco.mjtSolver.mjSOL_CG,
  "pgs": mujoco.mjtSolver.mjSOL_PGS,
}


@dataclass
class MujocoCfg:
  """Configuration for MuJoCo simulation parameters."""

  # Integrator settings.
  timestep: float = 0.002
  integrator: Literal["euler", "implicitfast"] = "implicitfast"

  # Friction settings.
  impratio: float = 1.0
  cone: Literal["pyramidal", "elliptic"] = "pyramidal"

  # Solver settings.
  jacobian: Literal["auto", "dense", "sparse"] = "auto"
  solver: Literal["newton", "cg", "pgs"] = "newton"
  iterations: int = 100
  tolerance: float = 1e-8
  ls_iterations: int = 50
  ls_tolerance: float = 0.01
  ccd_iterations: int = 50

  # Other.
  gravity: tuple[float, float, float] = (0, 0, -9.81)
  multiccd: bool = False

  def apply(self, model: mujoco.MjModel) -> None:
    """Apply configuration settings to a compiled MjModel."""
    model.opt.jacobian = _JACOBIAN_MAP[self.jacobian]
    model.opt.cone = _CONE_MAP[self.cone]
    model.opt.integrator = _INTEGRATOR_MAP[self.integrator]
    model.opt.solver = _SOLVER_MAP[self.solver]
    model.opt.timestep = self.timestep
    model.opt.impratio = self.impratio
    model.opt.gravity[:] = self.gravity
    model.opt.iterations = self.iterations
    model.opt.tolerance = self.tolerance
    model.opt.ls_iterations = self.ls_iterations
    model.opt.ls_tolerance = self.ls_tolerance
    model.opt.ccd_iterations = self.ccd_iterations
    if self.multiccd:
      model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD


@dataclass(kw_only=True)
class SimulationCfg:
  nconmax: int | None = None
  """Number of contacts to allocate per world.

  Contacts exist in large heterogenous arrays: one world may have more than nconmax
  contacts. If None, a heuristic value is used."""
  njmax: int | None = None
  """Number of constraints to allocate per world.

  Constraint arrays are batched by world: no world may have more than njmax
  constraints. If None, a heuristic value is used."""
  ls_parallel: bool = True  # Boosts perf quite noticeably.
  contact_sensor_maxmatch: int = 64
  mujoco: MujocoCfg = field(default_factory=MujocoCfg)
  nan_guard: NanGuardCfg = field(default_factory=NanGuardCfg)


class Simulation:
  """GPU-accelerated MuJoCo simulation powered by MJWarp.

  CUDA Graph Capture
  ------------------
  On CUDA devices with memory pools enabled, the simulation captures CUDA graphs
  for ``step()``, ``forward()``, and ``reset()`` operations. Graph capture records
  a sequence of GPU kernels and their memory addresses, then replays the entire
  sequence with a single kernel launch, eliminating CPU overhead from repeated
  kernel dispatches.

  **Important:** A captured graph holds pointers to the GPU arrays that existed
  at capture time. If those arrays are later replaced (e.g., via
  ``expand_model_fields()``), the graph will still read from the old arrays,
  silently ignoring any new values. The ``expand_model_fields()`` method handles
  this automatically by calling ``create_graph()`` after replacing arrays.

  If you write code that replaces model or data arrays after simulation
  initialization, you **must** call ``create_graph()`` afterward to re-capture
  the graphs with the new memory addresses.
  """

  def __init__(
    self, num_envs: int, cfg: SimulationCfg, model: mujoco.MjModel, device: str
  ):
    self.cfg = cfg
    self.device = device
    self.wp_device = wp.get_device(self.device)
    self.num_envs = num_envs
    self._default_model_fields: dict[str, torch.Tensor] = {}
    self._expanded_fields: set[str] = set()

    # MuJoCo model and data.
    self._mj_model = model
    cfg.mujoco.apply(self._mj_model)
    self._mj_data = mujoco.MjData(model)
    mujoco.mj_forward(self._mj_model, self._mj_data)

    # MJWarp model and data.
    with wp.ScopedDevice(self.wp_device):
      self._wp_model = mjwarp.put_model(self._mj_model)
      self._wp_model.opt.ls_parallel = cfg.ls_parallel
      self._wp_model.opt.contact_sensor_maxmatch = cfg.contact_sensor_maxmatch

      self._wp_data = mjwarp.put_data(
        self._mj_model,
        self._mj_data,
        nworld=self.num_envs,
        nconmax=self.cfg.nconmax,
        njmax=self.cfg.njmax,
      )

      self._reset_mask_wp = wp.zeros(num_envs, dtype=bool)
      self._reset_mask = TorchArray(self._reset_mask_wp)

    self._model_bridge = WarpBridge(self._wp_model, nworld=self.num_envs)
    self._data_bridge = WarpBridge(self._wp_data)
    self._sensor_context: SensorContext | None = None

    self.use_cuda_graph = self._should_use_cuda_graph()
    self.create_graph()

    self.nan_guard = NanGuard(cfg.nan_guard, self.num_envs, self._mj_model)

  def create_graph(self) -> None:
    """Capture CUDA graphs for step, forward, and reset operations.

    This method must be called whenever GPU arrays in the model or data are
    replaced after initialization. The captured graphs hold pointers to the
    arrays that existed at capture time. If those arrays are replaced, the
    graphs will silently read from the old arrays, ignoring any new values.

    Called automatically by:
    - ``__init__()`` during simulation initialization
    - ``expand_model_fields()`` after replacing model arrays

    On CPU devices or when memory pools are disabled, this is a no-op.
    """
    self.step_graph = None
    self.forward_graph = None
    self.reset_graph = None
    self.sense_graph = None
    if self.use_cuda_graph:
      with wp.ScopedDevice(self.wp_device):
        with wp.ScopedCapture() as capture:
          mjwarp.step(self.wp_model, self.wp_data)
        self.step_graph = capture.graph
        with wp.ScopedCapture() as capture:
          mjwarp.forward(self.wp_model, self.wp_data)
        self.forward_graph = capture.graph
        with wp.ScopedCapture() as capture:
          mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)
        self.reset_graph = capture.graph
        if self._sensor_context is not None:
          with wp.ScopedCapture() as capture:
            self._sense_kernel()
          self.sense_graph = capture.graph

  # Properties.

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mj_data(self) -> mujoco.MjData:
    return self._mj_data

  @property
  def wp_model(self) -> mjwarp.Model:
    return self._wp_model

  @property
  def wp_data(self) -> mjwarp.Data:
    return self._wp_data

  @property
  def data(self) -> "DataBridge":
    return cast("DataBridge", self._data_bridge)

  @property
  def model(self) -> "ModelBridge":
    return cast("ModelBridge", self._model_bridge)

  @property
  def default_model_fields(self) -> dict[str, torch.Tensor]:
    """Default values for expanded model fields, used in domain randomization."""
    return self._default_model_fields

  @property
  def expanded_fields(self) -> set[str]:
    """Names of model fields that have been expanded for per-env DR."""
    return self._expanded_fields

  # Methods.

  def expand_model_fields(self, fields: tuple[str, ...]) -> None:
    """Expand model fields to support per-environment parameters."""
    if not fields:
      return

    invalid_fields = [f for f in fields if not hasattr(self._mj_model, f)]
    if invalid_fields:
      raise ValueError(f"Fields not found in model: {invalid_fields}")

    expand_model_fields(self._wp_model, self.num_envs, list(fields))
    self._expanded_fields.update(fields)
    self._model_bridge.clear_cache()

    if self._sensor_context is not None:
      self._sensor_context.recreate(self._mj_model)

    # Field expansion allocates new arrays and replaces them via setattr. The
    # CUDA graph captured the old memory addresses, so we must recreate it.
    self.create_graph()

  def get_default_field(self, field: str) -> torch.Tensor:
    """Get the default value for a model field, caching for reuse.

    Returns the original values from the C MuJoCo model (mj_model), obtained
    from the final compiled scene spec before any randomization is applied.
    Not to be confused with the GPU Warp model (wp_model) which may have
    randomized values.
    """
    if field not in self._default_model_fields:
      if not hasattr(self._mj_model, field):
        raise ValueError(f"Field '{field}' not found in model")
      model_field = getattr(self.model, field)
      default_value = getattr(self._mj_model, field)
      self._default_model_fields[field] = torch.as_tensor(
        default_value, dtype=model_field.dtype, device=self.device
      ).clone()
    return self._default_model_fields[field]

  def recompute_constants(self, level: RecomputeLevel) -> None:
    """Recompute derived model constants after domain randomization.

    Args:
      level: Which constants to recompute. ``set_const`` is the most
        expensive (covers body_mass changes), ``set_const_0`` covers
        qpos0/body_inertia/dof_armature changes, and ``set_const_fixed``
        is the cheapest (covers body_gravcomp changes).
    """
    fn = getattr(mjwarp, level.name)
    with wp.ScopedDevice(self.wp_device):
      fn(self._wp_model, self._wp_data)

  def forward(self) -> None:
    with wp.ScopedDevice(self.wp_device):
      if self.use_cuda_graph and self.forward_graph is not None:
        wp.capture_launch(self.forward_graph)
      else:
        mjwarp.forward(self.wp_model, self.wp_data)

  def step(self) -> None:
    with wp.ScopedDevice(self.wp_device):
      with self.nan_guard.watch(self.data):
        if self.use_cuda_graph and self.step_graph is not None:
          wp.capture_launch(self.step_graph)
        else:
          mjwarp.step(self.wp_model, self.wp_data)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    with wp.ScopedDevice(self.wp_device):
      if env_ids is None:
        self._reset_mask.fill_(True)
      else:
        self._reset_mask.fill_(False)
        self._reset_mask[env_ids] = True

      if self.use_cuda_graph and self.reset_graph is not None:
        wp.capture_launch(self.reset_graph)
      else:
        mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)

  def set_sensor_context(self, ctx: SensorContext) -> None:
    """Wire a SensorContext for camera/raycast sensing.

    Automatically re-captures CUDA graphs so the sense_graph includes
    the new sensor kernels.
    """
    self._sensor_context = ctx
    self.create_graph()

  def sense(self) -> None:
    """Execute the sense pipeline: prepare -> graph -> finalize.

    Runs BVH refit, camera rendering, and raycasting in a single
    CUDA graph launch. Should be called once per env step, right
    before observation computation.
    """
    if self._sensor_context is None:
      return

    ctx = self._sensor_context
    ctx.prepare()

    with wp.ScopedDevice(self.wp_device):
      if self.use_cuda_graph and self.sense_graph is not None:
        wp.capture_launch(self.sense_graph)
      else:
        self._sense_kernel()

    ctx.finalize()

  # Private methods.

  def _sense_kernel(self) -> None:
    """GPU kernel sequence for sensing (captured in sense_graph)."""
    assert self._sensor_context is not None
    ctx = self._sensor_context
    rc = ctx.render_context

    mjwarp.refit_bvh(self.wp_model, self.wp_data, rc)

    if ctx.has_cameras:
      mjwarp.render(self.wp_model, self.wp_data, rc)
      ctx.unpack_rgb()

    for sensor in ctx.raycast_sensors:
      sensor.raycast_kernel(rc=rc)

  def _should_use_cuda_graph(self) -> bool:
    """Determine if CUDA graphs can be used based on device and driver version."""
    if not self.wp_device.is_cuda:
      return False

    driver_ver = wp.context.runtime.driver_version
    has_mempool = wp.is_mempool_enabled(self.wp_device)

    if driver_ver is None:
      print("[WARNING] CUDA Graphs disabled: driver version unavailable")
      return False

    if has_mempool and driver_ver >= _GRAPH_CAPTURE_MIN_DRIVER:
      return True

    reasons = []
    if not has_mempool:
      reasons.append("mempool disabled")
    if driver_ver < _GRAPH_CAPTURE_MIN_DRIVER:
      reasons.append(f"driver {driver_ver[0]}.{driver_ver[1]} < 12.4")
    print(f"[WARNING] CUDA Graphs disabled: {', '.join(reasons)}")
    return False
