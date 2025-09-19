from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence, cast

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from mjlab.sim.randomization import expand_model_fields
from mjlab.sim.sim_data import WarpBridge

if TYPE_CHECKING:
  ModelBridge = mjwarp.Model
  DataBridge = mjwarp.Data
else:
  ModelBridge = WarpBridge
  DataBridge = WarpBridge


@dataclass(kw_only=True)
class RenderCfg:
  enable_reflections: bool = True
  enable_shadows: bool = True
  camera: str | int | None = -1
  height: int = 240
  width: int = 320


@dataclass
class MujocoCfg:
  # Integrator settings.
  timestep: float = 0.002
  integrator: str = "implicitfast"
  # Friction settings.
  impratio: float = 1.0
  cone: str = "pyramidal"
  # Solver settings.
  jacobian: str = "auto"
  solver: str = "newton"
  iterations: int = 100
  tolerance: float = 1e-8
  ls_iterations: int = 50
  ls_tolerance: float = 0.01
  # Other.
  gravity: tuple[float, float, float] = (0, 0, -9.81)


@dataclass(kw_only=True)
class SimulationCfg:
  nconmax: int | None = None
  njmax: int | None = None
  ls_parallel: bool = True  # Boosts perf quite noticeably.
  mujoco: MujocoCfg = field(default_factory=MujocoCfg)
  render: RenderCfg = field(default_factory=RenderCfg)


class Simulation:
  """MjWarp simulation backend."""

  def __init__(
    self, num_envs: int, cfg: SimulationCfg, model: mujoco.MjModel, device: str
  ):
    self.cfg = cfg
    self.device = device
    self.wp_device = wp.get_device(self.device)
    self.num_envs = num_envs

    self._mj_model = model
    self._mj_data = mujoco.MjData(model)
    mujoco.mj_forward(self._mj_model, self._mj_data)

    with wp.ScopedDevice(self.wp_device):
      self._wp_model = mjwarp.put_model(self._mj_model)
      self._wp_model.opt.ls_parallel = cfg.ls_parallel

      self._wp_data = mjwarp.put_data(
        self._mj_model,
        self._mj_data,
        nworld=self.num_envs,
        nconmax=self.cfg.nconmax,
        njmax=self.cfg.njmax,
      )

    self._model_bridge = WarpBridge(self._wp_model)
    self._data_bridge = WarpBridge(self._wp_data)

    self.use_cuda_graph = self.wp_device.is_cuda and wp.is_mempool_enabled(
      self.wp_device
    )
    self.create_graph()

    self._mj_model.vis.global_.offheight = self.cfg.render.height
    self._mj_model.vis.global_.offwidth = self.cfg.render.width
    if not self.cfg.render.enable_shadows:
      self._mj_model.light_castshadow[:] = False
    if not self.cfg.render.enable_reflections:
      self._mj_model.mat_reflectance[:] = 0.0

    self._camera = self.cfg.render.camera or -1
    self._renderer: mujoco.Renderer | None = None

  def initialize_renderer(self) -> None:
    self._renderer = mujoco.Renderer(
      model=self._mj_model, height=self.cfg.render.height, width=self.cfg.render.height
    )

  def create_graph(self) -> None:
    self.step_graph = None
    self.forward_graph = None
    if self.use_cuda_graph:
      with wp.ScopedCapture() as capture:
        mjwarp.step(self.wp_model, self.wp_data)
      self.step_graph = capture.graph
      with wp.ScopedCapture() as capture:
        mjwarp.forward(self.wp_model, self.wp_data)
      self.forward_graph = capture.graph

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
  def renderer(self) -> mujoco.Renderer:
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize_renderer()' first.")

    return self._renderer

  # Methods.

  def expand_model_fields(self, fields: list[str]) -> None:
    for f in fields:
      if not hasattr(self._mj_model, f):
        raise ValueError(f"Field '{f}' not found in model.")

    expand_model_fields(self._wp_model, self.num_envs, fields)

  def reset(self) -> None:
    # TODO(kevin): Should we be doing anything here?
    pass

  def forward(self) -> None:
    with wp.ScopedDevice(self.wp_device):
      if self.use_cuda_graph and self.forward_graph is not None:
        wp.capture_launch(self.forward_graph)
      else:
        mjwarp.forward(self.wp_model, self.wp_data)

  def step(self) -> None:
    with wp.ScopedDevice(self.wp_device):
      if self.use_cuda_graph and self.step_graph is not None:
        wp.capture_launch(self.step_graph)
      else:
        mjwarp.step(self.wp_model, self.wp_data)

  # TODO(kevin): Consider moving this.
  def set_ctrl(self, ctrl: torch.Tensor, ctrl_ids: Sequence[int] | None = None) -> None:
    indices = slice(None) if ctrl_ids is None else ctrl_ids
    self.data.ctrl[:, indices] = ctrl[:, indices]

  def update_render(self) -> None:
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize_renderer()' first.")

    attrs_to_copy = ["qpos", "qvel", "mocap_pos", "mocap_quat", "xfrc_applied"]
    for attr in attrs_to_copy:
      setattr(self._mj_data, attr, getattr(self.data, attr)[0].cpu().numpy())

    mujoco.mj_forward(self._mj_model, self._mj_data)
    self._renderer.update_scene(data=self._mj_data, camera=self.cfg.render.camera)

  def render(self) -> np.ndarray:
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize_renderer()' first.")

    return self._renderer.render()

  def close(self) -> None:
    if self._renderer is not None:
      self._renderer.close()
      self._renderer = None
