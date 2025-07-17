from typing import Sequence
import torch
from mjlab.sim.sim_config import SimulationCfg
from mjlab.sim.sim_data import WarpBridge

import mujoco
import warp as wp
import mujoco_warp as mjwarp


class Simulation:
  """MjWarp simulation backend."""

  def __init__(self, cfg: SimulationCfg):
    self.cfg = cfg
    self.device = self.cfg.device
    self.num_envs = self.cfg.num_envs

    self._mj_model: mujoco.MjModel | None = None
    self._mj_data: mujoco.MjData | None = None
    self._wp_model: mjwarp.Model | None = None
    self._wp_data: mjwarp.Data | None = None

  # Properties.

  @property
  def mj_model(self) -> mujoco.MjModel:
    if self._mj_model is None:
      raise ValueError("Simulation has not been initialized")
    return self._mj_model

  @property
  def mj_data(self) -> mujoco.MjData:
    if self._mj_data is None:
      raise ValueError("Simulation has not been initialized")
    return self._mj_data

  @property
  def wp_model(self) -> mjwarp.Model:
    if self._wp_model is None:
      raise ValueError("Simulation has not been initialized")
    return self._wp_model

  @property
  def wp_data(self) -> mjwarp.Data:
    if self._wp_data is None:
      raise ValueError("Simulation has not been initialized")
    return self._wp_data

  @property
  def data(self) -> WarpBridge:
    return WarpBridge(self.wp_data)

  @property
  def model(self) -> WarpBridge:
    return WarpBridge(self.wp_model)

  # Methods.

  def initialize(self, model: mujoco.MjModel):
    self._mj_model = model
    self._mj_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(self._mj_model, self._mj_data, 0)
    mujoco.mj_forward(self._mj_model, self._mj_data)

    self._wp_model = mjwarp.put_model(self._mj_model)
    self._wp_data = mjwarp.put_data(
      self._mj_model,
      self._mj_data,
      nworld=self.cfg.num_envs,
      nconmax=self.cfg.nconmax,
      njmax=self.cfg.njmax,
    )

    with wp.ScopedCapture() as capture:
      mjwarp.step(self.wp_model, self.wp_data)
    self.graph = capture.graph

  def reset(self):
    print("Called sim.reset()")

  def forward(self):
    mjwarp.forward(self.wp_model, self.wp_data)

  def step(self):
    wp.capture_launch(self.graph)

  def set_ctrl(self, ctrl: torch.Tensor, ctrl_ids: Sequence[int] | None = None) -> None:
    if ctrl_ids is None:
      ctrl_ids = slice(None)
    self.data.ctrl[:, ctrl_ids] = ctrl
