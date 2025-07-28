from typing import Sequence
import numpy as np
import torch
from mjlab.sim.sim_config import SimulationCfg
from mjlab.sim.sim_data import WarpBridge

import mujoco
import warp as wp
import mujoco_warp as mjwarp


class Simulation:
  """MjWarp simulation backend."""

  def __init__(self, cfg: SimulationCfg, model: mujoco.MjModel):
    self.cfg = cfg
    self.device = self.cfg.device
    self.num_envs = self.cfg.num_envs

    self._mj_model = model
    self._mj_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(self._mj_model, self._mj_data, 0)
    mujoco.mj_forward(self._mj_model, self._mj_data)

    self._wp_model = mjwarp.put_model(self._mj_model)
    self._wp_model.opt.ls_parallel = cfg.ls_parallel

    self._wp_data = mjwarp.put_data(
      self._mj_model,
      self._mj_data,
      nworld=self.cfg.num_envs,
      nconmax=self.cfg.nconmax,
      njmax=self.cfg.njmax,
    )

    with wp.ScopedCapture() as capture:
      mjwarp.step(self.wp_model, self.wp_data)
    self.step_graph = capture.graph

    with wp.ScopedCapture() as capture:
      mjwarp.forward(self.wp_model, self.wp_data)
    self.forward_graph = capture.graph

    self._mj_model.vis.global_.offheight = self.cfg.render.height
    self._mj_model.vis.global_.offwidth = self.cfg.render.width
    if not self.cfg.render.enable_shadows:
      self._mj_model.light_castshadow[:] = False
    if not self.cfg.render.enable_reflections:
      self._mj_model.mat_reflectance[:] = 0.0

    self._camera = self.cfg.render.camera or -1
    self._renderer = mujoco.Renderer(
      model=model, height=self.cfg.render.height, width=self.cfg.render.height
    )

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
  def data(self) -> WarpBridge:
    return WarpBridge(self.wp_data)

  @property
  def model(self) -> WarpBridge:
    return WarpBridge(self.wp_model)

  @property
  def contact(self) -> WarpBridge:
    return WarpBridge(self.wp_data.contact)

  @property
  def renderer(self) -> mujoco.Renderer:
    return self._renderer

  # Methods.

  def reset(self):
    pass

  def forward(self):
    wp.capture_launch(self.forward_graph)

  def step(self):
    wp.capture_launch(self.step_graph)

  def set_ctrl(self, ctrl: torch.Tensor, ctrl_ids: Sequence[int] | None = None) -> None:
    if ctrl_ids is None:
      ctrl_ids = slice(None)
    self.data.ctrl[:, ctrl_ids] = ctrl[:, ctrl_ids]

  def update_render(self) -> None:
    attrs_to_copy = ["qpos", "qvel", "mocap_pos", "mocap_quat", "xfrc_applied"]
    for attr in attrs_to_copy:
      setattr(self._mj_data, attr, getattr(self.data, attr)[0].cpu().numpy())

    mujoco.mj_forward(self._mj_model, self._mj_data)
    self._renderer.update_scene(data=self._mj_data, camera=self.cfg.render.camera)

  def render(self) -> np.ndarray:
    return self._renderer.render()

  def close(self) -> None:
    self._renderer.close()
