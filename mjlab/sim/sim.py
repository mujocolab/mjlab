from typing import Sequence
import torch
from mjlab.sim.sim_config import SimulationCfg

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
      raise ValueError
    return self._mj_model

  @property
  def mj_data(self) -> mujoco.MjData:
    if self._mj_data is None:
      raise ValueError
    return self._mj_data

  @property
  def wp_model(self) -> mjwarp.Model:
    if self._wp_model is None:
      raise ValueError
    return self._wp_model

  @property
  def wp_data(self) -> mjwarp.Data:
    if self._wp_data is None:
      raise ValueError
    return self._wp_data

  # Methods.

  def initialize(self, model: mujoco.MjModel):
    self._mj_model = model
    self._mj_data = mujoco.MjData(model)
    mujoco.mj_forward(self._mj_model, self._mj_data)

    self._wp_model = mjwarp.put_model(self._mj_model)
    self._wp_data = mjwarp.put_data(
      self._mj_model,
      self._mj_data,
      nworld=self.cfg.num_envs,
      nconmax=self.cfg.nconmax,
      njmax=self.cfg.njmax,
    )

  def reset(self):
    pass

  def forward(self):
    mjwarp.forward(self._wp_model, self._wp_data)

  def step(self):
    mjwarp.step(self._wp_model, self._wp_data)

  def set_ctrl(self, ctrl: torch.Tensor, ctrl_ids: Sequence[int] | None = None) -> None:
    action = wp.array(ctrl, dtype=float)
    action_dim = ctrl.shape[-1]

    if ctrl_ids is None:
      ctrl_ids = list(range(action_dim))
    ctrl_ids_wp = wp.array(ctrl_ids, dtype=wp.int32)

    wp.launch(
      write_control_subset,
      dim=self.num_envs,
      inputs=[
        action.reshape(-1),
        action_dim,
        ctrl_ids_wp,
      ],
      outputs=[
        self._wp_data.ctrl,
      ],
      device=self.device,
    )


@wp.kernel
def write_control_subset(
  # Inputs.
  action: wp.array(dtype=wp.float32, ndim=1),  # (world, na)
  action_dim: int,
  ctrl_ids: wp.array(dtype=wp.int32, ndim=1),  # (num_actions,)
  # Outputs.
  ctrl: wp.array(dtype=wp.float32, ndim=2),  # (world, nu)
):
  env_id = wp.tid()
  for j in range(action_dim):
    act = action[action_dim * env_id + j]
    ctrl_j = ctrl_ids[j]
    ctrl[env_id, ctrl_j] = act
