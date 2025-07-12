from mjlab.sim.sim_config import SimulationCfg

import mujoco
import torch
import warp as wp
import mujoco_warp as mjwarp


class Simulation:
  def __init__(self, model: mujoco.MjModel, cfg: SimulationCfg | None = None):
    if cfg is None:
      self.cfg = SimulationCfg()
    else:
      self.cfg = cfg
    self.device = self.cfg.device
    self.num_envs = self.cfg.num_envs

    self.mjmodel = model
    self.mjdata = mujoco.MjData(model)
    mujoco.mj_forward(self.mjmodel, self.mjdata)

    self._wp_model = mjwarp.put_model(self.mjmodel)
    self._wp_data = mjwarp.put_data(
      self.mjmodel,
      self.mjdata,
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
