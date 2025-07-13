from typing import Any, Sequence
import numpy as np
import torch

from mjlab.envs.manager_based_env_config import ManagerBasedEnvCfg
from mjlab.entities.scene.scene import Scene
from mjlab.entities.common.editors import OptionEditor
from mjlab.sim.sim import Simulation
from mjlab.utils import random

from mjlab.managers.observation_manager import ObservationManager
from mjlab.managers.action_manager import ActionManager


class ManagerBasedEnv:
  def __init__(self, cfg: ManagerBasedEnvCfg):
    self.cfg = cfg

    if self.cfg.seed is not None:
      self.cfg.seed = self.seed(self.cfg.seed)
    else:
      print("No seed set for the environment.")

    self.sim = Simulation(self.cfg.sim)

    if "cuda" in self.device:
      torch.cuda.set_device(self.device)

    print("[INFO]: Base environment:")
    print(f"\tEnvironment device    : {self.device}")
    print(f"\tEnvironment seed      : {self.cfg.seed}")
    print(f"\tPhysics step-size     : {self.physics_dt}")
    print(f"\tEnvironment step-size : {self.step_dt}")

    # Generate the scene.
    self.scene = Scene(self.cfg.scene)
    OptionEditor(cfg=self.cfg.sim.mujoco).edit_spec(self.scene.spec)

    # TODO Event manager.

    self._sim_step_counter = 0
    self.extras = {}

    # Reset sim and step once.
    self.sim.initialize(self.scene.model)

    self.load_managers()

    self.obs_buf = {}

  @property
  def num_envs(self) -> int:
    return self.sim.num_envs

  @property
  def physics_dt(self) -> float:
    return self.cfg.sim.mujoco.timestep

  @property
  def step_dt(self) -> float:
    return self.cfg.sim.mujoco.timestep * self.cfg.decimation

  @property
  def device(self):
    return self.sim.device

  # Setup.

  def load_managers(self):
    self.observation_manager = ObservationManager(self.cfg.observations, self)
    print("[INFO] Observation Manager:", self.observation_manager)

    self.action_manager = ActionManager(self.cfg.actions, self)
    print("[INFO] Action Manager:", self.action_manager)

  # MDP operations.

  def reset(
    self,
    seed: int | None = None,
    env_ids: Sequence[int] | None = None,
    options: dict[str, Any] | None = None,
  ):
    del options  # Unused.
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
    if seed is not None:
      self.seed(seed)
    self._reset_idx(env_ids)
    self.sim.forward()
    self.obs_buf = self.observation_manager.compute()
    return self.obs_buf, self.extras

  def step(
    self,
    action: torch.Tensor,
  ) -> tuple[Any, dict]:
    self.action_manager.process_action(action.to(self.device))
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.sim.step()
    self.obs_buf = self.observation_manager.compute()
    return self.obs_buf, self.extras

  @staticmethod
  def seed(seed: int = -1) -> int:
    if seed == -1:
      seed = np.random.randint(0, 10_000)
    print(f"Setting seed: {seed}")
    random.seed_rng(seed)
    return seed

  def close(self):
    pass

  # Private methods.

  def _reset_idx(self, env_ids: torch.Tensor) -> None:
    # Observation manager.
    self.extras["log"] = dict()
    info = self.observation_manager.reset(env_ids)
    self.extras["log"].update(info)
    # Action manager.
    info = self.action_manager.reset(env_ids)
    self.extras["log"].update(info)
