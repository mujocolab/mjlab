from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from mjlab.envs import types
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.action_manager import ActionManager
from mjlab.managers.event_manager import EventManager
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
)
from mjlab.managers.observation_manager import ObservationManager
from mjlab.scene import Scene
from mjlab.scene.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.utils import random as random_utils
from mjlab.viewer.viewer_config import ViewerConfig


@dataclass(kw_only=True)
class ManagerBasedEnvCfg:
  """Configuration for a manager-based environment."""

  decimation: int
  scene: SceneCfg
  observations: dict[str, ObservationGroupCfg]
  actions: dict[str, ActionTermCfg]
  events: dict[str, EventTermCfg] = field(
    default_factory=lambda: {
      "reset_scene_to_default": EventTermCfg(
        func=reset_scene_to_default,
        mode="reset",
      )
    }
  )
  seed: int | None = None
  sim: SimulationCfg = field(default_factory=SimulationCfg)
  viewer: ViewerConfig = field(default_factory=ViewerConfig)


class ManagerBasedEnv:
  def __init__(self, cfg: ManagerBasedEnvCfg, device: str) -> None:
    self.cfg = cfg
    if self.cfg.seed is not None:
      self.cfg.seed = self.seed(self.cfg.seed)
    else:
      print("No seed set for the environment.")
    self._sim_step_counter = 0
    self.extras = {}
    self.obs_buf = {}

    self.scene = Scene(self.cfg.scene, device=device)
    self.cfg.sim.mujoco.edit_spec(self.scene.spec)
    print("[INFO]: Scene manager: ", self.scene)

    self.sim = Simulation(
      num_envs=self.scene.num_envs,
      cfg=self.cfg.sim,
      model=self.scene.compile(),
      device=device,
    )

    if "cuda" in self.device:
      torch.cuda.set_device(self.device)

    self.scene.initialize(
      mj_model=self.sim.mj_model,
      model=self.sim.model,
      data=self.sim.data,
    )

    print("[INFO]: Base environment:")
    print(f"\tEnvironment device    : {self.device}")
    print(f"\tEnvironment seed      : {self.cfg.seed}")
    print(f"\tPhysics step-size     : {self.physics_dt}")
    print(f"\tEnvironment step-size : {self.step_dt}")

    self.load_managers()
    self.setup_manager_visualizers()

  @property
  def num_envs(self) -> int:
    return self.scene.num_envs

  @property
  def physics_dt(self) -> float:
    return self.cfg.sim.mujoco.timestep

  @property
  def step_dt(self) -> float:
    return self.cfg.sim.mujoco.timestep * self.cfg.decimation

  @property
  def device(self) -> str:
    return self.sim.device

  # Setup.

  def setup_manager_visualizers(self) -> None:
    self.manager_visualizers = {}

  def load_managers(self) -> None:
    self.event_manager = EventManager(self.cfg.events, self)
    print("[INFO] Event manager: ", self.event_manager)

    self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)

    self.action_manager = ActionManager(self.cfg.actions, self)
    print("[INFO] Action Manager:", self.action_manager)
    self.observation_manager = ObservationManager(self.cfg.observations, self)
    print("[INFO] Observation Manager:", self.observation_manager)

    if (
      self.__class__ == ManagerBasedEnv
      and "startup" in self.event_manager.available_modes
    ):
      self.event_manager.apply(mode="startup")
      self.sim.create_graph()

  # MDP operations.

  def reset(
    self,
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[types.VecEnvObs, dict]:
    del options  # Unused.
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
    if seed is not None:
      self.seed(seed)
    self._reset_idx(env_ids)
    self.scene.write_data_to_sim()
    self.sim.forward()
    self.obs_buf = self.observation_manager.compute()
    return self.obs_buf, self.extras

  def step(
    self,
    action: torch.Tensor,
  ) -> tuple[types.VecEnvObs, dict]:
    self.action_manager.process_action(action.to(self.device))
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)
    self.obs_buf = self.observation_manager.compute()
    return self.obs_buf, self.extras

  def close(self) -> None:
    self.sim.close()

  @staticmethod
  def seed(seed: int = -1) -> int:
    if seed == -1:
      seed = np.random.randint(0, 10_000)
    print(f"Setting seed: {seed}")
    random_utils.seed_rng(seed)
    return seed

  def update_visualizers(self, scn) -> None:
    for mod in self.manager_visualizers.values():
      mod.debug_vis(scn)

  # Private methods.

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    self.scene.reset(env_ids)
    if "reset" in self.event_manager.available_modes:
      env_step_count = self._sim_step_counter // self.cfg.decimation
      self.event_manager.apply(
        mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
      )
    self.extras["log"] = dict()
    # Observation manager.
    info = self.observation_manager.reset(env_ids)
    self.extras["log"].update(info)
    # Action manager.
    info = self.action_manager.reset(env_ids)
    self.extras["log"].update(info)
    # Event manager.
    info = self.event_manager.reset(env_ids)
    self.extras["log"].update(info)
