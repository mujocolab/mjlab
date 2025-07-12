from typing import Any, Sequence
import torch

from mjlab.envs.manager_based_env_config import ManagerBasedEnvCfg
from mjlab.entities.scene.scene import Scene
from mjlab.sim.sim import Simulation


class ManagerBasedEnv:
  def __init__(self, cfg: ManagerBasedEnvCfg):
    self.cfg = cfg

    if self.cfg.seed is not None:
      self.cfg.seed = self.seed(self.cfg.seed)
    else:
      print("No seed set for the environment.")

    self.scene = Scene(self.cfg.scene)
    self.sim = Simulation(self.scene.compile(), cfg=self.cfg.sim)

    if "cuda" in self.device:
      torch.cuda.set_device(self.device)

    print("[INFO]: Base environment:")
    print(f"\tEnvironment device    : {self.device}")
    print(f"\tEnvironment seed      : {self.cfg.seed}")
    print(f"\tPhysics step-size     : {self.physics_dt}")
    print(f"\tEnvironment step-size : {self.step_dt}")

    self._sim_step_counter = 0
    self.extras = {}

    # Reset sim and step once.

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
    self.action_manager = None
    self.observation_manager = None

  # MDP operations.

  def reset(
    self,
    seed: int | None = None,
    env_ids: Sequence[int] | None = None,
    options: dict[str, Any] | None = None,
  ):
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

    if seed is not None:
      self.seed(seed)

    self._reset_idx(env_ids)

    self.sim.forward()

    # self.obs_buf = self.observation_manager.compute()

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
    pass

  def close(self):
    pass

  # Private methods.

  def _reset_idx(self, env_ids: torch.Tensor) -> None:
    # Reset using env_ids.
    pass


if __name__ == "__main__":
  from mjlab.sim.sim_config import SimulationCfg
  from mjlab.entities.scene.scene import Scene
  from mjlab.entities.scene.scene_config import SceneCfg, LightCfg
  from mjlab.entities.common.config import TextureCfg, OptionCfg
  from mjlab.entities.robots.go1.go1_constants import GO1_ROBOT_CFG
  from mjlab.entities.terrains.flat_terrain import FLAT_TERRAIN_CFG

  SCENE_CFG = SceneCfg(
    terrains=(FLAT_TERRAIN_CFG,),
    robots=(GO1_ROBOT_CFG,),
    lights=(LightCfg(pos=(0, 0, 1.5), type="directional"),),
    skybox=TextureCfg(
      name="skybox",
      type="skybox",
      builtin="gradient",
      rgb1=(0.3, 0.5, 0.7),
      rgb2=(0.1, 0.2, 0.3),
      width=512,
      height=3072,
    ),
  )

  SIM_CFG = SimulationCfg(
    num_envs=8,
    mujoco=OptionCfg(
      timestep=0.004,
      integrator="implicitfast",
    ),
  )

  cfg = ManagerBasedEnvCfg(
    seed=0,
    decimation=1,
    scene=SCENE_CFG,
    sim=SIM_CFG,
  )

  env = ManagerBasedEnv(cfg)
