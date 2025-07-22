import math
from typing import Sequence
import torch
from mjlab.envs.manager_based_env import ManagerBasedEnv
from mjlab.envs.manager_based_rl_env_config import ManagerBasedRlEnvCfg

from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager
from mjlab.managers.command_manager import CommandManager


class ManagerBasedRLEnv(ManagerBasedEnv):
  cfg: ManagerBasedRlEnvCfg

  def __init__(self, cfg: ManagerBasedRlEnvCfg):
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(
      cfg.sim.num_envs, device=cfg.sim.device, dtype=torch.long
    )
    super().__init__(cfg=cfg)
    print("[INFO]: Completed setting up the environment...")

  # Properties.

  @property
  def max_episode_length_s(self) -> float:
    return self.cfg.episode_length_s

  @property
  def max_episode_length(self) -> float:
    return math.ceil(self.max_episode_length_s / self.step_dt)

  # Methods.

  def setup_manager_visualizers(self):
    self.manager_visualizers = {
      "command_manager": self.command_manager,
    }

  def load_managers(self):
    # NOTE: Order is important.
    self.command_manager = CommandManager(self.cfg.commands, self)
    print("[INFO] Command Manager:", self.command_manager)
    super().load_managers()
    self.termination_manager = TerminationManager(self.cfg.terminations, self)
    print("[INFO] Termination Manager:", self.termination_manager)
    self.reward_manager = RewardManager(self.cfg.rewards, self)
    print("[INFO] Reward Manager:", self.reward_manager)
    # curriculum manager
    # setup gym env spaces

  def step(self, action: torch.Tensor):
    self.action_manager.process_action(action.to(self.device))
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()
      self.sim.forward()
      self.sim.data.qacc[:] = 0.0
      print("after reset: ")
      print(self.sim.data.qacc.max())
    self.command_manager.compute(dt=self.step_dt)
    self.obs_buf = self.observation_manager.compute()
    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def _reset_idx(self, env_ids: Sequence[int]) -> None:
    # self.scene.reset(env_ids)
    if "reset" in self.event_manager.available_modes:
      env_step_count = self._sim_step_counter // self.cfg.decimation
      self.event_manager.apply(
        mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
      )
    self.extras["log"] = dict()
    # observation manager.
    info = self.observation_manager.reset(env_ids)
    self.extras["log"].update(info)
    # action manager.
    info = self.action_manager.reset(env_ids)
    self.extras["log"].update(info)
    # rewards manager.
    info = self.reward_manager.reset(env_ids)
    self.extras["log"].update(info)
    # command manager
    info = self.command_manager.reset(env_ids)
    self.extras["log"].update(info)
    # event manager
    info = self.event_manager.reset(env_ids)
    self.extras["log"].update(info)
    # termination manager.
    info = self.termination_manager.reset(env_ids)
    self.extras["log"].update(info)
    # reset the episode length buffer.
    self.episode_length_buf[env_ids] = 0
