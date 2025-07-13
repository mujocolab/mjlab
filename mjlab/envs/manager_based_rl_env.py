import math
from typing import Sequence
import torch
from mjlab.envs.manager_based_env import ManagerBasedEnv
from mjlab.envs.manager_based_rl_env_config import ManagerBasedRlEnvCfg

from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager


class ManagerBasedRLEnv(ManagerBasedEnv):
  cfg: ManagerBasedRlEnvCfg

  def __init__(self, cfg: ManagerBasedRlEnvCfg):
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(
      cfg.sim.num_envs, device=cfg.sim.device, dtype=torch.long
    )
    super().__init__(cfg=cfg)

  # Properties.

  @property
  def max_episode_length_s(self) -> float:
    return self.cfg.episode_length_s

  @property
  def max_episode_length(self) -> float:
    return math.ceil(self.max_episode_length_s / self.step_dt)

  # Methods.

  def load_managers(self):
    # command manager
    # observation / action manager
    super().load_managers()
    # termination manager
    self.termination_manager = TerminationManager(self.cfg.terminations, self)
    # reward manager
    self.reward_manager = RewardManager(self.cfg.rewards, self)
    print("[INFO] Reward Manager:", self.reward_manager)
    # curriculum manager
    # setup gym env spaces

  def step(self, action: torch.Tensor):
    self.action_manager.process_action(action.to(self.device))
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.sim.step()
    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self._reset_idx(reset_env_ids)
      self.sim.forward()
    self.obs_buf = self.observation_manager.compute()
    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def _reset_idx(self, env_ids: Sequence[int]) -> None:
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
    # termination manager.
    info = self.termination_manager.reset(env_ids)
    self.extras["log"].update(info)
    # reset the episode lengh buffer.
    self.episode_length_buf[env_ids] = 0
