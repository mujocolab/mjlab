import math
import torch
from mjlab.envs.manager_based_env import ManagerBasedEnv
from mjlab.envs.manager_based_rl_env_config import ManagerBasedRlEnvCfg

from mjlab.managers.reward_manager import RewardManager


class ManagerBasedRLEnv(ManagerBasedEnv):
  cfg: ManagerBasedRlEnvCfg

  def __init__(self, cfg: ManagerBasedRlEnvCfg):
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
    return math.ceil(self.max_episode_length / self.step_dt)

  # Methods.

  def load_managers(self):
    # command manager
    # observation / action manager
    super().load_managers()
    # termination manager
    # reward manager
    self.reward_manager = RewardManager(self.cfg.rewards, self)
    print("[INFO] Reward Manager:", self.reward_manager)
    # curriculum manager
    # setup gym env spaces

  def step(self, action: torch.Tensor):
    pass

  def _reset_idx(self, env_ids):
    pass
