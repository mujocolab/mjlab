"""AMP neural network architectures.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch import nn


class DiscriminatorCfg :
  n_obs : int

  hidden_dim : int = 512

  use_layer_norm : bool = True

  device : torch.device | str | None = None

  n_updates : int = 10

  motion_file : str = "/home/manuelactis/walk1_subject1.pkl"

  weight : float = 2.0

class Discriminator(nn.Module):
  """Sigmoid simple model"""

  def __init__(
    self,
    cfg : DiscriminatorCfg
  ):
    super().__init__()
    self.n_out = 1
    self.cfg = cfg

    def _ln(dim: int) -> nn.Module:
      return nn.LayerNorm(dim, device=cfg.device) if cfg.use_layer_norm else nn.Identity()

    self.net = nn.Sequential(
      nn.Linear(2*cfg.n_obs, cfg.hidden_dim, device=cfg.device),
      _ln(cfg.hidden_dim),
      nn.ReLU(),
      nn.Linear(cfg.hidden_dim, cfg.hidden_dim, device=cfg.device),
      _ln(cfg.hidden_dim),
      nn.ReLU(),
    )
    self.prediction = nn.Linear(cfg.hidden_dim, self.n_out, device=cfg.device)

    self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

  def forward(
    self, obs: torch.Tensor
  ) -> torch.Tensor:
    out = self.net(obs)
    return self.prediction(out)
  
  def discriminator_objective(self, real_preds : torch.Tensor, fake_preds : torch.Tensor):
    loss_real = torch.mean((real_preds - 1) ** 2)   # real → 1
    loss_fake = torch.mean((fake_preds + 1) ** 2)   # fake → -1
    return loss_real + loss_fake
  
  def train_oneshot(
    self,
    real_data,
    fake_data
  ) -> None:
    
    self.train()

    # Forward pass
    real_preds = self.forward(real_data)
    fake_preds = self.forward(fake_data)

    loss = self.discriminator_objective(real_preds, fake_preds)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()