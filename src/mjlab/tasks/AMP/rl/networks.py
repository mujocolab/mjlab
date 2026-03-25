"""AMP neural network architectures.
"""

from __future__ import annotations

import torch

from torch import nn


class DiscriminatorCfg :
  n_obs : int = 1

  hidden_dim : int = 512

  use_layer_norm : bool = True

  device : torch.device | str | None = None

  n_updates : int = 2

  motion_file : str | None = None

  weight : float = 1.0

class Discriminator(nn.Module):
  """ReLU simple model"""

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

    self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
    

  def forward(
    self, obs: torch.Tensor
  ) -> torch.Tensor:
    out = self.net(obs)
    return self.prediction(out)
  
  def discriminator_objective(self, real_preds : torch.Tensor, fake_preds : torch.Tensor):
    loss_real = torch.mean((real_preds - 1) ** 2)   # real → 1
    loss_fake = torch.mean((fake_preds + 1) ** 2)   # fake → -1
    return loss_real + loss_fake
  
  def gradient_penalty(self, real_data: torch.Tensor, real_preds: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(
      outputs=real_preds,
      inputs=real_data,
      grad_outputs=torch.ones_like(real_preds),
      create_graph=True,
      retain_graph=True,
    )[0]
    return torch.mean(torch.sum(grads**2, dim=-1))

  def train_oneshot(self, real_data, fake_data) -> None:
    self.train()

    real_data = real_data.requires_grad_(True)
    fake_data = fake_data.requires_grad_(True)

    noise_std = 0.05
    _real_data = real_data + noise_std * torch.randn_like(real_data)
    _fake_data = fake_data + noise_std * torch.randn_like(fake_data)

    real_preds = self.forward(_real_data)
    fake_preds = self.forward(_fake_data)

    loss = self.discriminator_objective(real_preds, fake_preds)
    loss += 0.05 * self.gradient_penalty(real_data, real_preds)
    loss += 0.05 * self.gradient_penalty(fake_data, fake_preds)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.eval()