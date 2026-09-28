"""Collect MPC rollouts with a known behavior density, for off-policy PPO.

The collector drives its own copy of the task with a :class:`SamplingMpc`
controller. At every step it executes ``a = u0 + sigma * eps`` around the MPC
action ``u0`` with a fixed execution std ``sigma``, so the behavior policy is
the Gaussian ``mu(a|s) = N(a; u0(s), sigma^2 I)`` and ``log mu(a|s)`` is exact.
Segments are returned in the layout of :meth:`ReplayBuffer.insert`, so MPOPI's
importance correction can use them like past PPO rollouts.

Two variants produce labels for behavior cloning only (no behavior density):
``sigma = 0`` executes ``u0`` itself, and passing ``policy`` to
:meth:`MpcCollector.collect` lets the policy act while the MPC labels the
visited states (DAgger).
"""

import copy
import time
from collections.abc import Callable

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.mpc.config import SamplingMpcCfg
from mjlab.mpc.sampling_mpc import SamplingMpc, preserve_global_rng
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MpcCollector:
  """Rolls out MPC with Gaussian execution noise on ``num_envs`` real envs."""

  def __init__(
    self,
    env_cfg: ManagerBasedRlEnvCfg,
    num_envs: int,
    num_steps: int,
    planner_cfg: SamplingMpcCfg,
    execution_std: float,
    clip_actions: float | None = None,
    device: str = "cpu",
    seed: int = 0,
  ) -> None:
    """
    Args:
      env_cfg: Training config of the task (copied, not modified).
      num_envs: Real envs driven by the MPC controller.
      num_steps: Control steps per collected segment (T).
      planner_cfg: Config of the MPC planner.
      execution_std: Std of the executed Gaussian noise around the MPC action.
      clip_actions: Same action clipping as the PPO env wrapper.
      device: Torch / simulation device.
      seed: Seed of the real envs and of the execution noise.
    """
    if execution_std < 0.0:
      raise ValueError("execution_std must be >= 0.")
    self.num_envs = num_envs
    self.num_steps = num_steps
    self.execution_std = execution_std
    self.device = torch.device(device)
    real_cfg = copy.deepcopy(env_cfg)
    real_cfg.scene.num_envs = num_envs
    real_cfg.seed = seed
    with preserve_global_rng():
      self.env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=real_cfg, device=device), clip_actions=clip_actions
      )
    self.planner = SamplingMpc(env_cfg, num_envs, planner_cfg, device=device)
    self.generator = torch.Generator(device=self.device).manual_seed(seed)
    self.obs = self.env.get_observations()

  def collect(
    self, policy: Callable[[TensorDict], torch.Tensor] | None = None
  ) -> tuple[dict, dict[str, float]]:
    """Roll out one ``[T, N]`` segment.

    Args:
      policy: If given, this policy acts and the MPC action ``u0`` is only
        recorded as a label (DAgger). ``behavior_log_prob`` is then zero and
        not a density, as with ``execution_std == 0``.

    Returns:
      Keyword arguments for :meth:`ReplayBuffer.insert` (without
      ``policy_version``) and collection metrics.
    """
    t_steps, n = self.num_steps, self.num_envs
    obs_list: list[TensorDict] = []
    actions, rewards, dones, time_outs, log_mu, means = [], [], [], [], [], []
    ess_sum, start = 0.0, time.time()
    with torch.no_grad():
      for _ in range(t_steps):
        plan = self.planner.plan(self.env.unwrapped)
        mean = plan.action
        has_density = policy is None and self.execution_std > 0.0
        if policy is not None:
          action = policy(self.obs)
        elif has_density:
          noise = torch.randn(mean.shape, device=self.device, generator=self.generator)
          action = mean + self.execution_std * noise
        else:
          action = mean.clone()
        obs_list.append(self.obs)
        self.obs, reward, done, extras = self.env.step(action)
        reset_ids = done.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
          self.planner.reset(reset_ids)  # The warm-start plan is now invalid.
        actions.append(action)
        means.append(mean)
        if has_density:
          log_mu.append(self._log_prob(action, mean))
        else:
          log_mu.append(torch.zeros(n, 1, device=self.device))
        rewards.append(reward.view(n, 1).float())
        dones.append(done.view(n, 1).bool())
        time_out = extras.get("time_outs", torch.zeros_like(done))
        time_outs.append(time_out.view(n, 1).bool())
        ess_sum += float(plan.ess.mean())
    mean_t = torch.stack(means)
    reward_t, done_t = torch.stack(rewards), torch.stack(dones)
    segment = {
      "observations": torch.stack(obs_list),  # type: ignore[arg-type]
      "actions": torch.stack(actions),
      "rewards": reward_t,
      "dones": done_t,
      "time_outs": torch.stack(time_outs),
      "behavior_log_prob": torch.stack(log_mu),
      "behavior_distribution_params": (
        mean_t,
        torch.full_like(mean_t, self.execution_std),
      ),
      "bootstrap_observations": self.obs.clone(),
    }
    metrics = {
      "reward": float(reward_t.mean()),
      "episodes_done": float(done_t.sum()),
      "planner_ess": ess_sum / t_steps,
      "seconds": time.time() - start,
    }
    return segment, metrics

  def _log_prob(self, action: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """``log N(action; mean, sigma^2 I)`` summed over action dims -> ``[N, 1]``."""
    dist = torch.distributions.Normal(mean, self.execution_std)
    return dist.log_prob(action).sum(dim=-1, keepdim=True)

  def close(self) -> None:
    self.env.close()
    self.planner.close()
