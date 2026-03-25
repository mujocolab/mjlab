import os
from typing import cast

import torch
import wandb
import joblib
import pandas as pd
from rsl_rl.env.vec_env import VecEnv
from torch import nn
import time

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.AMP.rl.networks import Discriminator, DiscriminatorCfg
from mjlab.utils.spaces import Dict as DictSpace

##
#
# Implementation based on : https://arxiv.org/pdf/2104.02180
# "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control"
# By : Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, Angjoo Kanazawa
#
##

_DEFAULT_DISCRIMINATOR_CFG = DiscriminatorCfg()

def load_motion_data(file_name: str = "", source_fps: int = 0, target_fps: int = 50) -> torch.Tensor:

  ext = os.path.splitext(file_name)[-1].lower()

  if ext == ".csv":
    _data = torch.tensor(pd.read_csv(file_name, header=None).values, dtype=torch.float32)

  elif ext in (".pkl", ".joblib"):
    data = joblib.load(file_name)
    clip = data[list(data.keys())[0]]
    _data = torch.tensor(clip["dof"], dtype=torch.float32)

  else:
    raise ValueError(f"Unsupported file format: {ext}")

  if source_fps == 0:
    return _data

  T = _data.shape[0]
  t_orig = torch.linspace(0, T / source_fps, T)
  t_new = torch.linspace(0, T / source_fps, int(T * target_fps / source_fps))

  indices = torch.searchsorted(t_orig, t_new).clamp(1, T - 1)
  t_low = t_orig[indices - 1]
  t_high = t_orig[indices]
  alpha = ((t_new - t_low) / (t_high - t_low)).unsqueeze(-1)

  _data_low = _data[indices - 1]
  _data_high = _data[indices]

  _data_resampled = _data_low + alpha * (_data_high - _data_low)
  return _data_resampled


class _OnnxAmpModel(nn.Module):
  """ONNX-exportable model that wraps the policy and bundles motion reference data."""

  def __init__(self, actor):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)

  def forward(self, x):
    return (
      self.policy(x),
    )


class AmpOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
    discriminator_cfg: DiscriminatorCfg = _DEFAULT_DISCRIMINATOR_CFG,
    resample: int = 0,
    replay_buffer_size: int = 100_000,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

    discriminator_cfg.n_obs = cast(DictSpace, self.env.observation_space).spaces["discriminator"].shape[1]
    self.discriminator = Discriminator(discriminator_cfg)

    if discriminator_cfg.motion_file is not None:
      self.motion_data = load_motion_data(discriminator_cfg.motion_file, source_fps=resample, target_fps=int(1.0 / self.env.unwrapped.step_dt)).to(self.device)
      self.motion_mean = self.motion_data.mean(dim=0, keepdim=True)
      self.motion_std = self.motion_data.std(dim=0, keepdim=True) + 1e-6

    # Replay buffer: stores (obs_t, obs_t+1) pairs from past rollouts
    self._replay_buffer = torch.zeros(replay_buffer_size, 2 * discriminator_cfg.n_obs, device=self.device)
    self._replay_ptr = 0
    self._replay_size = 0
    self._replay_capacity = replay_buffer_size

  def _push_to_replay_buffer(self, fake_data_flat: torch.Tensor) -> None:
    """Insert a batch of transition pairs into the circular replay buffer."""
    n = fake_data_flat.shape[0]
    end = self._replay_ptr + n

    if end <= self._replay_capacity:
      self._replay_buffer[self._replay_ptr:end] = fake_data_flat
    else:
      # Wrap around
      first = self._replay_capacity - self._replay_ptr
      self._replay_buffer[self._replay_ptr:] = fake_data_flat[:first]
      self._replay_buffer[:n - first] = fake_data_flat[first:]

    self._replay_ptr = end % self._replay_capacity
    self._replay_size = min(self._replay_size + n, self._replay_capacity)

  def _sample_fake_data(self, n: int) -> torch.Tensor:
    """Sample n transition pairs from the replay buffer."""
    indices = torch.randint(0, self._replay_size, (n,), device=self.device)
    return self._replay_buffer[indices]

  # Override OnPolicyRunner learn() to add Discriminator | but keep similar structure
  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    # Randomize initial episode lengths (for exploration)
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    # Start learning
    obs = self.env.get_observations().to(self.device)

    # Retrieve observations used by discriminator (typically less than the actor or critic uses)
    discriminator_obs = obs["discriminator"]

    self.alg.train_mode()  # switch to train mode (for dropout for example)
    self.discriminator.eval()

    # Ensure all parameters are in-synced
    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    # Initialize the logging writer
    self.logger.init_logging_writer()

    # Start training
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      # Rollout

      # Trajectory buffer for discriminator updates
      trajectory_buffer = [discriminator_obs]
      trajectory_cursor = 0

      # Dones buffer to track invalid transitions
      done_buffer = []

      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          # Sample actions
          actions = self.alg.act(obs)
          # Step the environment
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

          # Save observations needed for discriminator
          discriminator_obs = obs["discriminator"]
          trajectory_buffer.append(discriminator_obs)
          trajectory_cursor += 1

          # Move to device
          obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

          done_buffer.append(dones)
          valid = (dones == 0).float()

          obs_t = (trajectory_buffer[trajectory_cursor-1] - self.motion_mean) / self.motion_std
          obs_tp1 = (trajectory_buffer[trajectory_cursor] - self.motion_mean) / self.motion_std
          discriminator_input = torch.cat((obs_t, obs_tp1), dim=-1)
          discriminator_input_noise = 0.05 * torch.randn_like(discriminator_input)
          discriminator_input = discriminator_input + discriminator_input_noise
          disc_out = self.discriminator.forward(discriminator_input).squeeze()

          self.env.unwrapped.extras["log"]["Metrics/Discriminator_ouput"] = torch.mean(disc_out)
          amp_reward = valid * self.discriminator.cfg.weight * torch.clamp(
            1.0 - 0.25 * torch.square(disc_out - 1.0),
            min=0.0
          )

          self.env.unwrapped.extras["log"]["Metrics/Discriminator_reward"] = torch.mean(amp_reward)

          rewards += amp_reward

          # Process the step
          self.alg.process_env_step(obs, rewards, dones, extras)
          # Extract intrinsic rewards if RND is used (only for logging)
          intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          # Book keeping
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

        stop = time.time()
        collect_time = stop - start
        start = stop

        # Compute returns
        self.alg.compute_returns(obs)

      # Build fake transition pairs from current rollout and push to replay buffer
      fake_stack = torch.stack(trajectory_buffer, dim=1)                            # (num_envs, T+1, n_obs)
      fake_data = torch.cat([fake_stack[:, :-1, :], fake_stack[:, 1:, :]], dim=-1)  # (num_envs, T, 2*n_obs)

      done_stack = torch.stack(done_buffer, dim=1)
      valid_mask = (done_stack == 0)

      fake_data_flat = fake_data.view(-1, 2 * self.discriminator.cfg.n_obs)
      valid_flat = valid_mask.view(-1)
      fake_data_flat = fake_data_flat[valid_flat]

      fake_t = fake_data_flat[:, :self.discriminator.cfg.n_obs]
      fake_tp1 = fake_data_flat[:, self.discriminator.cfg.n_obs:]

      # Normalize
      fake_t = (fake_t - self.motion_mean) / self.motion_std
      fake_tp1 = (fake_tp1 - self.motion_mean) / self.motion_std

      fake_data_flat = torch.cat([fake_t, fake_tp1], dim=-1)

      self._push_to_replay_buffer(fake_data_flat.detach())

      if it >= 50 and it % 4 == 0:
        # Update discriminator using replay buffer samples
        for _ in range(self.discriminator.cfg.n_updates):
          # Only sample from replay buffer once it has enough data
          n_fake = min(fake_data_flat.shape[0], self._replay_size, 512)
          sampled_fake = self._sample_fake_data(n_fake)

          # Sample real data to match fake batch size
          indices = torch.randint(0, self.motion_data.shape[0] - 1, (n_fake,), device=self.device)
          
          real_t = self.motion_data[indices]
          real_tp1 = self.motion_data[indices + 1]

          # Normalize each frame BEFORE concatenation
          real_t = (real_t - self.motion_mean) / self.motion_std
          real_tp1 = (real_tp1 - self.motion_mean) / self.motion_std

          real_data_flat = torch.cat([real_t, real_tp1], dim=-1)

          self.discriminator.train_oneshot(real_data_flat, sampled_fake)

      # Update policy
      loss_dict = self.alg.update()

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      # Log information
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.alg.rnd is not None else None,
      )

      # Save model
      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

    # Save the final model after training and stop the logging writer
    if self.logger.writer is not None:
      self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
      self.logger.stop_logging_writer()

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    model = _OnnxAmpModel(self.alg.get_policy())
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    torch.onnx.export(
      model,
      (obs,),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_axes={},
      dynamo=False,
    )

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = policy_path.split("/")[-2] + ".onnx"
    try:
      self.export_policy_to_onnx(policy_path, filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)

      attach_metadata_to_onnx(os.path.join(policy_path, filename), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)  # type: ignore
          self.registry_name = None
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")