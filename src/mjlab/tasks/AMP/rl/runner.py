import os
from typing import cast

import torch
import wandb
import joblib
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

_DEFAULT_DISCRIMINATOR_CFG = DiscriminatorCfg()

def load_motion_data(file_name: str = "", source_fps: int = 30, target_fps: int = 50) -> torch.Tensor:

  data = joblib.load(file_name)
  clip = data[list(data.keys())[0]]  # get first clip
  dof_data = torch.tensor(clip["dof"], dtype=torch.float32)  # (T, 23)
  
  T = dof_data.shape[0]
  t_orig = torch.linspace(0, T / source_fps, T)
  t_new = torch.linspace(0, T / source_fps, int(T * target_fps / source_fps))

  # Interpolate: need (1, 1, T) for grid_sample or just use numpy interp
  # torch has no built-in 1D interp, so we use searchsorted
  indices = torch.searchsorted(t_orig, t_new).clamp(1, T - 1)
  t_low = t_orig[indices - 1]
  t_high = t_orig[indices]
  alpha = ((t_new - t_low) / (t_high - t_low)).unsqueeze(-1)  # (T_new, 1)

  dof_low = dof_data[indices - 1]   # (T_new, 23)
  dof_high = dof_data[indices]      # (T_new, 23)

  dof_resampled = dof_low + alpha * (dof_high - dof_low)  # (T_new, 23)
  return dof_resampled


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
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name
    self.discriminator = Discriminator(discriminator_cfg)

    if discriminator_cfg.motion_file is not None:
      self.motion_data = load_motion_data(discriminator_cfg.motion_file).to(self.device)


  # Overide OnPolicyRunner learn() to add Discriminator | but keep similar structure
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

          discriminator_input = torch.cat((trajectory_buffer[trajectory_cursor-1],trajectory_buffer[trajectory_cursor]),dim=-1)
          disc_out = self.discriminator.forward(discriminator_input).squeeze()  # (num_envs,)
          rewards += self.discriminator.cfg.weight * torch.clamp(
              1.0 - 0.25 * torch.square(disc_out - 1.0),
              min=0.0
          )

          # Move to device
          obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
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

      # Here, update discriminator based on real and motion file samples
      for update_step in range(self.discriminator.cfg.n_updates):

        # Sample motion data slice
        max_start = self.motion_data.shape[0] - self.cfg["num_steps_per_env"] - 1
        start_idx = torch.randint(0, max_start, (1,)).item()
        motion_slice = self.motion_data[start_idx : start_idx + self.cfg["num_steps_per_env"] + 1]

        # Build (obs_n, obs_n+1) pairs from motion data
        real_data = torch.cat([motion_slice[:-1], motion_slice[1:]], dim=-1)

        # Build (obs_n, obs_n+1) pairs from trajectory buffer
        fake_stack = torch.stack(trajectory_buffer, dim=1)   # (num_envs, 25, n_obs) ✅
        fake_data = torch.cat([fake_stack[:, :-1, :], fake_stack[:, 1:, :]], dim=-1)  # (num_envs, 24, 2*n_obs)

        fake_data_flat = fake_data.view(-1, 2 * self.discriminator.cfg.n_obs)
        real_data_flat = real_data.view(-1, 2 * self.discriminator.cfg.n_obs)

        self.discriminator.train_oneshot(real_data_flat, fake_data_flat)

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
      output_names=[
        "actions",
      ],
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
