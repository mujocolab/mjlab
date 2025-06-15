from functools import partial
from typing import Generic, Optional, TypeVar

import jax
import jax.numpy as jp
import torch
from rsl_rl.env import VecEnv

from mujoco import mjx
from mjlab._src import mjx_env, mjx_task
from mjlab._src.types import State

ConfigT = TypeVar("ConfigT", bound=mjx_task.TaskConfig)
TaskT = TypeVar("TaskT", bound=mjx_task.MjxTask[ConfigT])

_PMAP_AXIS_NAME = "i"


def _jax_to_torch(tensor: jax.Array, device: str = "cuda:0") -> torch.Tensor:
  import torch.utils.dlpack as tpack

  torch_shards = []
  for buffer in [x.data for x in tensor.addressable_shards]:
    t = tpack.from_dlpack(buffer)
    if t.device != torch.device(device):
      t = t.to(device, non_blocking=True)
    torch_shards.append(t)
  return torch.cat(torch_shards, dim=0)


def _torch_to_jax(tensor: torch.Tensor) -> jax.Array:
  from jax.dlpack import from_dlpack

  return from_dlpack(tensor)


def shard_to_devices(x, num_devices):
  """Split batch over devices for pmap input."""
  return x.reshape((num_devices, -1) + x.shape[1:])


class RslRlVecEnvWrapper(VecEnv, Generic[ConfigT, TaskT]):
  """Wraps a mjx_env.MjxEnv to be compatible with RSL-RL."""

  def __init__(
    self,
    env: mjx_env.MjxEnv[TaskT],
    num_envs: int,
    seed: int,
    clip_actions: Optional[float] = None,
    max_devices_per_host: Optional[int] = None,
    device: str = "cpu",
  ):
    self.env = env
    self.num_envs = num_envs
    self.clip_actions = clip_actions
    self.device = device

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host is not None:
      local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * process_count
    self.local_devices_to_use = local_devices_to_use

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env = jax.random.split(local_key)
    del global_key

    if num_envs % device_count != 0:
      raise ValueError(
        f"Number of environments ({num_envs}) must be divisible by the number of devices ({device_count})"
      )

    randomization_batch_size = num_envs // device_count
    randomization_rng = jax.random.split(key_env, randomization_batch_size)
    v_randomization_fn = partial(env.task.domain_randomize, rng=randomization_rng)
    mjx_model_v, in_axes = v_randomization_fn(env.task.mjx_model)

    def _env_fn(mjx_model: mjx.Model) -> mjx_env.MjxEnv:
      env.unwrapped.task._mjx_model = mjx_model
      return env

    def _reset_fn(rng: jax.Array) -> State:
      def _reset(m, r):
        env = _env_fn(m)
        return env.reset(r)

      return jax.vmap(_reset, in_axes=[in_axes, 0])(mjx_model_v, rng)

    def _step_fn(state: State, action: jax.Array) -> State:
      def _step(m, s, a):
        env = _env_fn(m)
        return env.step(s, a)

      return jax.vmap(_step, in_axes=[in_axes, 0, 0])(mjx_model_v, state, action)

    v_reset = _reset_fn
    v_step = _step_fn
    if local_devices_to_use > 1:
      self._reset_fn = jax.pmap(v_reset, axis_name=_PMAP_AXIS_NAME)
      self._step_fn = jax.pmap(v_step, axis_name=_PMAP_AXIS_NAME)
    else:
      self._reset_fn = jax.jit(jax.vmap(v_reset))
      self._step_fn = jax.jit(jax.vmap(v_step))

    key_envs = jax.random.split(key_env, num_envs // process_count)
    self.key_reset = jp.reshape(
      key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
    )

    self.max_episode_length = env.unwrapped.task.cfg.max_episode_length
    self.num_actions = env.unwrapped.action_size
    obs_size = env.unwrapped.observation_size
    if isinstance(obs_size, dict):
      self.asymmetric_obs = True
      self.num_obs = obs_size["state"]
      self.num_privileged_obs = obs_size["privileged_state"]
    else:
      self.asymmetric_obs = False
      self.num_obs = obs_size
      self.num_privileged_obs = None
    self.cfg = env.unwrapped.task.cfg

    self._sim_step_counter: int = 0
    self.common_step_counter: int = 0
    self.first_state: State
    self.state: State
    self.episode_length_buf: torch.Tensor
    self.reset_terminated: torch.Tensor
    self.reset_time_outs: torch.Tensor
    self.reset_buf: torch.Tensor

    self.reset()

  def get_observations(self):
    extras = {"observations": {}}
    if self.asymmetric_obs:
      obs = _jax_to_torch(self.state.obs["state"], device=self.device)
      critic_obs = _jax_to_torch(self.state.obs["privileged_state"], device=self.device)
      extras["observations"]["critic"] = critic_obs.reshape(self.num_envs, -1)
    else:
      obs = _jax_to_torch(self.state.obs, device=self.device)
    obs = obs.reshape(self.num_envs, -1)
    return obs, extras

  def reset(self):
    self.first_state = self._reset_fn(self.key_reset)
    self.state = self.first_state.replace()

    # Reset buffers.
    self.episode_length_buf = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.reset_terminated = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.reset_time_outs = torch.zeros_like(self.reset_terminated, device=self.device)
    self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    return self.get_observations()

  def _reset_envs(self):
    done_mask = _torch_to_jax(self.reset_buf).reshape(self.local_devices_to_use, -1)

    def _apply_mask(first, current):
      def mask_leaf(f, c):
        mask = done_mask
        while mask.ndim < f.ndim:
          mask = jp.expand_dims(mask, axis=-1)
        return jp.where(mask, f, c)

      return jax.tree.map(mask_leaf, first, current)

    self.state = self.state.replace(
      obs=_apply_mask(self.first_state.obs, self.state.obs),
      data=_apply_mask(self.first_state.data, self.state.data),
    )

  def step(self, actions):
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    actions = actions.reshape(self.local_devices_to_use, -1, self.num_actions)
    self.state = self.state.replace(done=jp.zeros_like(self.state.done))
    self.state = self._step_fn(self.state, _torch_to_jax(actions))

    # def _warn_if_nan(path, x):
    #   if jp.isnan(x).any():
    #     from jax.tree_util import keystr
    #     print("NaN found at", keystr(path))
    # jax.tree.map_with_path(_warn_if_nan, self.state.data)

    self._sim_step_counter += self.env.task.n_substeps  # Total sim steps.
    self.episode_length_buf += 1  # Step in current episode.
    self.common_step_counter += 1  # Total step (common for all envs).

    def _totorch_reshape(x):
      x = _jax_to_torch(x, device=self.device)
      return x.reshape(self.num_envs, *x.shape[2:])

    self.reset_terminated[:] = _totorch_reshape(self.state.done)
    time_out = self.episode_length_buf >= self.max_episode_length
    self.reset_time_outs[:] = time_out
    self.reset_buf = self.reset_terminated | self.reset_time_outs
    dones = (self.reset_terminated | self.reset_time_outs).to(dtype=torch.long)
    rewards = _totorch_reshape(self.state.reward)

    # Reset envs that terminated or timed out.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self._reset_envs()
      self.episode_length_buf[reset_env_ids] = 0

    obs, extras = self.get_observations()
    extras["time_outs"] = self.reset_time_outs
    extras["log"] = {}
    for k, v in self.state.metrics.items():
      extras["log"][k] = _totorch_reshape(v).float().mean().item()

    return obs, rewards, dones, extras

  @property
  def unwrapped(self):
    return self.env.unwrapped
