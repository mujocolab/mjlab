from functools import partial
from typing import Generic, Optional, TypeVar

import jax
import jax.numpy as jp
import torch
from rsl_rl.env import VecEnv

from mujoco import mjx
from mjlab.core import mjx_env, mjx_task
from mjlab.core.types import State

ConfigT = TypeVar("ConfigT", bound=mjx_task.TaskConfig)
TaskT = TypeVar("TaskT", bound=mjx_task.MjxTask[ConfigT])


def _jax_to_torch(tensor: jax.Array, device) -> torch.Tensor:
  import torch.utils.dlpack as tpack

  return tpack.from_dlpack(tensor).to(device)


def _torch_to_jax(tensor: torch.Tensor) -> jax.Array:
  from jax.dlpack import from_dlpack

  return from_dlpack(tensor)


class RslRlVecEnvWrapper(VecEnv, Generic[ConfigT, TaskT]):
  """Wraps a mjx_env.MjxEnv to be compatible with RSL-RL."""

  def __init__(
    self,
    env: mjx_env.MjxEnv[TaskT],
    num_envs: int,
    seed: int,
    clip_actions: Optional[float] = None,
    device: str = "cuda:0",
    resample_on_reset: bool = False,
  ):
    self.env = env
    self.num_envs = num_envs
    self.clip_actions = clip_actions
    self.device = device
    self.resample_on_reset = resample_on_reset

    key = jax.random.PRNGKey(seed)
    _, key_env = jax.random.split(key)
    self.key_env = key_env

    randomization_rng = jax.random.split(key_env, num_envs)
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

    self._reset_fn = jax.jit(_reset_fn)
    self._step_fn = jax.jit(_step_fn)

    self.key_reset = self._generate_key_reset(self._next_key())

    self.max_episode_length = env.unwrapped.task.cfg.max_episode_length
    self.num_actions = env.unwrapped.action_size
    obs_size = env.unwrapped.observation_size
    self.asymmetric_obs = isinstance(obs_size, dict)
    self.num_obs = obs_size["state"] if self.asymmetric_obs else obs_size
    self.num_privileged_obs = (
      obs_size.get("privileged_state") if self.asymmetric_obs else None
    )
    self.cfg = env.unwrapped.task.cfg

    self._sim_step_counter = 0
    self.common_step_counter = 0
    self.state: State
    self.reset()

  def _next_key(self) -> jax.Array:
    self.key_env, subkey = jax.random.split(self.key_env)
    return subkey

  def _generate_key_reset(self, key: jax.Array) -> jax.Array:
    return jax.random.split(key, self.num_envs)

  def get_observations(self):
    extras = {"observations": {}}
    if self.asymmetric_obs:
      obs = _jax_to_torch(self.state.obs["state"], device=self.device)
      critic_obs = _jax_to_torch(self.state.obs["privileged_state"], device=self.device)
      extras["observations"]["critic"] = critic_obs.reshape(self.num_envs, -1)
    else:
      obs = _jax_to_torch(self.state.obs, device=self.device)
    return obs.reshape(self.num_envs, -1), extras

  def reset(self):
    self.first_state = self._reset_fn(self.key_reset)
    self.state = self.first_state.replace()
    self.episode_length_buf = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.reset_terminated = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.reset_time_outs = torch.zeros_like(self.reset_terminated)
    self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return self.get_observations()

  def _reset_envs(self):
    done_mask = _torch_to_jax(self.reset_buf)

    if self.resample_on_reset:
      self.key_reset = self._generate_key_reset(self._next_key())
      first_state = self._reset_fn(self.key_reset)
    else:
      first_state = self.first_state

    def _apply_mask(first, current):
      def mask_leaf(f, c):
        mask = done_mask
        while mask.ndim < f.ndim:
          mask = jp.expand_dims(mask, axis=-1)
        return jp.where(mask, f, c)

      return jax.tree.map(mask_leaf, first, current)

    self.state = self.state.replace(
      obs=_apply_mask(first_state.obs, self.state.obs),
      data=_apply_mask(first_state.data, self.state.data),
    )

  def step(self, actions):
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    actions = actions.reshape(self.num_envs, self.num_actions)
    self.state = self.state.replace(done=jp.zeros_like(self.state.done))
    self.state = self._step_fn(self.state, _torch_to_jax(actions))

    self._sim_step_counter += self.env.task.n_substeps
    self.episode_length_buf += 1
    self.common_step_counter += 1

    def _totorch_reshape(x):
      x = _jax_to_torch(x, device=self.device)
      return x.reshape(self.num_envs, *x.shape[1:])

    self.reset_terminated[:] = _totorch_reshape(self.state.done)
    self.reset_time_outs[:] = self.episode_length_buf >= self.max_episode_length
    self.reset_buf = self.reset_terminated | self.reset_time_outs
    dones = self.reset_buf.to(dtype=torch.long)
    rewards = _totorch_reshape(self.state.reward)

    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self._reset_envs()
      self.episode_length_buf[reset_env_ids] = 0

    obs, extras = self.get_observations()
    extras["time_outs"] = self.reset_time_outs
    extras["log"] = {
      k: _totorch_reshape(v).float().mean().item()
      for k, v in self.state.metrics.items()
    }

    return obs, rewards, dones, extras

  @property
  def unwrapped(self):
    return self.env.unwrapped
