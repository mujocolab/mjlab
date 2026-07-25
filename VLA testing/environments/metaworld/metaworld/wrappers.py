from __future__ import annotations

import base64

import gymnasium as gym
import numpy as np
from gymnasium import Env
from numpy.typing import NDArray

from metaworld.sawyer_xyz_env import SawyerXYZEnv
from metaworld.types import Task


class OneHotWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: Env, task_idx: int, num_tasks: int):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        env_lb = env.observation_space.low
        env_ub = env.observation_space.high
        one_hot_ub = np.ones(num_tasks)
        one_hot_lb = np.zeros(num_tasks)

        self.one_hot = np.zeros(num_tasks)
        self.one_hot[task_idx] = 1.0

        self._observation_space = gym.spaces.Box(
            np.concatenate([env_lb, one_hot_lb]), np.concatenate([env_ub, one_hot_ub])
        )

    def observation(self, obs: NDArray) -> NDArray:
        return np.concatenate([obs, self.one_hot])


def _serialize_task(task: Task) -> dict:
    return {
        "env_name": task.env_name,
        "data": base64.b64encode(task.data).decode("ascii"),
    }


def _deserialize_task(task_dict: dict[str, str]) -> Task:
    assert "env_name" in task_dict and "data" in task_dict

    return Task(
        env_name=task_dict["env_name"], data=base64.b64decode(task_dict["data"])
    )


class RNNBasedMetaRLWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically include prev_action / reward / done info in the observation.
    For use with RNN-based meta-RL algorithms."""

    def __init__(self, env: Env, normalize_reward: bool = True):
        super().__init__(env)
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        assert isinstance(self.env.action_space, gym.spaces.Box)
        obs_flat_dim = int(np.prod(self.env.observation_space.shape))
        action_flat_dim = int(np.prod(self.env.action_space.shape))
        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_flat_dim + action_flat_dim + 1 + 1,)
        )
        self._normalize_reward = normalize_reward

    def step(self, action):
        next_obs, reward, terminate, truncate, info = self.env.step(action)
        if self._normalize_reward:
            obs_reward = float(reward) / 10.0
        else:
            obs_reward = float(reward)

        recurrent_obs = np.concatenate(
            [
                next_obs,
                action,
                [obs_reward],
                [float(np.logical_or(terminate, truncate))],
            ]
        )
        return recurrent_obs, reward, terminate, truncate, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        assert isinstance(self.env.action_space, gym.spaces.Box)
        obs, info = self.env.reset(seed=seed, options=options)
        recurrent_obs = np.concatenate(
            [obs, np.zeros(self.env.action_space.shape), [0.0], [0.0]]
        )
        return recurrent_obs, info


class RandomTaskSelectWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically set / reset the environment to a random
    task."""

    tasks: list[Task]
    sample_tasks_on_reset: bool = True

    def _set_random_task(self):
        task_idx = self.np_random.choice(len(self.tasks))
        self.unwrapped.set_task(self.tasks[task_idx])

    def __init__(
        self,
        env: Env,
        tasks: list[Task],
        sample_tasks_on_reset: bool = True,
    ):
        super().__init__(env)
        self.unwrapped: SawyerXYZEnv
        self.tasks = tasks
        self.sample_tasks_on_reset = sample_tasks_on_reset

    def toggle_sample_tasks_on_reset(self, on: bool):
        self.sample_tasks_on_reset = on

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.sample_tasks_on_reset:
            self._set_random_task()
        return self.env.reset(seed=seed, options=options)

    def sample_tasks(self, *, seed: int | None = None, options: dict | None = None):
        self._set_random_task()
        return self.env.reset(seed=seed, options=options)

    def get_checkpoint(self) -> dict:
        return {
            "tasks": [_serialize_task(task) for task in self.tasks],
            "rng_state": self.np_random.bit_generator.state,
            "sample_tasks_on_reset": self.sample_tasks_on_reset,
            "env_rng_state": get_env_rng_checkpoint(self.unwrapped),
        }

    def load_checkpoint(self, ckpt: dict):
        assert "tasks" in ckpt
        assert "rng_state" in ckpt
        assert "sample_tasks_on_reset" in ckpt
        assert "env_rng_state" in ckpt

        self.tasks = [_deserialize_task(task) for task in ckpt["tasks"]]
        self.np_random.__setstate__(ckpt["rng_state"])
        self.sample_tasks_on_reset = ckpt["sample_tasks_on_reset"]
        set_env_rng(self.unwrapped, ckpt["env_rng_state"])


class PseudoRandomTaskSelectWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically reset the environment to a *pseudo*random task when explicitly called.

    Pseudorandom implies no collisions therefore the next task in the list will be used cyclically.
    However, the tasks will be shuffled every time the last task of the previous shuffle is reached.

    Doesn't sample new tasks on reset by default.
    """

    tasks: list[Task]
    current_task_idx: int
    sample_tasks_on_reset: bool = False

    def _set_pseudo_random_task(self):
        self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
        if self.current_task_idx == 0:
            self.np_random.shuffle(self.tasks)  # pyright: ignore [reportArgumentType]
        self.unwrapped.set_task(self.tasks[self.current_task_idx])

    def toggle_sample_tasks_on_reset(self, on: bool):
        self.sample_tasks_on_reset = on

    def __init__(
        self,
        env: Env,
        tasks: list[Task],
        sample_tasks_on_reset: bool = False,
    ):
        super().__init__(env)
        self.sample_tasks_on_reset = sample_tasks_on_reset
        self.tasks = tasks
        self.current_task_idx = -1

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.sample_tasks_on_reset:
            self._set_pseudo_random_task()
        return self.env.reset(seed=seed, options=options)

    def sample_tasks(self, *, seed: int | None = None, options: dict | None = None):
        self._set_pseudo_random_task()
        return self.env.reset(seed=seed, options=options)

    def get_checkpoint(self) -> dict:
        return {
            "tasks": [_serialize_task(task) for task in self.tasks],
            "current_task_idx": self.current_task_idx,
            "sample_tasks_on_reset": self.sample_tasks_on_reset,
            "env_rng_state": get_env_rng_checkpoint(self.unwrapped),
        }

    def load_checkpoint(self, ckpt: dict):
        assert "tasks" in ckpt
        assert "current_task_idx" in ckpt
        assert "sample_tasks_on_reset" in ckpt
        assert "env_rng_state" in ckpt

        self.tasks = [_deserialize_task(task) for task in ckpt["tasks"]]
        self.current_task_idx = ckpt["current_task_idx"]
        self.sample_tasks_on_reset = ckpt["sample_tasks_on_reset"]
        set_env_rng(self.unwrapped, ckpt["env_rng_state"])


class AutoTerminateOnSuccessWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically output a termination signal when the environment's task is solved.
    That is, when the 'success' key in the info dict is True.

    This is not the case by default in SawyerXYZEnv, because terminating on success during training leads to
    instability and poor evaluation performance. However, this behaviour is desired during said evaluation.
    Hence the existence of this wrapper.

    Best used *under* an AutoResetWrapper and RecordEpisodeStatistics and the like."""

    terminate_on_success: bool = True

    def __init__(self, env: Env):
        super().__init__(env)
        self.terminate_on_success = True

    def toggle_terminate_on_success(self, on: bool):
        self.terminate_on_success = on

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.terminate_on_success:
            terminated = info["success"] == 1.0
        return obs, reward, terminated, truncated, info


class NormalizeRewardsExponential(gym.Wrapper):
    def __init__(self, reward_alpha, env):
        super().__init__(env)
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.0
        self._reward_var = 1.0

    def _update_reward_estimate(self, reward):
        self._reward_mean = (
            1 - self._reward_alpha
        ) * self._reward_mean + self._reward_alpha * reward
        self._reward_var = (
            1 - self._reward_alpha
        ) * self._reward_var + self._reward_alpha * np.square(
            reward - self._reward_mean
        )

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    def step(self, action: NDArray):
        next_obs, reward, terminate, truncate, info = self.env.step(action)
        self._update_reward_estimate(reward)  # type: ignore
        reward = self._apply_normalize_reward(reward)  # type: ignore
        return next_obs, reward, terminate, truncate, info


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count


class CheckpointWrapper(gym.Wrapper):
    env_id: str

    def __init__(self, env: gym.Env, env_id: str):
        super().__init__(env)
        assert hasattr(self.env, "get_checkpoint") and callable(self.env.get_checkpoint)
        assert hasattr(self.env, "load_checkpoint") and callable(
            self.env.load_checkpoint
        )
        self.env_id = env_id

    def get_checkpoint(self) -> tuple[str, dict]:
        ckpt: dict = self.env.get_checkpoint()
        return (self.env_id, ckpt)

    def load_checkpoint(self, ckpts: list[tuple[str, dict]]) -> None:
        my_ckpt = None
        for env_id, ckpt in ckpts:
            if env_id == self.env_id:
                my_ckpt = ckpt
                break
        if my_ckpt is None:
            raise ValueError(
                f"Could not load checkpoint, no checkpoint found with id {self.env_id}. Checkpoint IDs: ",
                [env_id for env_id, _ in ckpts],
            )
        self.env.load_checkpoint(my_ckpt)


def get_env_rng_checkpoint(env: SawyerXYZEnv) -> dict[str, dict]:
    return {  # pyright: ignore [reportReturnType]
        "np_random_state": env.np_random.bit_generator.state,
        "action_space_rng_state": env.action_space.np_random.bit_generator.state,
        "obs_space_rng_state": env.observation_space.np_random.bit_generator.state,
        "goal_space_rng_state": env.goal_space.np_random.bit_generator.state,  # type: ignore
    }


def set_env_rng(env: SawyerXYZEnv, state: dict[str, dict]) -> None:
    assert "np_random_state" in state
    assert "action_space_rng_state" in state
    assert "obs_space_rng_state" in state
    assert "goal_space_rng_state" in state

    env.np_random.bit_generator.state = state["np_random_state"]
    env.action_space.np_random.bit_generator.state = state["action_space_rng_state"]
    env.observation_space.np_random.bit_generator.state = state["obs_space_rng_state"]
    env.goal_space.np_random.bit_generator.state = state["goal_space_rng_state"]  # type: ignore
