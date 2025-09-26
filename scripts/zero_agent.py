from typing import Literal

import gymnasium as gym
import torch
import tyro
from typing_extensions import assert_never

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserViewer


def main(
  task: str,
  num_envs: int | None = None,
  device: str = "cuda:0",
  render_all_envs: bool = False,
  viewer: Literal["native", "viser"] = "native",
):
  configure_torch_backends()

  env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")

  # Detect task type by config class
  is_velocity = isinstance(env_cfg, LocomotionVelocityEnvCfg)

  if not is_velocity:
    raise RuntimeError(
      f"Unsupported env cfg type: {type(env_cfg).__name__}. Expected a Velocity task."
    )

  env_cfg.scene.num_envs = num_envs or env_cfg.scene.num_envs

  env = gym.make(task, cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore

  class Policy:
    def __call__(self, obs) -> torch.Tensor:
      del obs  # Unused.
      return torch.zeros(action_shape, device=env.unwrapped.device)

  policy = Policy()

  if viewer == "native":
    NativeMujocoViewer(env, policy, render_all_envs=render_all_envs).run()
  elif viewer == "viser":
    ViserViewer(env, policy, render_all_envs=render_all_envs).run()
  else:
    assert_never(viewer)

  env.close()


if __name__ == "__main__":
  tyro.cli(main)
