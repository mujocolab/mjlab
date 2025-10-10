"""Script to use zero agent."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import gymnasium as gym
import torch
import tyro
from typing_extensions import assert_never

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserViewer


@dataclass(frozen=True)
class ZeroConfig:
  registry_name: str | None = None
  device: str | None = None
  render_all_envs: bool = False
  viewer: Literal["native", "viser"] = "native"


def run_zero(task: str, cfg: ZeroConfig) -> None:
  configure_torch_backends()

  device = cfg.device
  if device is None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f"[INFO]: Using device: {device}")

  env_cfg = cast(
    ManagerBasedRlEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point")
  )
  agent_cfg = cast(
    RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
  )

  if isinstance(env_cfg, TrackingEnvCfg):
    if not cfg.registry_name:
      raise ValueError("Must provide --registry-name for tracking tasks.")

    # Check if the registry name includes alias, if not, append ":latest".
    registry_name = cast(str, cfg.registry_name)
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    env_cfg.commands.motion.motion_file = str(Path(artifact.download()) / "motion.npz")

  env = gym.make(task, cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore

  class Policy:
    def __call__(self, obs) -> torch.Tensor:
      del obs  # Unused.
      return torch.zeros(action_shape, device=env.unwrapped.device)

  policy = Policy()

  if cfg.viewer == "native":
    NativeMujocoViewer(env, policy, render_all_envs=cfg.render_all_envs).run()
  elif cfg.viewer == "viser":
    ViserViewer(env, policy, render_all_envs=cfg.render_all_envs).run()
  else:
    assert_never(cfg.viewer)

  env.close()


def main():
  # Parse first argument to choose the task.
  task_prefix = "Mjlab-"
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(
      [k for k in gym.registry.keys() if k.startswith(task_prefix)]
    ),
    add_help=False,
    return_unknown_args=True,
  )
  del task_prefix

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")
  agent_cfg = load_cfg_from_registry(chosen_task, "rl_cfg_entry_point")
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  args = tyro.cli(
    ZeroConfig,
    args=remaining_args,
    default=ZeroConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del env_cfg, agent_cfg, remaining_args

  run_zero(chosen_task, args)


if __name__ == "__main__":
  main()
