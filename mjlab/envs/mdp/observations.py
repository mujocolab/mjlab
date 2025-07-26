"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entities.robots.robot import Robot

if TYPE_CHECKING:
  from mjlab.envs.manager_based_env import ManagerBasedEnv
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

##
# Root state.
##


def base_lin_vel(
  env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  asset: Robot = env.scene.entities[asset_cfg.name]
  return asset.data.root_link_lin_vel_b


def base_ang_vel(
  env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  asset: Robot = env.scene.entities[asset_cfg.name]
  return asset.data.root_link_ang_vel_b


def projected_gravity(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Robot = env.scene.entities[asset_cfg.name]
  return asset.data.projected_gravity_b


##
# Joint state.
##


def joint_pos_rel(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Robot = env.scene.entities[asset_cfg.name]
  return asset.data.joint_pos - asset.data.default_joint_pos


def joint_vel(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Robot = env.scene.entities[asset_cfg.name]
  return asset.data.joint_vel


##
# Actions.
##


def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_actions


##
# Commands.
##


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return env.command_manager.get_command(command_name)
