from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


def ball_out_of_control(
  env: ManagerBasedRlEnv,
  max_distance: float,
  min_forward: float,
  max_forward: float,
  max_lateral: float,
  max_height: float,
  ignore_episode_hidden: bool = False,
  ignore_when_ball_unseen: bool = False,
  ignore_when_sensor_hidden: bool = False,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the football leaves the robot's recoverable control region.

  When requested, environments whose football observation is intentionally hidden
  for the whole episode are exempt so they can keep following the velocity command.
  """
  ball: Entity = env.scene[ball_cfg.name]
  robot: Entity = env.scene[asset_cfg.name]
  ball_relative_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
  ball_relative_b = quat_apply_inverse(robot.data.root_link_quat_w, ball_relative_w)

  planar_distance = torch.linalg.vector_norm(ball_relative_w[:, :2], dim=1)
  height_above_origin = ball.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  out_of_control = (
    (planar_distance > max_distance)
    | (ball_relative_b[:, 0] < min_forward)
    | (ball_relative_b[:, 0] > max_forward)
    | (torch.abs(ball_relative_b[:, 1]) > max_lateral)
    | (height_above_origin > max_height)
  )
  if ignore_episode_hidden:
    visual_cache = vars(env).get("_football_masked_ball_visual")
    if isinstance(visual_cache, dict):
      episode_hidden = visual_cache.get("episode_hidden")
      if isinstance(episode_hidden, torch.Tensor):
        if episode_hidden.shape == out_of_control.shape:
          out_of_control &= ~episode_hidden
  if ignore_when_ball_unseen:
    visual_cache = vars(env).get("_football_masked_ball_visual")
    if isinstance(visual_cache, dict):
      visible = visual_cache.get("visible")
      if isinstance(visible, torch.Tensor):
        if visible.shape == (out_of_control.shape[0], 1):
          out_of_control &= visible[:, 0].bool()
  if ignore_when_sensor_hidden:
    visual_cache = vars(env).get("_football_masked_ball_visual")
    if isinstance(visual_cache, dict):
      sensor_hidden = visual_cache.get("synthetic_hidden")
      if isinstance(sensor_hidden, torch.Tensor):
        if sensor_hidden.shape == out_of_control.shape:
          out_of_control &= ~sensor_hidden
  return out_of_control


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def out_of_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Truncate if robot leaves the generated terrain footprint.

  Returns all-false for non-generator terrains (e.g. plane).
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None or terrain.terrain_origins is None:
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )

  asset: Entity = env.scene[asset_cfg.name]
  root_xy_w = asset.data.root_link_pos_w[:, :2]

  # Use the generated grid shape (curriculum mode overrides cfg.num_cols with
  # len(sub_terrains)), and include the flat border around the patch grid.
  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  half_x = 0.5 * (num_rows * terrain_generator.size[0]) + terrain_generator.border_width
  half_y = 0.5 * (num_cols * terrain_generator.size[1]) + terrain_generator.border_width
  limit_x = max(0.0, half_x - margin)
  limit_y = max(0.0, half_y - margin)

  return (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)


def terrain_edge_reached(
  env: ManagerBasedRlEnv,
  threshold_fraction: float = 0.95,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when robot displacement from spawn exceeds sub-terrain size.

  Intended as ``time_out=True`` (successful traversal, not penalized). Skips the first
  2 steps after reset to avoid stale-position triggers.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  asset: Entity = env.scene[asset_cfg.name]
  displacement = (
    asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  ).abs()

  half_x = terrain_generator.size[0] / 2.0 * threshold_fraction
  half_y = terrain_generator.size[1] / 2.0 * threshold_fraction

  at_edge = (displacement[:, 0] > half_x) | (displacement[:, 1] > half_y)

  # Don't fire on the first 2 steps after reset (position may be stale).
  at_edge &= env.episode_length_buf > 2

  return at_edge
