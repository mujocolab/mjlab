from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import mujoco
import torch
from conftest import get_test_device

from mjlab.sensor import CameraSensorData
from mjlab.tasks.manipulation.mdp.commands import MultiCubeLiftingCommand
from mjlab.tasks.manipulation.mdp.observations import (
  camera_segmentation,
  camera_target_cube_mask,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _make_env(segmentation: torch.Tensor, target_geom_ids: torch.Tensor):
  sensor = SimpleNamespace(data=CameraSensorData(segmentation=segmentation))

  command = object.__new__(MultiCubeLiftingCommand)
  command._padded_geom_ids = target_geom_ids
  command.target_selection = torch.arange(
    target_geom_ids.shape[0], device=target_geom_ids.device
  )

  command_manager = SimpleNamespace(get_term=lambda _: command)
  return SimpleNamespace(
    scene={"seg_cam": sensor},
    command_manager=command_manager,
  )


def test_camera_segmentation_returns_bchw():
  device = get_test_device()
  geom = int(mujoco.mjtObj.mjOBJ_GEOM)
  seg = torch.tensor(
    [
      [[[1, geom], [2, geom], [-1, -1]], [[3, geom], [4, geom], [-1, -1]]],
      [[[5, geom], [6, geom], [-1, -1]], [[7, geom], [8, geom], [-1, -1]]],
    ],
    dtype=torch.int32,
    device=device,
  )
  env = _make_env(seg, torch.tensor([[1], [7]], dtype=torch.int32, device=device))
  env = cast("ManagerBasedRlEnv", env)

  obs = camera_segmentation(env, "seg_cam")

  assert obs.shape == (2, 2, 2, 3)
  assert obs.dtype == torch.int32
  assert torch.equal(obs[:, 0], seg[..., 0])
  assert torch.equal(obs[:, 1], seg[..., 1])


def test_camera_target_cube_mask_filters_to_geom_hits():
  device = get_test_device()
  geom = int(mujoco.mjtObj.mjOBJ_GEOM)
  flex = int(mujoco.mjtObj.mjOBJ_FLEX)
  seg = torch.tensor(
    [
      [[[3, geom], [3, flex], [0, geom]], [[-1, -1], [4, geom], [3, geom]]],
      [[[5, geom], [7, flex], [5, geom]], [[7, geom], [-1, -1], [0, geom]]],
    ],
    dtype=torch.int32,
    device=device,
  )
  env = _make_env(seg, torch.tensor([[3], [7]], dtype=torch.int32, device=device))
  env = cast("ManagerBasedRlEnv", env)

  mask = camera_target_cube_mask(env, "seg_cam", "lift_height")

  expected = torch.tensor(
    [
      [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
      [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
    ],
    dtype=torch.float32,
    device=device,
  )
  assert mask.shape == (2, 1, 2, 3)
  assert torch.equal(mask, expected)
