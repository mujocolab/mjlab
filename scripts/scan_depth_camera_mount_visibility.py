"""Scan adjustable depth-camera poses for left/right sole visibility."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from mjlab.scripts.sim2sim.g1_football_depth import build_model
from mjlab.tasks.velocity_football_depth import (
  TEMPORAL_MOUNT_RANGE_FROZEN_MLP_DISTILLATION_TASK_ID,
)
from mjlab.tasks.velocity_football_depth.env_cfg import (
  DEPTH_CAMERA_LOWER_ESTIMATE_POS,
  DEPTH_CAMERA_LOWER_ESTIMATE_QUAT,
  DEPTH_CAMERA_UPPER_POS,
  DEPTH_CAMERA_UPPER_QUAT,
  DEPTH_SENSOR_NAME,
)

ALPHAS = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 2.0 / 3.0)
FOVY_DEGREES = 40.5
RAW_DEPTH_HEIGHT = 240
RAW_CROP_Y_PIXELS = (-4, 4)


def _wxyz_to_rotation(quaternion: np.ndarray) -> Rotation:
  return Rotation.from_quat(quaternion[[1, 2, 3, 0]])


def _rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
  xyzw = rotation.as_quat()
  return xyzw[[3, 0, 1, 2]]


def _shift_segmentation_y(segmentation: np.ndarray, shift: int) -> np.ndarray:
  height = segmentation.shape[0]
  source_rows = np.arange(height) + shift
  valid = (source_rows >= 0) & (source_rows < height)
  shifted = np.full_like(segmentation, -1)
  shifted[valid] = segmentation[source_rows[valid]]
  return shifted


def _foot_stats(
  segmentation: np.ndarray, geom_ids: np.ndarray
) -> tuple[int, int]:
  mask = (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
    segmentation[..., 0], geom_ids
  )
  rows = np.where(mask)[0]
  if rows.size == 0:
    return 0, -1
  bottom_margin = segmentation.shape[0] - 1 - int(rows.max())
  return int(mask.sum()), bottom_margin


def main() -> None:
  model, _, _, depth_cfg = build_model(
    task_id=TEMPORAL_MOUNT_RANGE_FROZEN_MLP_DISTILLATION_TASK_ID
  )
  data = mujoco.MjData(model)
  init_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, DEPTH_SENSOR_NAME
  )
  left_collision_ids = np.asarray(
    [
      i
      for i in range(model.ngeom)
      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
        "robot/left_foot"
      )
    ]
  )
  right_collision_ids = np.asarray(
    [
      i
      for i in range(model.ngeom)
      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
        "robot/right_foot"
      )
    ]
  )
  if left_collision_ids.size == 0 or right_collision_ids.size == 0:
    geom_names = [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
      for i in range(model.ngeom)
    ]
    foot_like = [name for name in geom_names if "foot" in name.lower()]
    raise RuntimeError(
      f"Could not resolve foot geoms: left={left_collision_ids.tolist()}, "
      f"right={right_collision_ids.tolist()}, foot-like names={foot_like}"
    )
  left_body_ids = np.unique(model.geom_bodyid[left_collision_ids])
  right_body_ids = np.unique(model.geom_bodyid[right_collision_ids])
  left_ids = np.flatnonzero(np.isin(model.geom_bodyid, left_body_ids))
  right_ids = np.flatnonzero(np.isin(model.geom_bodyid, right_body_ids))

  renderer = mujoco.Renderer(
    model, height=depth_cfg.height, width=depth_cfg.width
  )
  crop_y_pixels = tuple(
    round(value * depth_cfg.height / RAW_DEPTH_HEIGHT)
    for value in RAW_CROP_Y_PIXELS
  )
  option = mujoco.MjvOption()
  option.geomgroup[:] = 0
  for group in depth_cfg.enabled_geom_groups:
    option.geomgroup[group] = 1

  upper_position = np.asarray(DEPTH_CAMERA_UPPER_POS, dtype=np.float64)
  upper_rotation = _wxyz_to_rotation(
    np.asarray(DEPTH_CAMERA_UPPER_QUAT, dtype=np.float64)
  )
  lower_base_position = np.asarray(
    DEPTH_CAMERA_LOWER_ESTIMATE_POS, dtype=np.float64
  )
  lower_base_position[1] = DEPTH_CAMERA_UPPER_POS[1]
  lower_base_rotation = _wxyz_to_rotation(
    np.asarray(DEPTH_CAMERA_LOWER_ESTIMATE_QUAT, dtype=np.float64)
  )

  print(
    "alpha,min_left_pixels,min_right_pixels,min_left_bottom_margin,"
    "min_right_bottom_margin"
  )
  try:
    renderer.enable_segmentation_rendering()
    for alpha in ALPHAS:
      worst = [10**9, 10**9, 10**9, 10**9]
      for crop_y in crop_y_pixels:
        lower_position = lower_base_position.copy()
        position = (1.0 - alpha) * lower_position + alpha * upper_position

        rotations = Rotation.from_quat(
          np.stack(
            [lower_base_rotation.as_quat(), upper_rotation.as_quat()], axis=0
          )
        )
        rotation = Slerp([0.0, 1.0], rotations)([alpha])[0]
        model.cam_pos[camera_id] = position
        model.cam_quat[camera_id] = _rotation_to_wxyz(rotation)
        model.cam_fovy[camera_id] = FOVY_DEGREES

        mujoco.mj_resetDataKeyframe(model, data, init_key)
        mujoco.mj_forward(model, data)
        renderer.update_scene(
          data, camera=DEPTH_SENSOR_NAME, scene_option=option
        )
        segmentation = np.asarray(renderer.render()).copy()
        segmentation = _shift_segmentation_y(segmentation, crop_y)
        left_pixels, left_margin = _foot_stats(segmentation, left_ids)
        right_pixels, right_margin = _foot_stats(segmentation, right_ids)
        worst[0] = min(worst[0], left_pixels)
        worst[1] = min(worst[1], right_pixels)
        worst[2] = min(worst[2], left_margin)
        worst[3] = min(worst[3], right_margin)
      print(f"{alpha:.6f},{worst[0]},{worst[1]},{worst[2]},{worst[3]}")
  finally:
    renderer.close()


if __name__ == "__main__":
  main()
