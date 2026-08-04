"""MuJoCo offscreen renderer for headless visualization."""

import copy
from typing import Any, Callable

import mujoco
import numpy as np

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.viewer.model_sync import (
  VIEWER_MODEL_FIELDS,
  disable_model_sameframe_shortcuts,
  sync_model_fields,
)
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer
from mjlab.viewer.viewer_config import ViewerConfig, get_camera_body_id


class OffscreenRenderer:
  def __init__(
    self,
    model: mujoco.MjModel,
    cfg: ViewerConfig,
    scene: Scene,
    sim_model: Any | None = None,
    expanded_fields: set[str] | None = None,
  ) -> None:
    self._cfg = cfg
    self._sim_model = sim_model
    self._expanded_fields = expanded_fields

    # Copy the model to not globally change the model fields
    self._mj_model = copy.copy(model)
    self._mj_data = mujoco.MjData(self._mj_model)

    self._mj_model.vis.global_.offheight = cfg.height
    self._mj_model.vis.global_.offwidth = cfg.width

    if not cfg.enable_shadows:
      self._mj_model.light_castshadow[:] = False

    if not cfg.enable_reflections:
      self._mj_model.mat_reflectance[:] = 0.0

    if self._sim_model is not None:
      disable_model_sameframe_shortcuts(self._mj_model)

    # Keep extent override local to offscreen rendering so shadow/camera scaling
    # is not dominated by the full multi-env world bounds.
    self._mj_model.stat.extent = self._compute_render_extent(cfg.distance)

    # TODO(): This change alters the behavior compared to the previous implementation. Previously,
    #  fovy would be ignored if origin type is not AUTO or WORLD. This design choice seems somewhat
    #  counter-intuitive as one purposefully ignores the user value as the default is None.
    if cfg.fovy is not None:
      self._mj_model.vis.global_.fovy = cfg.fovy

    self._cam = self._setup_camera(
      cfg, mj_model=self._mj_model, entities=scene.entities
    )

    # Ids of the environments that are rendered each frame. The ids are computed once such that the
    # neighbor set is stable. Recomputing the environment ids at every step could cause the
    # video to flicker as the env_origins can mutate during training (e.g. the terrain curriculums).
    self._env_ids = self._get_environment_ids(cfg, scene.env_origins.cpu().numpy())

    self._renderer: mujoco.Renderer | None = None
    self._catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC
    self._pert = mujoco.MjvPerturb()
    self._opt = mujoco.MjvOption()
    self._opt.geomgroup[:] = np.array(cfg.geom_group, dtype=np.uint8)
    self._opt.sitegroup[:] = np.array(cfg.site_group, dtype=np.uint8)

  @property
  def renderer(self) -> mujoco.Renderer:
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    return self._renderer

  def initialize(self) -> None:
    if self._renderer is not None:
      raise RuntimeError(
        "Renderer is already initialized. Call 'close()' first to reinitialize."
      )
    self._renderer = mujoco.Renderer(
      model=self._mj_model, height=self._cfg.height, width=self._cfg.width
    )

  def update(
    self,
    data: Any,  # TODO(): Ideally one would fix this type annotation at the cost of more imports
    debug_vis_callback: Callable[[MujocoNativeDebugVisualizer], None] | None = None,
    camera: str | None = None,
  ) -> None:
    """Update renderer with simulation data."""
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    # TODO(): Should this raise an error? When should this case be triggered? Add comment why
    #  this is necessary.
    if int(data.nworld) <= 0:
      return

    # Render the primary env with update_scene: it frames the camera and draws the shared
    # world geoms. The debug overlay is drawn only for this env (get_env_indices resolves to
    # env_ids[0] unless show_all_envs is set), so overlays stay on the primary rendered env.
    primary_env_id = self._env_ids[0]
    self._sync_model_fields(env_idx=primary_env_id)
    self._sync_data_fields(data, env_idx=primary_env_id)

    cam = camera if camera is not None else self._cam
    self._renderer.update_scene(self._mj_data, camera=cam, scene_option=self._opt)

    # TODO(): The debug visualization is only called for ONE environment. Is this the desired
    #  behavior or not? Should it be applied to all rendered environments?
    # Note: update_scene() resets the scene each frame, so no need to manually clear.
    if debug_vis_callback is not None:
      visualizer = MujocoNativeDebugVisualizer(
        self._renderer.scene, self._mj_model, env_idx=primary_env_id
      )
      debug_vis_callback(visualizer)

    # Add the remaining envs as context geoms.
    for env_id in self._env_ids[1:]:
      self._sync_model_fields(env_idx=env_id)
      self._sync_data_fields(data, env_idx=env_id)

      mujoco.mjv_addGeoms(
        self._mj_model,
        self._mj_data,
        self._opt,
        self._pert,
        self._catmask.value,
        self._renderer.scene,
      )

  def render(self) -> np.ndarray:
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    return self._renderer.render()

  def close(self) -> None:
    if self._renderer is not None:
      self._renderer.close()
      self._renderer = None

  def _sync_data_fields(self, data: Any, env_idx: int) -> None:
    """Sync data fields into MjData and call mj_forward to compute the body positions."""
    if self._mj_model.nq > 0:
      self._mj_data.qpos[:] = data.qpos[env_idx].cpu().numpy()
      self._mj_data.qvel[:] = data.qvel[env_idx].cpu().numpy()

    if self._mj_model.nmocap > 0:
      self._mj_data.mocap_pos[:] = data.mocap_pos[env_idx].cpu().numpy()
      self._mj_data.mocap_quat[:] = data.mocap_quat[env_idx].cpu().numpy()

    # TODO(): Would a mj_kinematics be sufficient? I guess depends on the debug render default.
    mujoco.mj_forward(self._mj_model, self._mj_data)

  def _sync_model_fields(self, env_idx: int) -> None:
    """Sync visually relevant per-world model fields into the host MjModel."""
    if self._sim_model is None or self._expanded_fields is None:
      return

    fields = self._expanded_fields & VIEWER_MODEL_FIELDS
    sync_model_fields(self._mj_model, self._sim_model, fields, env_idx)

  @staticmethod
  def _get_environment_ids(
    cfg: ViewerConfig, env_origins: np.ndarray
  ) -> tuple[int, ...]:
    """Return the ids of the rendered environments.

    For the origin type ASSET_ROOT and ASSET_BODY, the closest environments next to the
    environment index are rendered. The distance is computed using the environment origin.

    For the origin type WORLD and AUTO, the closest environments to the camera viewpoint are
    rendered. The distance is computed using the camera look at position.
    """
    if cfg.max_extra_envs < 0:
      msg = f"'max_extra_envs' must be non-negative, got {cfg.max_extra_envs} < 0."
      raise ValueError(msg)

    n_env = 1 + cfg.max_extra_envs
    n_world = env_origins.shape[0]

    if n_env > n_world:
      msg = f"Number of environments (n = {n_env}) exceeds the number of worlds (n = {n_world})."
      raise ValueError(msg)

    if cfg.origin_type in (ViewerConfig.OriginType.AUTO, ViewerConfig.OriginType.WORLD):
      # Find environments closest to the camera viewpoint
      ref = np.array(cfg.lookat)

    else:
      # Find environments closest to the specified environment.
      if cfg.max_extra_envs == 0:
        return (cfg.env_idx,)

      ref = env_origins[cfg.env_idx]

    dist2 = np.sum((env_origins - ref) ** 2, axis=1)
    nearest = np.argpartition(dist2, kth=n_env - 1)[:n_env]
    nearest = nearest[np.argsort(dist2[nearest])]
    return tuple(int(i) for i in nearest)

  @staticmethod
  def _setup_camera(
    cfg: ViewerConfig,
    mj_model: mujoco.MjModel,
    entities: dict[str, Entity],
  ) -> mujoco.MjvCamera:
    """Setup camera based on config's origin_type."""
    # TODO(): Should one raise a ValueError for OriginType.AUTO as this is identical to
    #  OriginType.World, which does not match the ViewerConfig doc-string.

    # Infer the body id from the origin type. The body id is -1 when a free camera is used.
    body_id = get_camera_body_id(
      origin_type=cfg.origin_type,
      body_name=cfg.body_name,
      entity_name=cfg.entity_name,
      entities=entities,
    )

    # Infer the camera type
    origin_2_camera_type = {
      ViewerConfig.OriginType.AUTO: mujoco.mjtCamera.mjCAMERA_FREE.value,
      ViewerConfig.OriginType.WORLD: mujoco.mjtCamera.mjCAMERA_FREE.value,
      ViewerConfig.OriginType.ASSET_BODY: mujoco.mjtCamera.mjCAMERA_TRACKING.value,
      ViewerConfig.OriginType.ASSET_ROOT: mujoco.mjtCamera.mjCAMERA_TRACKING.value,
    }

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj_model, camera)
    camera.type = origin_2_camera_type[cfg.origin_type]
    camera.trackbodyid = body_id
    camera.fixedcamid = -1
    camera.elevation = cfg.elevation
    camera.azimuth = cfg.azimuth
    camera.distance = cfg.distance

    # The lookat is only relevant for the origin type WORLD as otherwise lookat is overridden
    # by the camera before rendering.
    camera.lookat[:] = cfg.lookat

    return camera

  @staticmethod
  def _compute_render_extent(distance: float) -> float:
    """Compute a stable extent for offscreen rendering from the camera distance.

    MuJoCo scales z-near/z-far and shadow clip with model.stat.extent. In large scenes
    this auto extent can become very large, which causes shadow-map precision artifacts
    in offscreen video rendering. We therefore use a local extent tied to the camera
    distance, keeping enough room for the tracked subject and camera motion.
    """
    return max(4.0, 1.5 * float(distance))
