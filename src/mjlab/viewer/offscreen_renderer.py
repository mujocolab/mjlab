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
from mjlab.viewer.viewer_config import ViewerConfig


def _get_camera_body_id(cfg: ViewerConfig, entities: dict[str, Entity]) -> int:
  """Resolve the body id the camera tracks, or -1 for free cameras."""
  if cfg.origin_type in (
    ViewerConfig.OriginType.AUTO,
    ViewerConfig.OriginType.WORLD,
  ):
    return -1

  entity_name = cfg.entity_name
  if entity_name is None:
    if len(entities) != 1:
      raise ValueError(
        f"Cannot auto-detect entity: scene has {len(entities)} entities "
        f"({list(entities.keys())}). Set ViewerConfig.entity_name to choose one."
      )
    entity_name = next(iter(entities))
  elif entity_name not in entities:
    raise ValueError(
      f"Entity '{entity_name}' not found in scene. "
      f"Available entities: {list(entities.keys())}."
    )
  entity = entities[entity_name]

  if cfg.origin_type == ViewerConfig.OriginType.ASSET_ROOT:
    return entity.indexing.root_body_id

  assert cfg.origin_type == ViewerConfig.OriginType.ASSET_BODY
  if not cfg.body_name:
    raise ValueError("body_name is required for the ASSET_BODY origin type.")
  if cfg.body_name not in entity.body_names:
    raise ValueError(f"Body '{cfg.body_name}' not found in entity '{entity_name}'.")
  body_ids, _ = entity.find_bodies(cfg.body_name)
  return entity.indexing.bodies[body_ids[0]].id


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
    self._scene = scene
    self._sim_model = sim_model
    self._expanded_fields = expanded_fields

    # Work on a copy so render-only tweaks (extent, shadows, reflections,
    # sameframe shortcuts) never leak into the shared model.
    self._model = copy.copy(model)
    self._data = mujoco.MjData(self._model)

    if self._sim_model is not None:
      disable_model_sameframe_shortcuts(self._model)

    # Keep extent override local to offscreen rendering so shadow/camera scaling
    # is not dominated by the full multi-env world bounds.
    self._model.stat.extent = self._compute_render_extent(cfg.distance)

    self._model.vis.global_.offheight = cfg.height
    self._model.vis.global_.offwidth = cfg.width
    if cfg.fovy is not None:
      self._model.vis.global_.fovy = cfg.fovy

    if not cfg.enable_shadows:
      self._model.light_castshadow[:] = False
    if not cfg.enable_reflections:
      self._model.mat_reflectance[:] = 0.0

    self._cam = self._setup_camera(cfg, self._model, scene.entities)
    self._env_ids: tuple[int, ...] | None = None

    self._renderer: mujoco.Renderer | None = None
    self._opt = mujoco.MjvOption()
    self._opt.geomgroup[:] = np.array(cfg.geom_group, dtype=np.uint8)
    self._opt.sitegroup[:] = np.array(cfg.site_group, dtype=np.uint8)
    self._pert = mujoco.MjvPerturb()
    self._catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

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
      model=self._model, height=self._cfg.height, width=self._cfg.width
    )

  def update(
    self,
    data: Any,
    debug_vis_callback: Callable[[MujocoNativeDebugVisualizer], None] | None = None,
    camera: str | None = None,
  ) -> None:
    """Update renderer with simulation data."""
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    if int(data.nworld) <= 0:
      return

    if self._env_ids is None:
      self._env_ids = self._get_env_ids(
        self._cfg, self._scene.env_origins.cpu().numpy()
      )

    # The primary env frames the camera and receives the debug overlays.
    primary_env = self._env_ids[0]
    self._sync_model_fields(primary_env)
    self._sync_data_fields(data, primary_env)

    cam = camera if camera is not None else self._cam
    self._renderer.update_scene(self._data, camera=cam, scene_option=self._opt)

    # Note: update_scene() resets the scene each frame, so no need to manually clear.
    if debug_vis_callback is not None:
      visualizer = MujocoNativeDebugVisualizer(
        self._renderer.scene, self._model, env_idx=primary_env
      )
      debug_vis_callback(visualizer)

    # Add neighboring environments as geoms for context.
    for env_id in self._env_ids[1:]:
      self._sync_model_fields(env_id)
      self._sync_data_fields(data, env_id)
      mujoco.mjv_addGeoms(
        self._model,
        self._data,
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
    """Copy one env's state into the host MjData and refresh kinematics."""
    if self._model.nq > 0:
      self._data.qpos[:] = data.qpos[env_idx].cpu().numpy()
      self._data.qvel[:] = data.qvel[env_idx].cpu().numpy()
    if self._model.nmocap > 0:
      self._data.mocap_pos[:] = data.mocap_pos[env_idx].cpu().numpy()
      self._data.mocap_quat[:] = data.mocap_quat[env_idx].cpu().numpy()
    mujoco.mj_forward(self._model, self._data)

  def _sync_model_fields(self, env_idx: int) -> None:
    """Sync visually relevant per-world model fields into the host MjModel."""
    if self._sim_model is None or self._expanded_fields is None:
      return
    fields = self._expanded_fields & VIEWER_MODEL_FIELDS
    sync_model_fields(self._model, self._sim_model, fields, env_idx)

  @staticmethod
  def _get_env_ids(cfg: ViewerConfig, env_origins: np.ndarray) -> tuple[int, ...]:
    """Return the ids of the environments to render, primary env first.

    The primary env is cfg.env_idx (clamped to the available worlds), followed
    by up to max_extra_envs nearest neighbors by env origin, so videos stay
    focused on the tracked robot and nearby peers.

    The set is resolved once, on first use, and kept stable: env_origins can
    mutate during training (e.g. the terrain curriculum reassigns origins on
    reset), and recomputing every frame would make the context robots pop in
    and out, causing video flicker.
    """
    if cfg.max_extra_envs < 0:
      raise ValueError(
        f"'max_extra_envs' must be non-negative, got {cfg.max_extra_envs}."
      )

    n_world = env_origins.shape[0]
    env_idx = max(0, min(int(cfg.env_idx), n_world - 1))
    k = min(cfg.max_extra_envs, n_world - 1)
    if k <= 0:
      return (env_idx,)

    dist2 = np.sum((env_origins - env_origins[env_idx]) ** 2, axis=1)
    dist2[env_idx] = np.inf
    nearest = np.argpartition(dist2, kth=k - 1)[:k]
    nearest = nearest[np.argsort(dist2[nearest])]
    return (env_idx, *(int(i) for i in nearest))

  @staticmethod
  def _setup_camera(
    cfg: ViewerConfig,
    model: mujoco.MjModel,
    entities: dict[str, Entity],
  ) -> mujoco.MjvCamera:
    """Setup camera based on config's origin_type."""
    body_id = _get_camera_body_id(cfg, entities)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    if body_id == -1:
      camera.type = mujoco.mjtCamera.mjCAMERA_FREE.value
    else:
      camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
    camera.trackbodyid = body_id
    camera.fixedcamid = -1
    camera.lookat[:] = cfg.lookat
    camera.elevation = cfg.elevation
    camera.azimuth = cfg.azimuth
    camera.distance = cfg.distance

    return camera

  @staticmethod
  def _compute_render_extent(distance: float) -> float:
    """Compute a stable extent for offscreen rendering.

    MuJoCo scales z-near/z-far and shadow clip with ``model.stat.extent``. In
    large scenes this auto extent can become very large, which causes
    shadow-map precision artifacts in offscreen video rendering. We therefore
    use a local extent tied to the camera distance, keeping enough room for
    the tracked subject and camera motion.
    """
    return max(4.0, 1.5 * float(distance))
