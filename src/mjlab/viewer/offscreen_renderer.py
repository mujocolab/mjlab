"""MuJoCo offscreen renderer for headless visualization."""

from typing import Any, Callable

import mujoco
import numpy as np

from mjlab.scene import Scene
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer
from mjlab.viewer.viewer_config import ViewerConfig

# Max number of envs to visualize (performance/memory limit).
# See FAQ: "How many environments can I visualize at once?"
_MAX_EXTRA_ENVS = 2


class OffscreenRenderer:
  def __init__(self, model: mujoco.MjModel, cfg: ViewerConfig, scene: Scene) -> None:
    self._cfg = cfg
    self._model = model
    self._data = mujoco.MjData(model)
    self._scene = scene
    self._orig_extent = float(self._model.stat.extent)
    self._render_extent = self._compute_render_extent()
    # Keep extent override local to offscreen rendering so shadow/camera scaling
    # is not dominated by the full multi-env world bounds.
    self._model.stat.extent = self._render_extent

    self._model.vis.global_.offheight = cfg.height
    self._model.vis.global_.offwidth = cfg.width
    if cfg.fovy is not None and cfg.origin_type == cfg.OriginType.WORLD:
      self._model.vis.global_.fovy = cfg.fovy

    if not cfg.enable_shadows:
      self._model.light_castshadow[:] = False
    if not cfg.enable_reflections:
      self._model.mat_reflectance[:] = 0.0

    self._cam = self._setup_camera()

    self._renderer: mujoco.Renderer | None = None
    self._opt = mujoco.MjvOption()
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

    nworld = int(data.nworld)
    if nworld <= 0:
      return

    env_idx = max(0, min(int(self._cfg.env_idx), nworld - 1))
    if self._model.nq > 0:
      self._data.qpos[:] = data.qpos[env_idx].cpu().numpy()
      self._data.qvel[:] = data.qvel[env_idx].cpu().numpy()
    if self._model.nmocap > 0:
      self._data.mocap_pos[:] = data.mocap_pos[env_idx].cpu().numpy()
      self._data.mocap_quat[:] = data.mocap_quat[env_idx].cpu().numpy()
    mujoco.mj_forward(self._model, self._data)

    cam = camera if camera is not None else self._cam
    self._renderer.update_scene(self._data, camera=cam)

    # Note: update_scene() resets the scene each frame, so no need to manually clear.
    if debug_vis_callback is not None:
      visualizer = MujocoNativeDebugVisualizer(
        self._renderer.scene, self._model, env_idx=self._cfg.env_idx
      )
      debug_vis_callback(visualizer)

    # Add nearest neighboring environments as geoms for context.
    for i in self._get_extra_env_ids(nworld, env_idx):
      if self._model.nq > 0:
        self._data.qpos[:] = data.qpos[i].cpu().numpy()
        self._data.qvel[:] = data.qvel[i].cpu().numpy()
      if self._model.nmocap > 0:
        self._data.mocap_pos[:] = data.mocap_pos[i].cpu().numpy()
        self._data.mocap_quat[:] = data.mocap_quat[i].cpu().numpy()
      mujoco.mj_forward(self._model, self._data)
      mujoco.mjv_addGeoms(
        self._model,
        self._data,
        self._opt,
        self._pert,
        self._catmask.value,
        self._renderer.scene,
      )

  def _get_extra_env_ids(self, nworld: int, env_idx: int) -> list[int]:
    """Return nearest neighboring env ids to render as context.

    We render a small local neighborhood around ``env_idx`` instead of the first
    N environments, so videos stay focused on the tracked robot and nearby peers.
    """
    if _MAX_EXTRA_ENVS <= 0 or nworld <= 1:
      return []

    k = min(_MAX_EXTRA_ENVS, nworld - 1)
    origins = self._scene.env_origins[:nworld].cpu().numpy()
    ref = origins[env_idx]
    dist2 = np.sum((origins - ref) ** 2, axis=1)
    dist2[env_idx] = np.inf

    nearest = np.argpartition(dist2, kth=k - 1)[:k]
    nearest = nearest[np.argsort(dist2[nearest])]
    return [int(i) for i in nearest]

  def render(self) -> np.ndarray:
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    return self._renderer.render()

  def _setup_camera(self) -> mujoco.MjvCamera:
    """Setup camera based on config's origin_type."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(self._model, camera)

    if self._cfg.origin_type == self._cfg.OriginType.WORLD:
      # Free camera, no tracking.
      camera.type = mujoco.mjtCamera.mjCAMERA_FREE.value
      camera.fixedcamid = -1
      camera.trackbodyid = -1

    elif self._cfg.origin_type == self._cfg.OriginType.ASSET_ROOT:
      from mjlab.entity import Entity

      if self._cfg.entity_name:
        robot: Entity = self._scene[self._cfg.entity_name]
      else:
        # Auto-detect if only one entity.
        if len(self._scene.entities) == 1:
          robot = list(self._scene.entities.values())[0]
        else:
          raise ValueError(
            f"Multiple entities in scene ({len(self._scene.entities)}). "
            "Specify entity_name to choose which one."
          )

      body_id = robot.indexing.root_body_id
      camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
      camera.trackbodyid = body_id
      camera.fixedcamid = -1

    elif self._cfg.origin_type == self._cfg.OriginType.ASSET_BODY:
      if not self._cfg.entity_name or not self._cfg.body_name:
        raise ValueError("entity_name/body_name required for ASSET_BODY origin type")

      from mjlab.entity import Entity

      robot: Entity = self._scene[self._cfg.entity_name]
      if self._cfg.body_name not in robot.body_names:
        raise ValueError(
          f"Body '{self._cfg.body_name}' not found in asset '{self._cfg.entity_name}'"
        )
      body_id_list, _ = robot.find_bodies(self._cfg.body_name)
      body_id = robot.indexing.bodies[body_id_list[0]].id

      camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
      camera.trackbodyid = body_id
      camera.fixedcamid = -1

    camera.lookat[:] = self._cfg.lookat
    camera.elevation = self._cfg.elevation
    camera.azimuth = self._cfg.azimuth
    camera.distance = self._cfg.distance

    return camera

  def close(self) -> None:
    if self._renderer is not None:
      self._renderer.close()
      self._renderer = None
    self._model.stat.extent = self._orig_extent

  def _compute_render_extent(self) -> float:
    """Compute a stable extent for offscreen rendering.

    MuJoCo scales z-near/z-far and shadow clip with ``model.stat.extent``. In large
    scenes this auto extent can become very large, which causes shadow-map precision
    artifacts in offscreen video rendering.

    We therefore use a local extent tied to the active camera distance.
    """
    # Keep enough room for the tracked subject and camera motion.
    return max(4.0, 1.5 * float(self._cfg.distance))
