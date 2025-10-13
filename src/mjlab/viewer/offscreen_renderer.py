from typing import TYPE_CHECKING, Any, Callable

import mujoco
import numpy as np

from mjlab.viewer.mujoco_native_visualizer import MujocoNativeDebugVisualizer
from mjlab.viewer.viewer_config import ViewerConfig

if TYPE_CHECKING:
  from mjlab.scene import Scene


class OffscreenRenderer:
  def __init__(self, model: mujoco.MjModel, cfg: ViewerConfig, scene: "Scene"):
    self._cfg = cfg
    self._model = model
    self._data = mujoco.MjData(model)
    self._scene = scene

    self._model.vis.global_.offheight = cfg.height
    self._model.vis.global_.offwidth = cfg.width

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
  ) -> None:
    """Update renderer with simulation data."""
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    # Copy selected environment's data for initial scene update
    env_idx = self._cfg.env_idx
    self._data.qpos[:] = data.qpos[env_idx].cpu().numpy()
    self._data.qvel[:] = data.qvel[env_idx].cpu().numpy()
    mujoco.mj_forward(self._model, self._data)
    self._renderer.update_scene(self._data, camera=self._cam)

    # Add debug visualizations if callback provided
    # Note: update_scene() resets the scene each frame, so no need to manually clear
    if debug_vis_callback is not None:
      visualizer = MujocoNativeDebugVisualizer(
        self._renderer.scene, self._model, env_idx=self._cfg.env_idx
      )
      debug_vis_callback(visualizer)

    # Add additional environments as geoms (up to 32 for performance)
    nworld = data.qpos.shape[0]
    for i in range(min(nworld, 32)):
      self._data.qpos[:] = data.qpos[i].cpu().numpy()
      self._data.qvel[:] = data.qvel[i].cpu().numpy()
      mujoco.mj_forward(self._model, self._data)
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

  def _setup_camera(self) -> mujoco.MjvCamera:
    """Setup camera based on config's origin_type."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(self._model, camera)

    if self._cfg.origin_type == self._cfg.OriginType.WORLD:
      # Free camera, no tracking (default behavior)
      camera.type = mujoco.mjtCamera.mjCAMERA_FREE.value
      camera.fixedcamid = -1
      camera.trackbodyid = -1

    elif self._cfg.origin_type == self._cfg.OriginType.ASSET_ROOT:
      from mjlab.entity import Entity

      # Get entity from scene
      if self._cfg.asset_name:
        robot: Entity = self._scene[self._cfg.asset_name]
      else:
        # Auto-detect if only one entity
        if len(self._scene.entities) == 1:
          robot = list(self._scene.entities.values())[0]
        else:
          raise ValueError(
            f"Multiple entities in scene ({len(self._scene.entities)}). "
            "Specify asset_name to choose which one."
          )

      body_id = robot.indexing.root_body_id
      camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
      camera.trackbodyid = body_id
      camera.fixedcamid = -1

    elif self._cfg.origin_type == self._cfg.OriginType.ASSET_BODY:
      if not self._cfg.asset_name or not self._cfg.body_name:
        raise ValueError("asset_name/body_name required for ASSET_BODY origin type")

      from mjlab.entity import Entity

      robot: Entity = self._scene[self._cfg.asset_name]
      if self._cfg.body_name not in robot.body_names:
        raise ValueError(
          f"Body '{self._cfg.body_name}' not found in asset '{self._cfg.asset_name}'"
        )
      body_id_list, _ = robot.find_bodies(self._cfg.body_name)
      body_id = robot.indexing.bodies[body_id_list[0]].id

      camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
      camera.trackbodyid = body_id
      camera.fixedcamid = -1

    # Apply camera parameters from config
    camera.lookat[:] = self._cfg.lookat
    camera.elevation = self._cfg.elevation
    camera.azimuth = self._cfg.azimuth
    camera.distance = self._cfg.distance

    return camera

  def close(self) -> None:
    if self._renderer is not None:
      self._renderer.close()
      self._renderer = None
