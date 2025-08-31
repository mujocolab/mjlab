"""
Defines a renderer class for the MuJoCo Python native bindings.
"""

from __future__ import annotations
from mujoco import _enums
import mujoco._enums
from mujoco import _functions
from mujoco import _render
from mujoco import _structs
import mujoco._structs
from mujoco import gl_context
import numpy as np
import numpy

__all__: list[str] = ["Renderer", "gl_context", "np"]

class Renderer:
  """
  Renders MuJoCo scenes.
  """
  def __del__(self) -> None: ...
  def __enter__(self): ...
  def __exit__(self, exc_type, exc_value, traceback): ...
  def __init__(
    self,
    model: mujoco._structs.MjModel,
    height: int = 240,
    width: int = 320,
    max_geom: int = 10000,
    font_scale: mujoco._enums.mjtFontScale = ...,
  ) -> None:
    """
    Initializes a new `Renderer`.

    Args:
      model: an MjModel instance.
      height: image height in pixels.
      width: image width in pixels.
      max_geom: Optional integer specifying the maximum number of geoms that can
        be rendered in the same scene. If None this will be chosen automatically
        based on the estimated maximum number of renderable geoms in the model.
      font_scale: Optional enum specifying the font scale for text.

    Raises:
      ValueError: If `camera_id` is outside the valid range, or if `width` or
        `height` exceed the dimensions of MuJoCo's offscreen framebuffer.
    """
  def close(self) -> None:
    """
    Frees the resources used by the renderer.

    This method can be used directly:

    ```python
    renderer = Renderer(...)
    # Use renderer.
    renderer.close()
    ```

    or via a context manager:

    ```python
    with Renderer(...) as renderer:
      # Use renderer.
    ```
    """
  def disable_depth_rendering(self): ...
  def disable_segmentation_rendering(self): ...
  def enable_depth_rendering(self): ...
  def enable_segmentation_rendering(self): ...
  def render(self, *, out: typing.Optional[numpy.ndarray] = None) -> numpy.ndarray:
    """
    Renders the scene as a numpy array of pixel values.

    Args:
      out: Alternative output array in which to place the resulting pixels. It
        must have the same shape as the expected output but the type will be
        cast if necessary. The expted shape depends on the value of
        `self._depth_rendering`: when `True`, we expect `out.shape == (width,
        height)`, and `out.shape == (width, height, 3)` when `False`.

    Returns:
      A new numpy array holding the pixels with shape `(H, W)` or `(H, W, 3)`,
      depending on the value of `self._depth_rendering` unless
      `out is None`, in which case a reference to `out` is returned.

    Raises:
      RuntimeError: if this method is called after the close method.
    """
  def update_scene(
    self,
    data: mujoco._structs.MjData,
    camera: typing.Union[int, str, mujoco._structs.MjvCamera] = -1,
    scene_option: typing.Optional[mujoco._structs.MjvOption] = None,
  ):
    """
    Updates geometry used for rendering.

    Args:
      data: An instance of `MjData`.
      camera: An instance of `MjvCamera`, a string or an integer
      scene_option: A custom `MjvOption` instance to use to render the scene
        instead of the default.

    Raises:
      ValueError: If `camera_id` is outside the valid range, or if camera does
        not exist.
    """
  @property
  def height(self): ...
  @property
  def model(self): ...
  @property
  def scene(self) -> mujoco._structs.MjvScene: ...
  @property
  def width(self): ...
