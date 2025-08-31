"""
Interactive GUI viewer for MuJoCo.
"""

from __future__ import annotations
import abc as abc
import atexit as atexit
import contextlib as contextlib
import glfw as glfw
import math as math
import mujoco as mujoco
from mujoco import _simulate
from mujoco._simulate import Simulate as _Simulate
import numpy as np
import os as os
import queue as queue
import sys as sys
import threading as threading
import time as time
import typing
import weakref as weakref

__all__: list[str] = [
  "CallbackType",
  "Handle",
  "KeyCallbackType",
  "LoaderType",
  "MAX_SYNC_MISALIGN",
  "PERCENT_REALTIME",
  "SIM_REFRESH_FRACTION",
  "abc",
  "atexit",
  "contextlib",
  "glfw",
  "launch",
  "launch_from_path",
  "launch_passive",
  "math",
  "mujoco",
  "np",
  "os",
  "queue",
  "sys",
  "threading",
  "time",
  "weakref",
]

class Handle:
  """
  A handle for interacting with a MuJoCo viewer.
  """
  def __enter__(self): ...
  def __exit__(self, exc_type, exc_val, exc_tb): ...
  def __init__(
    self,
    sim: mujoco._simulate.Simulate,
    cam: mujoco._structs.MjvCamera,
    opt: mujoco._structs.MjvOption,
    pert: mujoco._structs.MjvPerturb,
    user_scn: typing.Optional[mujoco._structs.MjvScene],
  ): ...
  def _get_sim(self) -> typing.Optional[mujoco._simulate.Simulate]: ...
  def clear_figures(self): ...
  def clear_images(self): ...
  def clear_texts(self): ...
  def close(self): ...
  def is_running(self) -> bool: ...
  def lock(self): ...
  def set_figures(
    self,
    viewports_figures: typing.Union[
      typing.Tuple[mujoco._render.MjrRect, mujoco._structs.MjvFigure],
      typing.List[typing.Tuple[mujoco._render.MjrRect, mujoco._structs.MjvFigure]],
    ],
  ):
    """
    Overlay figures on the viewer.

    Args:
      viewports_figures: Single tuple or list of tuples of (viewport, figure)
        viewport: Rectangle defining position and size of the figure
        figure: MjvFigure object containing the figure data to display
    """
  def set_images(
    self,
    viewports_images: typing.Union[
      typing.Tuple[mujoco._render.MjrRect, numpy.ndarray],
      typing.List[typing.Tuple[mujoco._render.MjrRect, numpy.ndarray]],
    ],
  ):
    """
    Overlay images on the viewer.

    Args:
      viewports_images: Single tuple or list of tuples of (viewport, image)
        viewport: Rectangle defining position and size of the image
        image: RGB image with shape (height, width, 3)
    """
  def set_texts(
    self,
    texts: typing.Union[
      typing.Tuple[
        typing.Optional[int],
        typing.Optional[int],
        typing.Optional[str],
        typing.Optional[str],
      ],
      typing.List[
        typing.Tuple[
          typing.Optional[int],
          typing.Optional[int],
          typing.Optional[str],
          typing.Optional[str],
        ]
      ],
    ],
  ):
    """
    Overlay text on the viewer.

    Args:
      texts: Single tuple or list of tuples of (font, gridpos, text1, text2)
        font: Font style from mujoco.mjtFontScale
        gridpos: Position of text box from mujoco.mjtGridPos
        text1: Left text column, defaults to empty string if None
        text2: Right text column, defaults to empty string if None
    """
  def sync(self, state_only: bool = False): ...
  def update_hfield(self, hfieldid: int): ...
  def update_mesh(self, meshid: int): ...
  def update_texture(self, texid: int): ...
  @property
  def cam(self): ...
  @property
  def d(self): ...
  @property
  def m(self): ...
  @property
  def opt(self): ...
  @property
  def perturb(self): ...
  @property
  def user_scn(self): ...
  @property
  def viewport(self): ...

class _MjPythonBase:
  __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset()
  _abc_impl: typing.ClassVar[_abc._abc_data]  # value = <_abc._abc_data object>
  def launch_on_ui_thread(
    self,
    model: mujoco._structs.MjModel,
    data: mujoco._structs.MjData,
    handle_return: typing.Optional[ForwardRef("queue.Queue[Handle]")],
    key_callback: typing.Optional[typing.Callable[[int], NoneType]],
  ): ...

def _file_loader(
  path: str,
) -> typing.Callable[
  [], typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData, str]
]:
  """
  Loads an MJCF model from file path.
  """

def _launch_internal(
  model: typing.Optional[mujoco._structs.MjModel] = None,
  data: typing.Optional[mujoco._structs.MjData] = None,
  *,
  run_physics_thread: bool,
  loader: typing.Union[
    typing.Callable[[], typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData]],
    typing.Callable[
      [], typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData, str]
    ],
    NoneType,
  ] = None,
  handle_return: typing.Optional[ForwardRef("queue.Queue[Handle]")] = None,
  key_callback: typing.Optional[typing.Callable[[int], NoneType]] = None,
  show_left_ui: bool = True,
  show_right_ui: bool = True,
) -> None:
  """
  Internal API, so that the public API has more readable type annotations.
  """

def _physics_loop(
  simulate: mujoco._simulate.Simulate,
  loader: typing.Union[
    typing.Callable[[], typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData]],
    typing.Callable[
      [], typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData, str]
    ],
    NoneType,
  ],
):
  """
  Physics loop for the GUI, to be run in a separate thread.
  """

def _reload(
  simulate: mujoco._simulate.Simulate,
  loader: typing.Union[
    typing.Callable[[], typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData]],
    typing.Callable[
      [], typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData, str]
    ],
  ],
  notify_loaded: typing.Optional[typing.Callable[[], NoneType]] = None,
) -> typing.Optional[typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData]]:
  """
  Internal function for reloading a model in the viewer.
  """

def launch(
  model: typing.Optional[mujoco._structs.MjModel] = None,
  data: typing.Optional[mujoco._structs.MjData] = None,
  *,
  loader: typing.Optional[
    typing.Callable[[], typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData]]
  ] = None,
  show_left_ui: bool = True,
  show_right_ui: bool = True,
) -> None:
  """
  Launches the Simulate GUI.
  """

def launch_from_path(path: str) -> None:
  """
  Launches the Simulate GUI from file path.
  """

def launch_passive(
  model: mujoco._structs.MjModel,
  data: mujoco._structs.MjData,
  *,
  key_callback: typing.Optional[typing.Callable[[int], NoneType]] = None,
  show_left_ui: bool = True,
  show_right_ui: bool = True,
) -> Handle:
  """
  Launches a passive Simulate GUI without blocking the running thread.
  """

CallbackType: typing._CallableGenericAlias  # value = typing.Callable[[mujoco._structs.MjModel, mujoco._structs.MjData], NoneType]
KeyCallbackType: (
  typing._CallableGenericAlias
)  # value = typing.Callable[[int], NoneType]
LoaderType: typing._CallableGenericAlias  # value = typing.Callable[[], typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData]]
MAX_SYNC_MISALIGN: float = 0.1
PERCENT_REALTIME: tuple = (
  100,
  80,
  66,
  50,
  40,
  33,
  25,
  20,
  16,
  13,
  10,
  8,
  6.6,
  5,
  4,
  3.3,
  2.5,
  2,
  1.6,
  1.3,
  1,
  0.8,
  0.66,
  0.5,
  0.4,
  0.33,
  0.25,
  0.2,
  0.16,
  0.13,
  0.1,
)
SIM_REFRESH_FRACTION: float = 0.7
_InternalLoaderType: typing._UnionGenericAlias  # value = typing.Union[typing.Callable[[], typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData]], typing.Callable[[], typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData, str]]]
_LoaderWithPathType: typing._CallableGenericAlias  # value = typing.Callable[[], typing.Tuple[mujoco._structs.MjModel, mujoco._structs.MjData, str]]
_MJPYTHON = None
