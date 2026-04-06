"""
Exports GLContext for MuJoCo Python bindings.
"""
from __future__ import annotations
import ctypes as ctypes
from mujoco.cgl import GLContext
from mujoco.cgl import GLContext as _GLContext
import os as os
import platform as platform
__all__: list[str] = ['GLContext', 'ctypes', 'os', 'platform']
_MUJOCO_GL: str = ''
_SYSTEM: str = 'Darwin'
_VALID_MUJOCO_GL: tuple = ('enable', 'enabled', 'on', 'true', '1', 'glfw', '', 'cgl')
