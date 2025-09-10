"""
An OpenGL context created via GLFW.
"""

from __future__ import annotations
import glfw as glfw

__all__: list[str] = ["GLContext", "glfw"]

class GLContext:
  """
  An OpenGL context created via GLFW.
  """
  def __del__(self): ...
  def __init__(self, max_width, max_height): ...
  def free(self): ...
  def make_current(self): ...
