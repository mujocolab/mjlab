from __future__ import annotations
import mujoco._structs
import numpy
import numpy.typing
import typing

__all__: list[str] = [
  "MjrContext",
  "MjrRect",
  "mjr_addAux",
  "mjr_blitAux",
  "mjr_blitBuffer",
  "mjr_changeFont",
  "mjr_drawPixels",
  "mjr_figure",
  "mjr_findRect",
  "mjr_finish",
  "mjr_getError",
  "mjr_label",
  "mjr_maxViewport",
  "mjr_overlay",
  "mjr_readPixels",
  "mjr_rectangle",
  "mjr_render",
  "mjr_resizeOffscreen",
  "mjr_restoreBuffer",
  "mjr_setAux",
  "mjr_setBuffer",
  "mjr_text",
  "mjr_uploadHField",
  "mjr_uploadMesh",
  "mjr_uploadTexture",
]

class MjrContext:
  @typing.overload
  def __init__(self) -> None: ...
  @typing.overload
  def __init__(
    self, arg0: mujoco._structs.MjModel, arg1: typing.SupportsInt
  ) -> None: ...
  def free(self) -> None:
    """
    Frees resources in current active OpenGL context, sets struct to default.
    """
  @property
  def auxColor(self) -> numpy.typing.NDArray[numpy.uint32]: ...
  @auxColor.setter
  def auxColor(self, arg1: typing.Any) -> None: ...
  @property
  def auxColor_r(self) -> numpy.typing.NDArray[numpy.uint32]: ...
  @auxColor_r.setter
  def auxColor_r(self, arg1: typing.Any) -> None: ...
  @property
  def auxFBO(self) -> numpy.typing.NDArray[numpy.uint32]: ...
  @auxFBO.setter
  def auxFBO(self, arg1: typing.Any) -> None: ...
  @property
  def auxFBO_r(self) -> numpy.typing.NDArray[numpy.uint32]: ...
  @auxFBO_r.setter
  def auxFBO_r(self, arg1: typing.Any) -> None: ...
  @property
  def auxHeight(self) -> numpy.typing.NDArray[numpy.int32]: ...
  @auxHeight.setter
  def auxHeight(self, arg1: typing.Any) -> None: ...
  @property
  def auxSamples(self) -> numpy.typing.NDArray[numpy.int32]: ...
  @auxSamples.setter
  def auxSamples(self, arg1: typing.Any) -> None: ...
  @property
  def auxWidth(self) -> numpy.typing.NDArray[numpy.int32]: ...
  @auxWidth.setter
  def auxWidth(self, arg1: typing.Any) -> None: ...
  @property
  def baseBuiltin(self) -> int: ...
  @baseBuiltin.setter
  def baseBuiltin(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def baseFontBig(self) -> int: ...
  @baseFontBig.setter
  def baseFontBig(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def baseFontNormal(self) -> int: ...
  @baseFontNormal.setter
  def baseFontNormal(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def baseFontShadow(self) -> int: ...
  @baseFontShadow.setter
  def baseFontShadow(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def baseHField(self) -> int: ...
  @baseHField.setter
  def baseHField(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def baseMesh(self) -> int: ...
  @baseMesh.setter
  def baseMesh(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def basePlane(self) -> int: ...
  @basePlane.setter
  def basePlane(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def charHeight(self) -> int: ...
  @charHeight.setter
  def charHeight(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def charHeightBig(self) -> int: ...
  @charHeightBig.setter
  def charHeightBig(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def charWidth(self) -> numpy.typing.NDArray[numpy.int32]: ...
  @charWidth.setter
  def charWidth(self, arg1: typing.Any) -> None: ...
  @property
  def charWidthBig(self) -> numpy.typing.NDArray[numpy.int32]: ...
  @charWidthBig.setter
  def charWidthBig(self, arg1: typing.Any) -> None: ...
  @property
  def currentBuffer(self) -> int: ...
  @currentBuffer.setter
  def currentBuffer(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def fogEnd(self) -> float: ...
  @fogEnd.setter
  def fogEnd(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def fogRGBA(self) -> numpy.typing.NDArray[numpy.float32]: ...
  @fogRGBA.setter
  def fogRGBA(self, arg1: typing.Any) -> None: ...
  @property
  def fogStart(self) -> float: ...
  @fogStart.setter
  def fogStart(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def fontScale(self) -> int: ...
  @fontScale.setter
  def fontScale(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def glInitialized(self) -> int: ...
  @glInitialized.setter
  def glInitialized(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def lineWidth(self) -> float: ...
  @lineWidth.setter
  def lineWidth(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def mat_texid(self) -> numpy.typing.NDArray[numpy.int32]: ...
  @mat_texid.setter
  def mat_texid(self, arg1: typing.Any) -> None: ...
  @property
  def mat_texrepeat(self) -> numpy.typing.NDArray[numpy.float32]: ...
  @mat_texrepeat.setter
  def mat_texrepeat(self, arg1: typing.Any) -> None: ...
  @property
  def mat_texuniform(self) -> numpy.typing.NDArray[numpy.int32]: ...
  @mat_texuniform.setter
  def mat_texuniform(self, arg1: typing.Any) -> None: ...
  @property
  def nskin(self) -> int: ...
  @property
  def ntexture(self) -> int: ...
  @ntexture.setter
  def ntexture(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def offColor(self) -> int: ...
  @offColor.setter
  def offColor(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def offColor_r(self) -> int: ...
  @offColor_r.setter
  def offColor_r(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def offDepthStencil(self) -> int: ...
  @offDepthStencil.setter
  def offDepthStencil(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def offDepthStencil_r(self) -> int: ...
  @offDepthStencil_r.setter
  def offDepthStencil_r(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def offFBO(self) -> int: ...
  @offFBO.setter
  def offFBO(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def offFBO_r(self) -> int: ...
  @offFBO_r.setter
  def offFBO_r(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def offHeight(self) -> int: ...
  @offHeight.setter
  def offHeight(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def offSamples(self) -> int: ...
  @offSamples.setter
  def offSamples(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def offWidth(self) -> int: ...
  @offWidth.setter
  def offWidth(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def rangeBuiltin(self) -> int: ...
  @rangeBuiltin.setter
  def rangeBuiltin(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def rangeFont(self) -> int: ...
  @rangeFont.setter
  def rangeFont(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def rangeHField(self) -> int: ...
  @rangeHField.setter
  def rangeHField(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def rangeMesh(self) -> int: ...
  @rangeMesh.setter
  def rangeMesh(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def rangePlane(self) -> int: ...
  @rangePlane.setter
  def rangePlane(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def readDepthMap(self) -> int: ...
  @readDepthMap.setter
  def readDepthMap(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def readPixelFormat(self) -> int: ...
  @readPixelFormat.setter
  def readPixelFormat(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def shadowClip(self) -> float: ...
  @shadowClip.setter
  def shadowClip(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def shadowFBO(self) -> int: ...
  @shadowFBO.setter
  def shadowFBO(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def shadowScale(self) -> float: ...
  @shadowScale.setter
  def shadowScale(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def shadowSize(self) -> int: ...
  @shadowSize.setter
  def shadowSize(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def shadowTex(self) -> int: ...
  @shadowTex.setter
  def shadowTex(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def skinfaceVBO(self) -> tuple: ...
  @property
  def skinnormalVBO(self) -> tuple: ...
  @property
  def skintexcoordVBO(self) -> tuple: ...
  @property
  def skinvertVBO(self) -> tuple: ...
  @property
  def texture(self) -> numpy.typing.NDArray[numpy.uint32]: ...
  @texture.setter
  def texture(self, arg1: typing.Any) -> None: ...
  @property
  def textureType(self) -> numpy.typing.NDArray[numpy.int32]: ...
  @textureType.setter
  def textureType(self, arg1: typing.Any) -> None: ...
  @property
  def windowAvailable(self) -> int: ...
  @windowAvailable.setter
  def windowAvailable(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def windowDoublebuffer(self) -> int: ...
  @windowDoublebuffer.setter
  def windowDoublebuffer(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def windowSamples(self) -> int: ...
  @windowSamples.setter
  def windowSamples(self, arg1: typing.SupportsFloat) -> None: ...
  @property
  def windowStereo(self) -> int: ...
  @windowStereo.setter
  def windowStereo(self, arg1: typing.SupportsFloat) -> None: ...

class MjrRect:
  __hash__: typing.ClassVar[None] = None
  def __copy__(self) -> MjrRect: ...
  def __deepcopy__(self, arg0: dict) -> MjrRect: ...
  def __eq__(self, arg0: typing.Any) -> bool: ...
  def __init__(
    self,
    left: typing.SupportsInt,
    bottom: typing.SupportsInt,
    width: typing.SupportsInt,
    height: typing.SupportsInt,
  ) -> None: ...
  def __repr__(self) -> str: ...
  @property
  def bottom(self) -> int: ...
  @bottom.setter
  def bottom(self, arg0: typing.SupportsInt) -> None: ...
  @property
  def height(self) -> int: ...
  @height.setter
  def height(self, arg0: typing.SupportsInt) -> None: ...
  @property
  def left(self) -> int: ...
  @left.setter
  def left(self, arg0: typing.SupportsInt) -> None: ...
  @property
  def width(self) -> int: ...
  @width.setter
  def width(self, arg0: typing.SupportsInt) -> None: ...

def mjr_addAux(
  index: typing.SupportsInt,
  width: typing.SupportsInt,
  height: typing.SupportsInt,
  samples: typing.SupportsInt,
  con: MjrContext,
) -> None:
  """
  Add Aux buffer with given index to context; free previous Aux buffer.
  """

def mjr_blitAux(
  index: typing.SupportsInt,
  src: MjrRect,
  left: typing.SupportsInt,
  bottom: typing.SupportsInt,
  con: MjrContext,
) -> None:
  """
  Blit from Aux buffer to con->currentBuffer.
  """

def mjr_blitBuffer(
  src: MjrRect,
  dst: MjrRect,
  flg_color: typing.SupportsInt,
  flg_depth: typing.SupportsInt,
  con: MjrContext,
) -> None:
  """
  Blit from src viewpoint in current framebuffer to dst viewport in other framebuffer. If src, dst have different size and flg_depth==0, color is interpolated with GL_LINEAR.
  """

def mjr_changeFont(fontscale: typing.SupportsInt, con: MjrContext) -> None:
  """
  Change font of existing context.
  """

def mjr_drawPixels(
  rgb: typing.Annotated[numpy.typing.NDArray[numpy.uint8], "[m, 1]"] | None,
  depth: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[m, 1]"] | None,
  viewport: MjrRect,
  con: MjrContext,
) -> None:
  """
  Draw pixels from client buffer to current OpenGL framebuffer. Viewport is in OpenGL framebuffer; client buffer starts at (0,0).
  """

def mjr_figure(
  viewport: MjrRect, fig: mujoco._structs.MjvFigure, con: MjrContext
) -> None:
  """
  Draw 2D figure.
  """

def mjr_findRect(
  x: typing.SupportsInt, y: typing.SupportsInt, nrect: typing.SupportsInt, rect: MjrRect
) -> int:
  """
  Find first rectangle containing mouse, -1: not found.
  """

def mjr_finish() -> None:
  """
  Call glFinish.
  """

def mjr_getError() -> int:
  """
  Call glGetError and return result.
  """

def mjr_label(
  viewport: MjrRect,
  font: typing.SupportsInt,
  txt: str,
  r: typing.SupportsFloat,
  g: typing.SupportsFloat,
  b: typing.SupportsFloat,
  a: typing.SupportsFloat,
  rt: typing.SupportsFloat,
  gt: typing.SupportsFloat,
  bt: typing.SupportsFloat,
  con: MjrContext,
) -> None:
  """
  Draw rectangle with centered text.
  """

def mjr_maxViewport(con: MjrContext) -> MjrRect:
  """
  Get maximum viewport for active buffer.
  """

def mjr_overlay(
  font: typing.SupportsInt,
  gridpos: typing.SupportsInt,
  viewport: MjrRect,
  overlay: str,
  overlay2: str,
  con: MjrContext,
) -> None:
  """
  Draw text overlay; font is mjtFont; gridpos is mjtGridPos.
  """

def mjr_readPixels(
  rgb: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8] | None,
  depth: typing.Annotated[numpy.typing.ArrayLike, numpy.float32] | None,
  viewport: MjrRect,
  con: MjrContext,
) -> None:
  """
  Read pixels from current OpenGL framebuffer to client buffer. Viewport is in OpenGL framebuffer; client buffer starts at (0,0).
  """

def mjr_rectangle(
  viewport: MjrRect,
  r: typing.SupportsFloat,
  g: typing.SupportsFloat,
  b: typing.SupportsFloat,
  a: typing.SupportsFloat,
) -> None:
  """
  Draw rectangle.
  """

def mjr_render(
  viewport: MjrRect, scn: mujoco._structs.MjvScene, con: MjrContext
) -> None:
  """
  Render 3D scene.
  """

def mjr_resizeOffscreen(
  width: typing.SupportsInt, height: typing.SupportsInt, con: MjrContext
) -> None:
  """
  Resize offscreen buffers.
  """

def mjr_restoreBuffer(con: MjrContext) -> None:
  """
  Make con->currentBuffer current again.
  """

def mjr_setAux(index: typing.SupportsInt, con: MjrContext) -> None:
  """
  Set Aux buffer for custom OpenGL rendering (call restoreBuffer when done).
  """

def mjr_setBuffer(framebuffer: typing.SupportsInt, con: MjrContext) -> None:
  """
  Set OpenGL framebuffer for rendering: mjFB_WINDOW or mjFB_OFFSCREEN. If only one buffer is available, set that buffer and ignore framebuffer argument.
  """

def mjr_text(
  font: typing.SupportsInt,
  txt: str,
  con: MjrContext,
  x: typing.SupportsFloat,
  y: typing.SupportsFloat,
  r: typing.SupportsFloat,
  g: typing.SupportsFloat,
  b: typing.SupportsFloat,
) -> None:
  """
  Draw text at (x,y) in relative coordinates; font is mjtFont.
  """

def mjr_uploadHField(
  m: mujoco._structs.MjModel, con: MjrContext, hfieldid: typing.SupportsInt
) -> None:
  """
  Upload height field to GPU, overwriting previous upload if any.
  """

def mjr_uploadMesh(
  m: mujoco._structs.MjModel, con: MjrContext, meshid: typing.SupportsInt
) -> None:
  """
  Upload mesh to GPU, overwriting previous upload if any.
  """

def mjr_uploadTexture(
  m: mujoco._structs.MjModel, con: MjrContext, texid: typing.SupportsInt
) -> None:
  """
  Upload texture to GPU, overwriting previous upload if any.
  """
