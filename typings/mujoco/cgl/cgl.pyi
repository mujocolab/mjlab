"""
Bindings for Apple CGL.
"""
from __future__ import annotations
import ctypes as ctypes
from ctypes import c_int as GLint
from ctypes import c_void_p as CGLContextObj
from ctypes import c_void_p as CGLPixelFormatObj
import enum as enum
import typing
__all__: list[str] = ['CGLContextObj', 'CGLError', 'CGLOpenGLProfile', 'CGLPixelFormatAttribute', 'CGLPixelFormatObj', 'CGLReleaseContext', 'CGLReleasePixelFormat', 'GLint', 'ctypes', 'enum']
class CGLError(RuntimeError):
    pass
class CGLOpenGLProfile(enum.IntEnum):
    CGLOGLPVersion_3_2_Core: typing.ClassVar[CGLOpenGLProfile]  # value = <CGLOpenGLProfile.CGLOGLPVersion_3_2_Core: 12800>
    CGLOGLPVersion_GL4_Core: typing.ClassVar[CGLOpenGLProfile]  # value = <CGLOpenGLProfile.CGLOGLPVersion_GL4_Core: 16640>
    CGLOGLPVersion_Legacy: typing.ClassVar[CGLOpenGLProfile]  # value = <CGLOpenGLProfile.CGLOGLPVersion_Legacy: 4096>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
class CGLPixelFormatAttribute(enum.IntEnum):
    """
    CGLPixelFormatAttribute enum values.
    """
    CGLPFAAccelerated: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAAccelerated: 73>
    CGLPFAAcceleratedCompute: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAAcceleratedCompute: 97>
    CGLPFAAccumSize: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAAccumSize: 14>
    CGLPFAAllRenderers: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAAllRenderers: 1>
    CGLPFAAllowOfflineRenderers: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAAllowOfflineRenderers: 96>
    CGLPFAAlphaSize: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAAlphaSize: 11>
    CGLPFAAuxBuffers: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAAuxBuffers: 7>
    CGLPFAAuxDepthStencil: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAAuxDepthStencil: 57>
    CGLPFABackingStore: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFABackingStore: 76>
    CGLPFABackingVolatile: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFABackingVolatile: 77>
    CGLPFAClosestPolicy: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAClosestPolicy: 74>
    CGLPFAColorFloat: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAColorFloat: 58>
    CGLPFAColorSize: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAColorSize: 8>
    CGLPFACompliant: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFACompliant: 83>
    CGLPFADepthSize: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFADepthSize: 12>
    CGLPFADisplayMask: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFADisplayMask: 84>
    CGLPFADoubleBuffer: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFADoubleBuffer: 5>
    CGLPFAFullScreen: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAFullScreen: 54>
    CGLPFAMPSafe: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAMPSafe: 78>
    CGLPFAMaximumPolicy: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAMaximumPolicy: 52>
    CGLPFAMinimumPolicy: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAMinimumPolicy: 51>
    CGLPFAMultiScreen: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAMultiScreen: 81>
    CGLPFAMultisample: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAMultisample: 59>
    CGLPFANoRecovery: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFANoRecovery: 72>
    CGLPFAOffScreen: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAOffScreen: 53>
    CGLPFAOpenGLProfile: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAOpenGLProfile: 99>
    CGLPFAPBuffer: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAPBuffer: 90>
    CGLPFARemotePBuffer: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFARemotePBuffer: 91>
    CGLPFARendererID: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFARendererID: 70>
    CGLPFARobust: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFARobust: 75>
    CGLPFASample: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFASample: 56>
    CGLPFASampleAlpha: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFASampleAlpha: 61>
    CGLPFASampleBuffers: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFASampleBuffers: 55>
    CGLPFASingleRenderer: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFASingleRenderer: 71>
    CGLPFAStencilSize: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAStencilSize: 13>
    CGLPFAStereo: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAStereo: 6>
    CGLPFASupersample: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFASupersample: 60>
    CGLPFASupportsAutomaticGraphicsSwitching: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFASupportsAutomaticGraphicsSwitching: 101>
    CGLPFATripleBuffer: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFATripleBuffer: 3>
    CGLPFAVirtualScreenCount: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAVirtualScreenCount: 128>
    CGLPFAWindow: typing.ClassVar[CGLPixelFormatAttribute]  # value = <CGLPixelFormatAttribute.CGLPFAWindow: 80>
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
def _make_checked(func):
    ...
CGLReleaseContext: ctypes.CDLL.__init__.<locals>._FuncPtr  # value = <_FuncPtr object>
CGLReleasePixelFormat: ctypes.CDLL.__init__.<locals>._FuncPtr  # value = <_FuncPtr object>
_CGL: ctypes.CDLL  # value = <CDLL '/System/Library/Frameworks/OpenGL.framework/OpenGL', handle 3e664afc8 at 0x104c2d400>
_CGLChoosePixelFormat: ctypes.CDLL.__init__.<locals>._FuncPtr  # value = <_FuncPtr object>
_CGLCreateContext: ctypes.CDLL.__init__.<locals>._FuncPtr  # value = <_FuncPtr object>
_CGLErrorString: ctypes.CDLL.__init__.<locals>._FuncPtr  # value = <_FuncPtr object>
_CGLLockContext: ctypes.CDLL.__init__.<locals>._FuncPtr  # value = <_FuncPtr object>
_CGLReleaseContext: ctypes.CDLL.__init__.<locals>._FuncPtr  # value = <_FuncPtr object>
_CGLReleasePixelFormat: ctypes.CDLL.__init__.<locals>._FuncPtr  # value = <_FuncPtr object>
_CGLSetCurrentContext: ctypes.CDLL.__init__.<locals>._FuncPtr  # value = <_FuncPtr object>
_CGLUnlockContext: ctypes.CDLL.__init__.<locals>._FuncPtr  # value = <_FuncPtr object>
