"""
Python bindings for MuJoCo.
"""

from __future__ import annotations
import ctypes as ctypes
from mujoco._callbacks import get_mjcb_act_bias
from mujoco._callbacks import get_mjcb_act_dyn
from mujoco._callbacks import get_mjcb_act_gain
from mujoco._callbacks import get_mjcb_contactfilter
from mujoco._callbacks import get_mjcb_control
from mujoco._callbacks import get_mjcb_passive
from mujoco._callbacks import get_mjcb_sensor
from mujoco._callbacks import get_mjcb_time
from mujoco._callbacks import get_mju_user_free
from mujoco._callbacks import get_mju_user_malloc
from mujoco._callbacks import get_mju_user_warning
from mujoco._callbacks import set_mjcb_act_bias
from mujoco._callbacks import set_mjcb_act_dyn
from mujoco._callbacks import set_mjcb_act_gain
from mujoco._callbacks import set_mjcb_contactfilter
from mujoco._callbacks import set_mjcb_control
from mujoco._callbacks import set_mjcb_passive
from mujoco._callbacks import set_mjcb_sensor
from mujoco._callbacks import set_mjcb_time
from mujoco._callbacks import set_mju_user_free
from mujoco._callbacks import set_mju_user_malloc
from mujoco._callbacks import set_mju_user_warning
from mujoco._enums import mjtAlignFree
from mujoco._enums import mjtBias
from mujoco._enums import mjtBuiltin
from mujoco._enums import mjtButton
from mujoco._enums import mjtCamLight
from mujoco._enums import mjtCamera
from mujoco._enums import mjtCatBit
from mujoco._enums import mjtColorSpace
from mujoco._enums import mjtConDataField
from mujoco._enums import mjtCone
from mujoco._enums import mjtConstraint
from mujoco._enums import mjtConstraintState
from mujoco._enums import mjtDataType
from mujoco._enums import mjtDepthMap
from mujoco._enums import mjtDisableBit
from mujoco._enums import mjtDyn
from mujoco._enums import mjtEnableBit
from mujoco._enums import mjtEq
from mujoco._enums import mjtEvent
from mujoco._enums import mjtFlexSelf
from mujoco._enums import mjtFont
from mujoco._enums import mjtFontScale
from mujoco._enums import mjtFrame
from mujoco._enums import mjtFramebuffer
from mujoco._enums import mjtGain
from mujoco._enums import mjtGeom
from mujoco._enums import mjtGeomInertia
from mujoco._enums import mjtGridPos
from mujoco._enums import mjtInertiaFromGeom
from mujoco._enums import mjtIntegrator
from mujoco._enums import mjtItem
from mujoco._enums import mjtJacobian
from mujoco._enums import mjtJoint
from mujoco._enums import mjtLRMode
from mujoco._enums import mjtLabel
from mujoco._enums import mjtLightType
from mujoco._enums import mjtLimited
from mujoco._enums import mjtMark
from mujoco._enums import mjtMeshBuiltin
from mujoco._enums import mjtMeshInertia
from mujoco._enums import mjtMouse
from mujoco._enums import mjtObj
from mujoco._enums import mjtOrientation
from mujoco._enums import mjtPertBit
from mujoco._enums import mjtPluginCapabilityBit
from mujoco._enums import mjtRndFlag
from mujoco._enums import mjtSDFType
from mujoco._enums import mjtSameFrame
from mujoco._enums import mjtSection
from mujoco._enums import mjtSensor
from mujoco._enums import mjtSolver
from mujoco._enums import mjtStage
from mujoco._enums import mjtState
from mujoco._enums import mjtStereo
from mujoco._enums import mjtTaskStatus
from mujoco._enums import mjtTexture
from mujoco._enums import mjtTextureRole
from mujoco._enums import mjtTimer
from mujoco._enums import mjtTrn
from mujoco._enums import mjtVisFlag
from mujoco._enums import mjtWarning
from mujoco._enums import mjtWrap
from mujoco._functions import mj_Euler
from mujoco._functions import mj_RungeKutta
from mujoco._functions import mj_addContact
from mujoco._functions import mj_addM
from mujoco._functions import mj_angmomMat
from mujoco._functions import mj_applyFT
from mujoco._functions import mj_camlight
from mujoco._functions import mj_checkAcc
from mujoco._functions import mj_checkPos
from mujoco._functions import mj_checkVel
from mujoco._functions import mj_collision
from mujoco._functions import mj_comPos
from mujoco._functions import mj_comVel
from mujoco._functions import mj_compareFwdInv
from mujoco._functions import mj_constraintUpdate
from mujoco._functions import mj_contactForce
from mujoco._functions import mj_crb
from mujoco._functions import mj_defaultLROpt
from mujoco._functions import mj_defaultOption
from mujoco._functions import mj_defaultSolRefImp
from mujoco._functions import mj_defaultVisual
from mujoco._functions import mj_differentiatePos
from mujoco._functions import mj_energyPos
from mujoco._functions import mj_energyVel
from mujoco._functions import mj_factorM
from mujoco._functions import mj_flex
from mujoco._functions import mj_forward
from mujoco._functions import mj_forwardSkip
from mujoco._functions import mj_fullM
from mujoco._functions import mj_fwdAcceleration
from mujoco._functions import mj_fwdActuation
from mujoco._functions import mj_fwdConstraint
from mujoco._functions import mj_fwdPosition
from mujoco._functions import mj_fwdVelocity
from mujoco._functions import mj_geomDistance
from mujoco._functions import mj_getState
from mujoco._functions import mj_getTotalmass
from mujoco._functions import mj_id2name
from mujoco._functions import mj_implicit
from mujoco._functions import mj_integratePos
from mujoco._functions import mj_invConstraint
from mujoco._functions import mj_invPosition
from mujoco._functions import mj_invVelocity
from mujoco._functions import mj_inverse
from mujoco._functions import mj_inverseSkip
from mujoco._functions import mj_isDual
from mujoco._functions import mj_isPyramidal
from mujoco._functions import mj_isSparse
from mujoco._functions import mj_island
from mujoco._functions import mj_jac
from mujoco._functions import mj_jacBody
from mujoco._functions import mj_jacBodyCom
from mujoco._functions import mj_jacDot
from mujoco._functions import mj_jacGeom
from mujoco._functions import mj_jacPointAxis
from mujoco._functions import mj_jacSite
from mujoco._functions import mj_jacSubtreeCom
from mujoco._functions import mj_kinematics
from mujoco._functions import mj_loadAllPluginLibraries
from mujoco._functions import mj_loadPluginLibrary
from mujoco._functions import mj_local2Global
from mujoco._functions import mj_makeConstraint
from mujoco._functions import mj_makeM
from mujoco._functions import mj_mulJacTVec
from mujoco._functions import mj_mulJacVec
from mujoco._functions import mj_mulM
from mujoco._functions import mj_mulM2
from mujoco._functions import mj_multiRay
from mujoco._functions import mj_name2id
from mujoco._functions import mj_normalizeQuat
from mujoco._functions import mj_objectAcceleration
from mujoco._functions import mj_objectVelocity
from mujoco._functions import mj_passive
from mujoco._functions import mj_printData
from mujoco._functions import mj_printFormattedData
from mujoco._functions import mj_printFormattedModel
from mujoco._functions import mj_printModel
from mujoco._functions import mj_printSchema
from mujoco._functions import mj_projectConstraint
from mujoco._functions import mj_ray
from mujoco._functions import mj_rayHfield
from mujoco._functions import mj_rayMesh
from mujoco._functions import mj_referenceConstraint
from mujoco._functions import mj_resetCallbacks
from mujoco._functions import mj_resetData
from mujoco._functions import mj_resetDataDebug
from mujoco._functions import mj_resetDataKeyframe
from mujoco._functions import mj_rne
from mujoco._functions import mj_rnePostConstraint
from mujoco._functions import mj_saveLastXML
from mujoco._functions import mj_saveModel
from mujoco._functions import mj_sensorAcc
from mujoco._functions import mj_sensorPos
from mujoco._functions import mj_sensorVel
from mujoco._functions import mj_setConst
from mujoco._functions import mj_setKeyframe
from mujoco._functions import mj_setLengthRange
from mujoco._functions import mj_setState
from mujoco._functions import mj_setTotalmass
from mujoco._functions import mj_sizeModel
from mujoco._functions import mj_solveM
from mujoco._functions import mj_solveM2
from mujoco._functions import mj_stateSize
from mujoco._functions import mj_step
from mujoco._functions import mj_step1
from mujoco._functions import mj_step2
from mujoco._functions import mj_subtreeVel
from mujoco._functions import mj_tendon
from mujoco._functions import mj_transmission
from mujoco._functions import mj_version
from mujoco._functions import mj_versionString
from mujoco._functions import mjd_inverseFD
from mujoco._functions import mjd_quatIntegrate
from mujoco._functions import mjd_subQuat
from mujoco._functions import mjd_transitionFD
from mujoco._functions import mju_Halton
from mujoco._functions import mju_L1
from mujoco._functions import mju_add
from mujoco._functions import mju_add3
from mujoco._functions import mju_addScl
from mujoco._functions import mju_addScl3
from mujoco._functions import mju_addTo
from mujoco._functions import mju_addTo3
from mujoco._functions import mju_addToScl
from mujoco._functions import mju_addToScl3
from mujoco._functions import mju_axisAngle2Quat
from mujoco._functions import mju_band2Dense
from mujoco._functions import mju_bandDiag
from mujoco._functions import mju_bandMulMatVec
from mujoco._functions import mju_boxQP
from mujoco._functions import mju_cholFactor
from mujoco._functions import mju_cholFactorBand
from mujoco._functions import mju_cholSolve
from mujoco._functions import mju_cholSolveBand
from mujoco._functions import mju_cholUpdate
from mujoco._functions import mju_clip
from mujoco._functions import mju_copy
from mujoco._functions import mju_copy3
from mujoco._functions import mju_copy4
from mujoco._functions import mju_cross
from mujoco._functions import mju_d2n
from mujoco._functions import mju_decodePyramid
from mujoco._functions import mju_dense2Band
from mujoco._functions import mju_dense2sparse
from mujoco._functions import mju_derivQuat
from mujoco._functions import mju_dist3
from mujoco._functions import mju_dot
from mujoco._functions import mju_dot3
from mujoco._functions import mju_eig3
from mujoco._functions import mju_encodePyramid
from mujoco._functions import mju_euler2Quat
from mujoco._functions import mju_eye
from mujoco._functions import mju_f2n
from mujoco._functions import mju_fill
from mujoco._functions import mju_insertionSort
from mujoco._functions import mju_insertionSortInt
from mujoco._functions import mju_isBad
from mujoco._functions import mju_isZero
from mujoco._functions import mju_mat2Quat
from mujoco._functions import mju_mat2Rot
from mujoco._functions import mju_max
from mujoco._functions import mju_min
from mujoco._functions import mju_mulMatMat
from mujoco._functions import mju_mulMatMatT
from mujoco._functions import mju_mulMatTMat
from mujoco._functions import mju_mulMatTVec
from mujoco._functions import mju_mulMatTVec3
from mujoco._functions import mju_mulMatVec
from mujoco._functions import mju_mulMatVec3
from mujoco._functions import mju_mulPose
from mujoco._functions import mju_mulQuat
from mujoco._functions import mju_mulQuatAxis
from mujoco._functions import mju_mulVecMatVec
from mujoco._functions import mju_muscleBias
from mujoco._functions import mju_muscleDynamics
from mujoco._functions import mju_muscleGain
from mujoco._functions import mju_n2d
from mujoco._functions import mju_n2f
from mujoco._functions import mju_negPose
from mujoco._functions import mju_negQuat
from mujoco._functions import mju_norm
from mujoco._functions import mju_norm3
from mujoco._functions import mju_normalize
from mujoco._functions import mju_normalize3
from mujoco._functions import mju_normalize4
from mujoco._functions import mju_printMat
from mujoco._functions import mju_printMatSparse
from mujoco._functions import mju_quat2Mat
from mujoco._functions import mju_quat2Vel
from mujoco._functions import mju_quatIntegrate
from mujoco._functions import mju_quatZ2Vec
from mujoco._functions import mju_rayFlex
from mujoco._functions import mju_rayGeom
from mujoco._functions import mju_raySkin
from mujoco._functions import mju_rotVecQuat
from mujoco._functions import mju_round
from mujoco._functions import mju_scl
from mujoco._functions import mju_scl3
from mujoco._functions import mju_sigmoid
from mujoco._functions import mju_sign
from mujoco._functions import mju_sparse2dense
from mujoco._functions import mju_springDamper
from mujoco._functions import mju_sqrMatTD
from mujoco._functions import mju_standardNormal
from mujoco._functions import mju_str2Type
from mujoco._functions import mju_sub
from mujoco._functions import mju_sub3
from mujoco._functions import mju_subFrom
from mujoco._functions import mju_subFrom3
from mujoco._functions import mju_subQuat
from mujoco._functions import mju_sum
from mujoco._functions import mju_symmetrize
from mujoco._functions import mju_transformSpatial
from mujoco._functions import mju_transpose
from mujoco._functions import mju_trnVecPose
from mujoco._functions import mju_type2Str
from mujoco._functions import mju_unit4
from mujoco._functions import mju_warningText
from mujoco._functions import mju_writeLog
from mujoco._functions import mju_writeNumBytes
from mujoco._functions import mju_zero
from mujoco._functions import mju_zero3
from mujoco._functions import mju_zero4
from mujoco._functions import mjv_addGeoms
from mujoco._functions import mjv_alignToCamera
from mujoco._functions import mjv_applyPerturbForce
from mujoco._functions import mjv_applyPerturbPose
from mujoco._functions import mjv_cameraInModel
from mujoco._functions import mjv_cameraInRoom
from mujoco._functions import mjv_connector
from mujoco._functions import mjv_defaultCamera
from mujoco._functions import mjv_defaultFigure
from mujoco._functions import mjv_defaultFreeCamera
from mujoco._functions import mjv_defaultOption
from mujoco._functions import mjv_defaultPerturb
from mujoco._functions import mjv_frustumHeight
from mujoco._functions import mjv_initGeom
from mujoco._functions import mjv_initPerturb
from mujoco._functions import mjv_makeLights
from mujoco._functions import mjv_model2room
from mujoco._functions import mjv_moveCamera
from mujoco._functions import mjv_moveModel
from mujoco._functions import mjv_movePerturb
from mujoco._functions import mjv_room2model
from mujoco._functions import mjv_select
from mujoco._functions import mjv_updateCamera
from mujoco._functions import mjv_updateScene
from mujoco._functions import mjv_updateSkin
from mujoco._render import MjrContext
from mujoco._render import MjrRect
from mujoco._render import mjr_addAux
from mujoco._render import mjr_blitAux
from mujoco._render import mjr_blitBuffer
from mujoco._render import mjr_changeFont
from mujoco._render import mjr_drawPixels
from mujoco._render import mjr_figure
from mujoco._render import mjr_findRect
from mujoco._render import mjr_finish
from mujoco._render import mjr_getError
from mujoco._render import mjr_label
from mujoco._render import mjr_maxViewport
from mujoco._render import mjr_overlay
from mujoco._render import mjr_readPixels
from mujoco._render import mjr_rectangle
from mujoco._render import mjr_render
from mujoco._render import mjr_resizeOffscreen
from mujoco._render import mjr_restoreBuffer
from mujoco._render import mjr_setAux
from mujoco._render import mjr_setBuffer
from mujoco._render import mjr_text
from mujoco._render import mjr_uploadHField
from mujoco._render import mjr_uploadMesh
from mujoco._render import mjr_uploadTexture
from mujoco._specs import MjByteVec
from mujoco._specs import MjCharVec
from mujoco._specs import MjDoubleVec
from mujoco._specs import MjFloatVec
from mujoco._specs import MjIntVec
from mujoco._specs import MjSpec
from mujoco._specs import MjStringVec
from mujoco._specs import MjVisualHeadlight
from mujoco._specs import MjVisualRgba
from mujoco._specs import MjsActuator
from mujoco._specs import MjsBody
from mujoco._specs import MjsCamera
from mujoco._specs import MjsCompiler
from mujoco._specs import MjsDefault
from mujoco._specs import MjsElement
from mujoco._specs import MjsEquality
from mujoco._specs import MjsExclude
from mujoco._specs import MjsFlex
from mujoco._specs import MjsFrame
from mujoco._specs import MjsGeom
from mujoco._specs import MjsHField
from mujoco._specs import MjsJoint
from mujoco._specs import MjsKey
from mujoco._specs import MjsLight
from mujoco._specs import MjsMaterial
from mujoco._specs import MjsMesh
from mujoco._specs import MjsNumeric
from mujoco._specs import MjsOrientation
from mujoco._specs import MjsPair
from mujoco._specs import MjsPlugin
from mujoco._specs import MjsSensor
from mujoco._specs import MjsSite
from mujoco._specs import MjsSkin
from mujoco._specs import MjsTendon
from mujoco._specs import MjsText
from mujoco._specs import MjsTexture
from mujoco._specs import MjsTuple
from mujoco._specs import MjsWrap
from mujoco._structs import MjContact
from mujoco._structs import MjData
from mujoco._structs import MjLROpt
from mujoco._structs import MjModel
from mujoco._structs import MjOption
from mujoco._structs import MjSolverStat
from mujoco._structs import MjStatistic
from mujoco._structs import MjTimerStat
from mujoco._structs import MjVisual
from mujoco._structs import MjWarningStat
from mujoco._structs import MjvCamera
from mujoco._structs import MjvFigure
from mujoco._structs import MjvGLCamera
from mujoco._structs import MjvGeom
from mujoco._structs import MjvLight
from mujoco._structs import MjvOption
from mujoco._structs import MjvPerturb
from mujoco._structs import MjvScene
from mujoco._structs import mjv_averageCamera
from mujoco.glfw import GLContext
from mujoco.renderer import Renderer
import os as os
import platform as platform
import subprocess as subprocess
import typing
from typing import Any
from typing import IO
import warnings as warnings
import zipfile as zipfile
from . import _callbacks
from . import _constants
from . import _enums
from . import _errors
from . import _functions
from . import _render
from . import _specs
from . import _structs
from . import gl_context
from . import glfw
from . import renderer

__all__: list[str] = [
  "Any",
  "FatalError",
  "GLContext",
  "HEADERS_DIR",
  "IO",
  "MjByteVec",
  "MjCharVec",
  "MjContact",
  "MjData",
  "MjDoubleVec",
  "MjFloatVec",
  "MjIntVec",
  "MjLROpt",
  "MjModel",
  "MjOption",
  "MjSolverStat",
  "MjSpec",
  "MjStatistic",
  "MjStringVec",
  "MjStruct",
  "MjTimerStat",
  "MjVisual",
  "MjVisualHeadlight",
  "MjVisualRgba",
  "MjWarningStat",
  "MjrContext",
  "MjrRect",
  "MjsActuator",
  "MjsBody",
  "MjsCamera",
  "MjsCompiler",
  "MjsDefault",
  "MjsElement",
  "MjsEquality",
  "MjsExclude",
  "MjsFlex",
  "MjsFrame",
  "MjsGeom",
  "MjsHField",
  "MjsJoint",
  "MjsKey",
  "MjsLight",
  "MjsMaterial",
  "MjsMesh",
  "MjsNumeric",
  "MjsOrientation",
  "MjsPair",
  "MjsPlugin",
  "MjsSensor",
  "MjsSite",
  "MjsSkin",
  "MjsTendon",
  "MjsText",
  "MjsTexture",
  "MjsTuple",
  "MjsWrap",
  "MjvCamera",
  "MjvFigure",
  "MjvGLCamera",
  "MjvGeom",
  "MjvLight",
  "MjvOption",
  "MjvPerturb",
  "MjvScene",
  "PLUGINS_DIR",
  "PLUGIN_HANDLES",
  "Renderer",
  "UnexpectedError",
  "ctypes",
  "from_zip",
  "get_mjcb_act_bias",
  "get_mjcb_act_dyn",
  "get_mjcb_act_gain",
  "get_mjcb_contactfilter",
  "get_mjcb_control",
  "get_mjcb_passive",
  "get_mjcb_sensor",
  "get_mjcb_time",
  "get_mju_user_free",
  "get_mju_user_malloc",
  "get_mju_user_warning",
  "gl_context",
  "glfw",
  "mjDISABLESTRING",
  "mjENABLESTRING",
  "mjFRAMESTRING",
  "mjLABELSTRING",
  "mjMAXCONPAIR",
  "mjMAXIMP",
  "mjMAXLIGHT",
  "mjMAXLINE",
  "mjMAXLINEPNT",
  "mjMAXOVERLAY",
  "mjMAXPLANEGRID",
  "mjMAXVAL",
  "mjMINIMP",
  "mjMINMU",
  "mjMINVAL",
  "mjNBIAS",
  "mjNDYN",
  "mjNEQDATA",
  "mjNGAIN",
  "mjNGROUP",
  "mjNIMP",
  "mjNREF",
  "mjNSOLVER",
  "mjPI",
  "mjRNDSTRING",
  "mjTIMERSTRING",
  "mjVERSION_HEADER",
  "mjVISSTRING",
  "mj_Euler",
  "mj_RungeKutta",
  "mj_addContact",
  "mj_addM",
  "mj_angmomMat",
  "mj_applyFT",
  "mj_camlight",
  "mj_checkAcc",
  "mj_checkPos",
  "mj_checkVel",
  "mj_collision",
  "mj_comPos",
  "mj_comVel",
  "mj_compareFwdInv",
  "mj_constraintUpdate",
  "mj_contactForce",
  "mj_crb",
  "mj_defaultLROpt",
  "mj_defaultOption",
  "mj_defaultSolRefImp",
  "mj_defaultVisual",
  "mj_differentiatePos",
  "mj_energyPos",
  "mj_energyVel",
  "mj_factorM",
  "mj_flex",
  "mj_forward",
  "mj_forwardSkip",
  "mj_fullM",
  "mj_fwdAcceleration",
  "mj_fwdActuation",
  "mj_fwdConstraint",
  "mj_fwdPosition",
  "mj_fwdVelocity",
  "mj_geomDistance",
  "mj_getState",
  "mj_getTotalmass",
  "mj_id2name",
  "mj_implicit",
  "mj_integratePos",
  "mj_invConstraint",
  "mj_invPosition",
  "mj_invVelocity",
  "mj_inverse",
  "mj_inverseSkip",
  "mj_isDual",
  "mj_isPyramidal",
  "mj_isSparse",
  "mj_island",
  "mj_jac",
  "mj_jacBody",
  "mj_jacBodyCom",
  "mj_jacDot",
  "mj_jacGeom",
  "mj_jacPointAxis",
  "mj_jacSite",
  "mj_jacSubtreeCom",
  "mj_kinematics",
  "mj_loadAllPluginLibraries",
  "mj_loadPluginLibrary",
  "mj_local2Global",
  "mj_makeConstraint",
  "mj_makeM",
  "mj_mulJacTVec",
  "mj_mulJacVec",
  "mj_mulM",
  "mj_mulM2",
  "mj_multiRay",
  "mj_name2id",
  "mj_normalizeQuat",
  "mj_objectAcceleration",
  "mj_objectVelocity",
  "mj_passive",
  "mj_printData",
  "mj_printFormattedData",
  "mj_printFormattedModel",
  "mj_printModel",
  "mj_printSchema",
  "mj_projectConstraint",
  "mj_ray",
  "mj_rayHfield",
  "mj_rayMesh",
  "mj_referenceConstraint",
  "mj_resetCallbacks",
  "mj_resetData",
  "mj_resetDataDebug",
  "mj_resetDataKeyframe",
  "mj_rne",
  "mj_rnePostConstraint",
  "mj_saveLastXML",
  "mj_saveModel",
  "mj_sensorAcc",
  "mj_sensorPos",
  "mj_sensorVel",
  "mj_setConst",
  "mj_setKeyframe",
  "mj_setLengthRange",
  "mj_setState",
  "mj_setTotalmass",
  "mj_sizeModel",
  "mj_solveM",
  "mj_solveM2",
  "mj_stateSize",
  "mj_step",
  "mj_step1",
  "mj_step2",
  "mj_subtreeVel",
  "mj_tendon",
  "mj_transmission",
  "mj_version",
  "mj_versionString",
  "mjd_inverseFD",
  "mjd_quatIntegrate",
  "mjd_subQuat",
  "mjd_transitionFD",
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
  "mjtAlignFree",
  "mjtBias",
  "mjtBuiltin",
  "mjtButton",
  "mjtCamLight",
  "mjtCamera",
  "mjtCatBit",
  "mjtColorSpace",
  "mjtConDataField",
  "mjtCone",
  "mjtConstraint",
  "mjtConstraintState",
  "mjtDataType",
  "mjtDepthMap",
  "mjtDisableBit",
  "mjtDyn",
  "mjtEnableBit",
  "mjtEq",
  "mjtEvent",
  "mjtFlexSelf",
  "mjtFont",
  "mjtFontScale",
  "mjtFrame",
  "mjtFramebuffer",
  "mjtGain",
  "mjtGeom",
  "mjtGeomInertia",
  "mjtGridPos",
  "mjtInertiaFromGeom",
  "mjtIntegrator",
  "mjtItem",
  "mjtJacobian",
  "mjtJoint",
  "mjtLRMode",
  "mjtLabel",
  "mjtLightType",
  "mjtLimited",
  "mjtMark",
  "mjtMeshBuiltin",
  "mjtMeshInertia",
  "mjtMouse",
  "mjtObj",
  "mjtOrientation",
  "mjtPertBit",
  "mjtPluginCapabilityBit",
  "mjtRndFlag",
  "mjtSDFType",
  "mjtSameFrame",
  "mjtSection",
  "mjtSensor",
  "mjtSolver",
  "mjtStage",
  "mjtState",
  "mjtStereo",
  "mjtTaskStatus",
  "mjtTexture",
  "mjtTextureRole",
  "mjtTimer",
  "mjtTrn",
  "mjtVisFlag",
  "mjtWarning",
  "mjtWrap",
  "mju_Halton",
  "mju_L1",
  "mju_add",
  "mju_add3",
  "mju_addScl",
  "mju_addScl3",
  "mju_addTo",
  "mju_addTo3",
  "mju_addToScl",
  "mju_addToScl3",
  "mju_axisAngle2Quat",
  "mju_band2Dense",
  "mju_bandDiag",
  "mju_bandMulMatVec",
  "mju_boxQP",
  "mju_cholFactor",
  "mju_cholFactorBand",
  "mju_cholSolve",
  "mju_cholSolveBand",
  "mju_cholUpdate",
  "mju_clip",
  "mju_copy",
  "mju_copy3",
  "mju_copy4",
  "mju_cross",
  "mju_d2n",
  "mju_decodePyramid",
  "mju_dense2Band",
  "mju_dense2sparse",
  "mju_derivQuat",
  "mju_dist3",
  "mju_dot",
  "mju_dot3",
  "mju_eig3",
  "mju_encodePyramid",
  "mju_euler2Quat",
  "mju_eye",
  "mju_f2n",
  "mju_fill",
  "mju_insertionSort",
  "mju_insertionSortInt",
  "mju_isBad",
  "mju_isZero",
  "mju_mat2Quat",
  "mju_mat2Rot",
  "mju_max",
  "mju_min",
  "mju_mulMatMat",
  "mju_mulMatMatT",
  "mju_mulMatTMat",
  "mju_mulMatTVec",
  "mju_mulMatTVec3",
  "mju_mulMatVec",
  "mju_mulMatVec3",
  "mju_mulPose",
  "mju_mulQuat",
  "mju_mulQuatAxis",
  "mju_mulVecMatVec",
  "mju_muscleBias",
  "mju_muscleDynamics",
  "mju_muscleGain",
  "mju_n2d",
  "mju_n2f",
  "mju_negPose",
  "mju_negQuat",
  "mju_norm",
  "mju_norm3",
  "mju_normalize",
  "mju_normalize3",
  "mju_normalize4",
  "mju_printMat",
  "mju_printMatSparse",
  "mju_quat2Mat",
  "mju_quat2Vel",
  "mju_quatIntegrate",
  "mju_quatZ2Vec",
  "mju_rayFlex",
  "mju_rayGeom",
  "mju_raySkin",
  "mju_rotVecQuat",
  "mju_round",
  "mju_scl",
  "mju_scl3",
  "mju_sigmoid",
  "mju_sign",
  "mju_sparse2dense",
  "mju_springDamper",
  "mju_sqrMatTD",
  "mju_standardNormal",
  "mju_str2Type",
  "mju_sub",
  "mju_sub3",
  "mju_subFrom",
  "mju_subFrom3",
  "mju_subQuat",
  "mju_sum",
  "mju_symmetrize",
  "mju_transformSpatial",
  "mju_transpose",
  "mju_trnVecPose",
  "mju_type2Str",
  "mju_unit4",
  "mju_warningText",
  "mju_writeLog",
  "mju_writeNumBytes",
  "mju_zero",
  "mju_zero3",
  "mju_zero4",
  "mjv_addGeoms",
  "mjv_alignToCamera",
  "mjv_applyPerturbForce",
  "mjv_applyPerturbPose",
  "mjv_averageCamera",
  "mjv_cameraInModel",
  "mjv_cameraInRoom",
  "mjv_connector",
  "mjv_defaultCamera",
  "mjv_defaultFigure",
  "mjv_defaultFreeCamera",
  "mjv_defaultOption",
  "mjv_defaultPerturb",
  "mjv_frustumHeight",
  "mjv_initGeom",
  "mjv_initPerturb",
  "mjv_makeLights",
  "mjv_model2room",
  "mjv_moveCamera",
  "mjv_moveModel",
  "mjv_movePerturb",
  "mjv_room2model",
  "mjv_select",
  "mjv_updateCamera",
  "mjv_updateScene",
  "mjv_updateSkin",
  "os",
  "platform",
  "renderer",
  "set_mjcb_act_bias",
  "set_mjcb_act_dyn",
  "set_mjcb_act_gain",
  "set_mjcb_contactfilter",
  "set_mjcb_control",
  "set_mjcb_passive",
  "set_mjcb_sensor",
  "set_mjcb_time",
  "set_mju_user_free",
  "set_mju_user_malloc",
  "set_mju_user_warning",
  "subprocess",
  "to_zip",
  "warnings",
  "zipfile",
]

class FatalError(Exception):
  pass

class UnexpectedError(Exception):
  pass

class _MjBindData:
  def __getattr__(self, key: str): ...
  def __init__(self, elements: typing.Sequence[typing.Any]): ...

class _MjBindModel:
  def __getattr__(self, key: str): ...
  def __init__(self, elements: typing.Sequence[typing.Any]): ...

def _bind_data(
  data: _structs.MjData,
  specs: typing.Union[
    typing.Sequence[
      typing.Union[
        mujoco._specs.MjsBody,
        mujoco._specs.MjsFrame,
        mujoco._specs.MjsGeom,
        mujoco._specs.MjsJoint,
        mujoco._specs.MjsLight,
        mujoco._specs.MjsMaterial,
        mujoco._specs.MjsSite,
        mujoco._specs.MjsMesh,
        mujoco._specs.MjsSkin,
        mujoco._specs.MjsTexture,
        mujoco._specs.MjsText,
        mujoco._specs.MjsTuple,
        mujoco._specs.MjsCamera,
        mujoco._specs.MjsFlex,
        mujoco._specs.MjsHField,
        mujoco._specs.MjsKey,
        mujoco._specs.MjsNumeric,
        mujoco._specs.MjsPair,
        mujoco._specs.MjsExclude,
        mujoco._specs.MjsEquality,
        mujoco._specs.MjsTendon,
        mujoco._specs.MjsSensor,
        mujoco._specs.MjsActuator,
        mujoco._specs.MjsPlugin,
      ]
    ],
    mujoco._specs.MjsBody,
    mujoco._specs.MjsFrame,
    mujoco._specs.MjsGeom,
    mujoco._specs.MjsJoint,
    mujoco._specs.MjsLight,
    mujoco._specs.MjsMaterial,
    mujoco._specs.MjsSite,
    mujoco._specs.MjsMesh,
    mujoco._specs.MjsSkin,
    mujoco._specs.MjsTexture,
    mujoco._specs.MjsText,
    mujoco._specs.MjsTuple,
    mujoco._specs.MjsCamera,
    mujoco._specs.MjsFlex,
    mujoco._specs.MjsHField,
    mujoco._specs.MjsKey,
    mujoco._specs.MjsNumeric,
    mujoco._specs.MjsPair,
    mujoco._specs.MjsExclude,
    mujoco._specs.MjsEquality,
    mujoco._specs.MjsTendon,
    mujoco._specs.MjsSensor,
    mujoco._specs.MjsActuator,
    mujoco._specs.MjsPlugin,
  ],
):
  """
  Bind a Mujoco spec to a mjData.

  Args:
    data: The mjData to bind to.
    specs: The mjSpec elements to use for binding, can be a single element or a
      sequence.
  Returns:
    A MjDataGroupedViews object or a list of the same type.
  """

def _bind_model(
  model: _structs.MjModel,
  specs: typing.Union[
    typing.Sequence[
      typing.Union[
        mujoco._specs.MjsBody,
        mujoco._specs.MjsFrame,
        mujoco._specs.MjsGeom,
        mujoco._specs.MjsJoint,
        mujoco._specs.MjsLight,
        mujoco._specs.MjsMaterial,
        mujoco._specs.MjsSite,
        mujoco._specs.MjsMesh,
        mujoco._specs.MjsSkin,
        mujoco._specs.MjsTexture,
        mujoco._specs.MjsText,
        mujoco._specs.MjsTuple,
        mujoco._specs.MjsCamera,
        mujoco._specs.MjsFlex,
        mujoco._specs.MjsHField,
        mujoco._specs.MjsKey,
        mujoco._specs.MjsNumeric,
        mujoco._specs.MjsPair,
        mujoco._specs.MjsExclude,
        mujoco._specs.MjsEquality,
        mujoco._specs.MjsTendon,
        mujoco._specs.MjsSensor,
        mujoco._specs.MjsActuator,
        mujoco._specs.MjsPlugin,
      ]
    ],
    mujoco._specs.MjsBody,
    mujoco._specs.MjsFrame,
    mujoco._specs.MjsGeom,
    mujoco._specs.MjsJoint,
    mujoco._specs.MjsLight,
    mujoco._specs.MjsMaterial,
    mujoco._specs.MjsSite,
    mujoco._specs.MjsMesh,
    mujoco._specs.MjsSkin,
    mujoco._specs.MjsTexture,
    mujoco._specs.MjsText,
    mujoco._specs.MjsTuple,
    mujoco._specs.MjsCamera,
    mujoco._specs.MjsFlex,
    mujoco._specs.MjsHField,
    mujoco._specs.MjsKey,
    mujoco._specs.MjsNumeric,
    mujoco._specs.MjsPair,
    mujoco._specs.MjsExclude,
    mujoco._specs.MjsEquality,
    mujoco._specs.MjsTendon,
    mujoco._specs.MjsSensor,
    mujoco._specs.MjsActuator,
    mujoco._specs.MjsPlugin,
  ],
):
  """
  Bind a Mujoco spec to a mjModel.

  Args:
    model: The mjModel to bind to.
    specs: The mjSpec elements to use for binding, can be a single element or a
      sequence.
  Returns:
    A MjModelGroupedViews object or a list of the same type.
  """

def _load_all_bundled_plugins(): ...
def from_zip(file: typing.Union[str, typing.IO[bytes]]) -> _specs.MjSpec:
  """
  Reads a zip file and returns an MjSpec.

  Args:
    file: The path to the file to read from or the file object to read from.
  Returns:
    An MjSpec object.
  """

def to_zip(spec: _specs.MjSpec, file: typing.Union[str, typing.IO[bytes]]) -> None:
  """
  Converts an MjSpec to a zip file.

  Args:
    spec: The mjSpec to save to a file.
    file: The path to the file to save to or the file object to write to.
  """

HEADERS_DIR: str = (
  "/home/brent/mjlab/.venv/lib/python3.13/site-packages/mujoco/include/mujoco"
)
MjStruct: typing._UnionGenericAlias  # value = typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin]
PLUGINS_DIR: str = "/home/brent/mjlab/.venv/lib/python3.13/site-packages/mujoco/plugin"
PLUGIN_HANDLES: list  # value = [<CDLL '/home/brent/mjlab/.venv/lib/python3.13/site-packages/mujoco/plugin/libactuator.so', handle bb46f40 at 0x78d8333a6490>, <CDLL '/home/brent/mjlab/.venv/lib/python3.13/site-packages/mujoco/plugin/libelasticity.so', handle bb47c40 at 0x78d82f76d950>, <CDLL '/home/brent/mjlab/.venv/lib/python3.13/site-packages/mujoco/plugin/libsdf_plugin.so', handle bd943f0 at 0x78d82f76da90>, <CDLL '/home/brent/mjlab/.venv/lib/python3.13/site-packages/mujoco/plugin/libsensor.so', handle bd94d80 at 0x78d82f76dbd0>]
_SYSTEM: str = "Linux"
__version__: str = "3.3.6"
mjDISABLESTRING: tuple = (
  "Constraint",
  "Equality",
  "Frictionloss",
  "Limit",
  "Contact",
  "Passive",
  "Gravity",
  "Clampctrl",
  "Warmstart",
  "Filterparent",
  "Actuation",
  "Refsafe",
  "Sensor",
  "Midphase",
  "Eulerdamp",
  "AutoReset",
  "NativeCCD",
)
mjENABLESTRING: tuple = (
  "Override",
  "Energy",
  "Fwdinv",
  "InvDiscrete",
  "MultiCCD",
  "Island",
)
mjFRAMESTRING: tuple = (
  "None",
  "Body",
  "Geom",
  "Site",
  "Camera",
  "Light",
  "Contact",
  "World",
)
mjLABELSTRING: tuple = (
  "None",
  "Body",
  "Joint",
  "Geom",
  "Site",
  "Camera",
  "Light",
  "Tendon",
  "Actuator",
  "Constraint",
  "Flex",
  "Skin",
  "Selection",
  "SelPoint",
  "Contact",
  "ContactForce",
  "Island",
)
mjMAXCONPAIR: int = 50
mjMAXIMP: float = 0.9999
mjMAXLIGHT: int = 100
mjMAXLINE: int = 100
mjMAXLINEPNT: int = 1001
mjMAXOVERLAY: int = 500
mjMAXPLANEGRID: int = 200
mjMAXVAL: float = 10000000000.0
mjMINIMP: float = 0.0001
mjMINMU: float = 1e-05
mjMINVAL: float = 1e-15
mjNBIAS: int = 10
mjNDYN: int = 10
mjNEQDATA: int = 11
mjNGAIN: int = 10
mjNGROUP: int = 6
mjNIMP: int = 5
mjNREF: int = 2
mjNSOLVER: int = 200
mjPI: float = 3.141592653589793
mjRNDSTRING: tuple = (
  ("Shadow", "1", "S"),
  ("Wireframe", "0", "W"),
  ("Reflection", "1", "R"),
  ("Additive", "0", "L"),
  ("Skybox", "1", "K"),
  ("Fog", "0", "G"),
  ("Haze", "1", "/"),
  ("Segment", "0", ","),
  ("Id Color", "0", ""),
  ("Cull Face", "1", ""),
)
mjTIMERSTRING: tuple = (
  "step",
  "forward",
  "inverse",
  "position",
  "velocity",
  "actuation",
  "constraint",
  "advance",
  "pos_kinematics",
  "pos_inertia",
  "pos_collision",
  "pos_make",
  "pos_project",
  "col_broadphase",
  "col_narrowphase",
)
mjVERSION_HEADER: int = 336
mjVISSTRING: tuple = (
  ("Convex Hull", "0", "H"),
  ("Texture", "1", "X"),
  ("Joint", "0", "J"),
  ("Camera", "0", "Q"),
  ("Actuator", "0", "U"),
  ("Activation", "0", ","),
  ("Light", "0", "Z"),
  ("Tendon", "1", "V"),
  ("Range Finder", "1", "Y"),
  ("Equality", "0", "E"),
  ("Inertia", "0", "I"),
  ("Scale Inertia", "0", "'"),
  ("Perturb Force", "0", "B"),
  ("Perturb Object", "1", "O"),
  ("Contact Point", "0", "C"),
  ("Island", "0", "N"),
  ("Contact Force", "0", "F"),
  ("Contact Split", "0", "P"),
  ("Transparent", "0", "T"),
  ("Auto Connect", "0", "A"),
  ("Center of Mass", "0", "M"),
  ("Select Point", "0", ""),
  ("Static Body", "1", "D"),
  ("Skin", "1", ";"),
  ("Flex Vert", "0", ""),
  ("Flex Edge", "1", ""),
  ("Flex Face", "0", ""),
  ("Flex Skin", "1", ""),
  ("Body Tree", "0", "`"),
  ("Mesh Tree", "0", "\\"),
  ("SDF iters", "0", ""),
)
