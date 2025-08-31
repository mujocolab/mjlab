from __future__ import annotations
import mujoco._structs
import numpy
import numpy.typing
import typing

__all__: list[str] = [
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
]

def _realloc_con_efc(
  d: mujoco._structs.MjData,
  ncon: typing.SupportsInt,
  nefc: typing.SupportsInt,
  nJ: typing.SupportsInt = -1,
) -> None: ...
def mj_Euler(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Euler integrator, semi-implicit in velocity.
  """

def mj_RungeKutta(
  m: mujoco._structs.MjModel, d: mujoco._structs.MjData, N: typing.SupportsInt
) -> None:
  """
  Runge-Kutta explicit order-N integrator.
  """

def mj_addContact(
  m: mujoco._structs.MjModel, d: mujoco._structs.MjData, con: mujoco._structs.MjContact
) -> int:
  """
  Add contact to d->contact list; return 0 if success; 1 if buffer full.
  """

def mj_addM(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  dst: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  rownnz: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[m, 1]", "flags.writeable"
  ],
  rowadr: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[m, 1]", "flags.writeable"
  ],
  colind: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[m, 1]", "flags.writeable"
  ],
) -> None:
  """
  Add inertia matrix to destination matrix. Destination can be sparse or dense when all int* are NULL.
  """

def mj_angmomMat(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  body: typing.SupportsInt,
) -> None:
  """
  Compute subtree angular momentum matrix.
  """

def mj_applyFT(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  force: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  torque: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  point: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  body: typing.SupportsInt,
  qfrc_target: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
) -> None:
  """
  Apply Cartesian force and torque (outside xfrc_applied mechanism).
  """

def mj_camlight(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Compute camera and light positions and orientations.
  """

def mj_checkAcc(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Check qacc, reset if any element is too big or nan.
  """

def mj_checkPos(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Check qpos, reset if any element is too big or nan.
  """

def mj_checkVel(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Check qvel, reset if any element is too big or nan.
  """

def mj_collision(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Run collision detection.
  """

def mj_comPos(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Map inertias and motion dofs to global frame centered at CoM.
  """

def mj_comVel(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Compute cvel, cdof_dot.
  """

def mj_compareFwdInv(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Compare forward and inverse dynamics, save results in fwdinv.
  """

def mj_constraintUpdate(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  jar: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  cost: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[1, 1]", "flags.writeable"
  ]
  | None,
  flg_coneHessian: typing.SupportsInt,
) -> None:
  """
  Compute efc_state, efc_force, qfrc_constraint, and (optionally) cone Hessians. If cost is not NULL, set *cost = s(jar) where jar = Jac*qacc-aref.
  """

def mj_contactForce(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  id: typing.SupportsInt,
  result: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[6, 1]", "flags.writeable"
  ],
) -> None:
  """
  Extract 6D force:torque given contact id, in the contact frame.
  """

def mj_crb(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Run composite rigid body inertia algorithm (CRB).
  """

def mj_defaultLROpt(opt: mujoco._structs.MjLROpt) -> None:
  """
  Set default options for length range computation.
  """

def mj_defaultOption(opt: mujoco._structs.MjOption) -> None:
  """
  Set physics options to default values.
  """

def mj_defaultSolRefImp(
  solref: typing.SupportsFloat, solimp: typing.SupportsFloat
) -> None:
  """
  Set solver parameters to default values.
  """

def mj_defaultVisual(vis: mujoco._structs.MjVisual) -> None:
  """
  Set visual options to default values.
  """

def mj_differentiatePos(
  m: mujoco._structs.MjModel,
  qvel: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  dt: typing.SupportsFloat,
  qpos1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  qpos2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Compute velocity by finite-differencing two positions.
  """

def mj_energyPos(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Evaluate position-dependent energy (potential).
  """

def mj_energyVel(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Evaluate velocity-dependent energy (kinetic).
  """

def mj_factorM(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Compute sparse L'*D*L factorizaton of inertia matrix.
  """

def mj_flex(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Compute flex-related quantities.
  """

def mj_forward(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Forward dynamics: same as mj_step but do not integrate in time.
  """

def mj_forwardSkip(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  skipstage: typing.SupportsInt,
  skipsensor: typing.SupportsInt,
) -> None:
  """
  Forward dynamics with skip; skipstage is mjtStage.
  """

def mj_fullM(
  m: mujoco._structs.MjModel,
  dst: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  M: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Convert sparse inertia matrix M into full (i.e. dense) matrix.
  """

def mj_fwdAcceleration(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Add up all non-constraint forces, compute qacc_smooth.
  """

def mj_fwdActuation(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Compute actuator force qfrc_actuator.
  """

def mj_fwdConstraint(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Run selected constraint solver.
  """

def mj_fwdPosition(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Run position-dependent computations.
  """

def mj_fwdVelocity(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Run velocity-dependent computations.
  """

def mj_geomDistance(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  geom1: typing.SupportsInt,
  geom2: typing.SupportsInt,
  distmax: typing.SupportsFloat,
  fromto: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
) -> float:
  """
  Returns smallest signed distance between two geoms and optionally segment from geom1 to geom2.
  """

def mj_getState(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  state: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  sig: typing.SupportsInt,
) -> None:
  """
  Get state.
  """

def mj_getTotalmass(m: mujoco._structs.MjModel) -> float:
  """
  Sum all body masses.
  """

def mj_id2name(
  m: mujoco._structs.MjModel, type: typing.SupportsInt, id: typing.SupportsInt
) -> str:
  """
  Get name of object with the specified mjtObj type and id, returns NULL if name not found.
  """

def mj_implicit(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Implicit-in-velocity integrators.
  """

def mj_integratePos(
  m: mujoco._structs.MjModel,
  qpos: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  qvel: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  dt: typing.SupportsFloat,
) -> None:
  """
  Integrate position with given velocity.
  """

def mj_invConstraint(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Apply the analytical formula for inverse constraint dynamics.
  """

def mj_invPosition(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Run position-dependent computations in inverse dynamics.
  """

def mj_invVelocity(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Run velocity-dependent computations in inverse dynamics.
  """

def mj_inverse(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Inverse dynamics: qacc must be set before calling.
  """

def mj_inverseSkip(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  skipstage: typing.SupportsInt,
  skipsensor: typing.SupportsInt,
) -> None:
  """
  Inverse dynamics with skip; skipstage is mjtStage.
  """

def mj_isDual(m: mujoco._structs.MjModel) -> int:
  """
  Determine type of solver (PGS is dual, CG and Newton are primal).
  """

def mj_isPyramidal(m: mujoco._structs.MjModel) -> int:
  """
  Determine type of friction cone.
  """

def mj_isSparse(m: mujoco._structs.MjModel) -> int:
  """
  Determine type of constraint Jacobian.
  """

def mj_island(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Find constraint islands.
  """

def mj_jac(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  jacp: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  jacr: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  point: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  body: typing.SupportsInt,
) -> None:
  """
  Compute 3/6-by-nv end-effector Jacobian of global point attached to given body.
  """

def mj_jacBody(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  jacp: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  jacr: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  body: typing.SupportsInt,
) -> None:
  """
  Compute body frame end-effector Jacobian.
  """

def mj_jacBodyCom(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  jacp: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  jacr: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  body: typing.SupportsInt,
) -> None:
  """
  Compute body center-of-mass end-effector Jacobian.
  """

def mj_jacDot(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  jacp: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  jacr: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  point: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  body: typing.SupportsInt,
) -> None:
  """
  Compute 3/6-by-nv Jacobian time derivative of global point attached to given body.
  """

def mj_jacGeom(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  jacp: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  jacr: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  geom: typing.SupportsInt,
) -> None:
  """
  Compute geom end-effector Jacobian.
  """

def mj_jacPointAxis(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  jacPoint: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  jacAxis: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  point: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  axis: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  body: typing.SupportsInt,
) -> None:
  """
  Compute translation end-effector Jacobian of point, and rotation Jacobian of axis.
  """

def mj_jacSite(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  jacp: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  jacr: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  site: typing.SupportsInt,
) -> None:
  """
  Compute site end-effector Jacobian.
  """

def mj_jacSubtreeCom(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  jacp: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  body: typing.SupportsInt,
) -> None:
  """
  Compute subtree center-of-mass end-effector Jacobian.
  """

def mj_kinematics(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Run forward kinematics.
  """

def mj_loadAllPluginLibraries(directory: str) -> None:
  """
  Scan a directory and load all dynamic libraries. Dynamic libraries in the specified directory are assumed to register one or more plugins. Optionally, if a callback is specified, it is called for each dynamic library encountered that registers plugins.
  """

def mj_loadPluginLibrary(path: str) -> None:
  """
  Load a dynamic library. The dynamic library is assumed to register one or more plugins.
  """

def mj_local2Global(
  d: mujoco._structs.MjData,
  xpos: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  xmat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[9, 1]", "flags.writeable"
  ],
  pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
  body: typing.SupportsInt,
  sameframe: typing.SupportsInt,
) -> None:
  """
  Map from body local to global Cartesian coordinates, sameframe takes values from mjtSameFrame.
  """

def mj_makeConstraint(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Construct constraints.
  """

def mj_makeM(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Make inertia matrix.
  """

def mj_mulJacTVec(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Multiply dense or sparse constraint Jacobian transpose by vector.
  """

def mj_mulJacVec(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Multiply dense or sparse constraint Jacobian by vector.
  """

def mj_mulM(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Multiply vector by inertia matrix.
  """

def mj_mulM2(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Multiply vector by (inertia matrix)^(1/2).
  """

def mj_multiRay(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  pnt: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  geomgroup: typing.Annotated[numpy.typing.NDArray[numpy.uint8], "[6, 1]"] | None,
  flg_static: typing.SupportsInt,
  bodyexclude: typing.SupportsInt,
  geomid: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[m, 1]", "flags.writeable"
  ],
  dist: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  nray: typing.SupportsInt,
  cutoff: typing.SupportsFloat,
) -> None:
  """
  Intersect multiple rays emanating from a single point. Similar semantics to mj_ray, but vec is an array of (nray x 3) directions.
  """

def mj_name2id(m: mujoco._structs.MjModel, type: typing.SupportsInt, name: str) -> int:
  """
  Get id of object with the specified mjtObj type and name, returns -1 if id not found.
  """

def mj_normalizeQuat(
  m: mujoco._structs.MjModel,
  qpos: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
) -> None:
  """
  Normalize all quaternions in qpos-type vector.
  """

def mj_objectAcceleration(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  objtype: typing.SupportsInt,
  objid: typing.SupportsInt,
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[6, 1]", "flags.writeable"
  ],
  flg_local: typing.SupportsInt,
) -> None:
  """
  Compute object 6D acceleration (rot:lin) in object-centered frame, world/local orientation.
  """

def mj_objectVelocity(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  objtype: typing.SupportsInt,
  objid: typing.SupportsInt,
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[6, 1]", "flags.writeable"
  ],
  flg_local: typing.SupportsInt,
) -> None:
  """
  Compute object 6D velocity (rot:lin) in object-centered frame, world/local orientation.
  """

def mj_passive(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Compute qfrc_passive from spring-dampers, gravity compensation and fluid forces.
  """

def mj_printData(
  m: mujoco._structs.MjModel, d: mujoco._structs.MjData, filename: str
) -> None:
  """
  Print data to text file.
  """

def mj_printFormattedData(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  filename: str,
  float_format: str,
) -> None:
  """
  Print mjData to text file, specifying format. float_format must be a valid printf-style format string for a single float value
  """

def mj_printFormattedModel(
  m: mujoco._structs.MjModel, filename: str, float_format: str
) -> None:
  """
  Print mjModel to text file, specifying format. float_format must be a valid printf-style format string for a single float value.
  """

def mj_printModel(m: mujoco._structs.MjModel, filename: str) -> None:
  """
  Print model to text file.
  """

def mj_printSchema(flg_html: bool, flg_pad: bool) -> str:
  """
  Print internal XML schema as plain text or HTML, with style-padding or &nbsp;.
  """

def mj_projectConstraint(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Compute inverse constraint inertia efc_AR.
  """

def mj_ray(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  pnt: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  geomgroup: typing.Annotated[numpy.typing.NDArray[numpy.uint8], "[6, 1]"] | None,
  flg_static: typing.SupportsInt,
  bodyexclude: typing.SupportsInt,
  geomid: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[1, 1]", "flags.writeable"
  ],
) -> float:
  """
  Intersect ray (pnt+x*vec, x>=0) with visible geoms, except geoms in bodyexclude. Return distance (x) to nearest surface, or -1 if no intersection and output geomid. geomgroup, flg_static are as in mjvOption; geomgroup==NULL skips group exclusion.
  """

def mj_rayHfield(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  geomid: typing.SupportsInt,
  pnt: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
  """
  Intersect ray with hfield, return nearest distance or -1 if no intersection.
  """

def mj_rayMesh(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  geomid: typing.SupportsInt,
  pnt: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
  """
  Intersect ray with mesh, return nearest distance or -1 if no intersection.
  """

def mj_referenceConstraint(
  m: mujoco._structs.MjModel, d: mujoco._structs.MjData
) -> None:
  """
  Compute efc_vel, efc_aref.
  """

def mj_resetCallbacks() -> None:
  """
  Reset all callbacks to NULL pointers (NULL is the default).
  """

def mj_resetData(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Reset data to defaults.
  """

def mj_resetDataDebug(
  m: mujoco._structs.MjModel, d: mujoco._structs.MjData, debug_value: typing.SupportsInt
) -> None:
  """
  Reset data to defaults, fill everything else with debug_value.
  """

def mj_resetDataKeyframe(
  m: mujoco._structs.MjModel, d: mujoco._structs.MjData, key: typing.SupportsInt
) -> None:
  """
  Reset data. If 0 <= key < nkey, set fields from specified keyframe.
  """

def mj_rne(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  flg_acc: typing.SupportsInt,
  result: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
) -> None:
  """
  RNE: compute M(qpos)*qacc + C(qpos,qvel); flg_acc=0 removes inertial term.
  """

def mj_rnePostConstraint(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  RNE with complete data: compute cacc, cfrc_ext, cfrc_int.
  """

def mj_saveLastXML(filename: str, m: mujoco._structs.MjModel) -> None:
  """
  Update XML data structures with info from low-level model created with mj_loadXML, save as MJCF. If error is not NULL, it must have size error_sz.
  """

def mj_saveModel(
  m: mujoco._structs.MjModel,
  filename: str | None = None,
  buffer: typing.Annotated[
    numpy.typing.NDArray[numpy.uint8], "[m, 1]", "flags.writeable"
  ]
  | None = None,
) -> None:
  """
  Save model to binary MJB file or memory buffer; buffer has precedence when given.
  """

def mj_sensorAcc(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Evaluate acceleration and force-dependent sensors.
  """

def mj_sensorPos(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Evaluate position-dependent sensors.
  """

def mj_sensorVel(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Evaluate velocity-dependent sensors.
  """

def mj_setConst(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Set constant fields of mjModel, corresponding to qpos0 configuration.
  """

def mj_setKeyframe(
  m: mujoco._structs.MjModel, d: mujoco._structs.MjData, k: typing.SupportsInt
) -> None:
  """
  Copy current state to the k-th model keyframe.
  """

def mj_setLengthRange(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  index: typing.SupportsInt,
  opt: mujoco._structs.MjLROpt,
) -> None:
  """
  Set actuator_lengthrange for specified actuator; return 1 if ok, 0 if error.
  """

def mj_setState(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  state: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  sig: typing.SupportsInt,
) -> None:
  """
  Set state.
  """

def mj_setTotalmass(m: mujoco._structs.MjModel, newmass: typing.SupportsFloat) -> None:
  """
  Scale body masses and inertias to achieve specified total mass.
  """

def mj_sizeModel(m: mujoco._structs.MjModel) -> int:
  """
  Return size of buffer needed to hold model.
  """

def mj_solveM(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  x: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  y: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
) -> None:
  """
  Solve linear system M * x = y using factorization:  x = inv(L'*D*L)*y
  """

def mj_solveM2(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  x: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  y: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  sqrtInvD: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
) -> None:
  """
  Half of linear solve:  x = sqrt(inv(D))*inv(L')*y
  """

def mj_stateSize(m: mujoco._structs.MjModel, sig: typing.SupportsInt) -> int:
  """
  Return size of state signature.
  """

def mj_step(
  m: mujoco._structs.MjModel, d: mujoco._structs.MjData, nstep: typing.SupportsInt = 1
) -> None:
  """
  Advance simulation, use control callback to obtain external force and control. Optionally, repeat nstep times.
  """

def mj_step1(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Advance simulation in two steps: before external force and control is set by user.
  """

def mj_step2(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Advance simulation in two steps: after external force and control is set by user.
  """

def mj_subtreeVel(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Sub-tree linear velocity and angular momentum: compute subtree_linvel, subtree_angmom.
  """

def mj_tendon(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Compute tendon lengths, velocities and moment arms.
  """

def mj_transmission(m: mujoco._structs.MjModel, d: mujoco._structs.MjData) -> None:
  """
  Compute actuator transmission lengths and moments.
  """

def mj_version() -> int:
  """
  Return version number: 1.0.2 is encoded as 102.
  """

def mj_versionString() -> str:
  """
  Return the current version of MuJoCo as a null-terminated string.
  """

def mjd_inverseFD(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  eps: typing.SupportsFloat,
  flg_actuation: typing.SupportsInt,
  DfDq: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  DfDv: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  DfDa: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  DsDq: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  DsDv: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  DsDa: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  DmDq: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
) -> None:
  """
  Finite differenced Jacobians of (force, sensors) = mj_inverse(state, acceleration)   All outputs are optional. Output dimensions (transposed w.r.t Control Theory convention):     DfDq: (nv x nv)     DfDv: (nv x nv)     DfDa: (nv x nv)     DsDq: (nv x nsensordata)     DsDv: (nv x nsensordata)     DsDa: (nv x nsensordata)     DmDq: (nv x nM)   single-letter shortcuts:     inputs: q=qpos, v=qvel, a=qacc     outputs: f=qfrc_inverse, s=sensordata, m=qM   notes:     optionally computes mass matrix Jacobian DmDq     flg_actuation specifies whether to subtract qfrc_actuator from qfrc_inverse
  """

def mjd_quatIntegrate(
  vel: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  scale: typing.SupportsFloat,
  Dquat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[9, 1]", "flags.writeable"
  ],
  Dvel: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[9, 1]", "flags.writeable"
  ],
  Dscale: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
) -> None:
  """
  Derivatives of mju_quatIntegrate.
  """

def mjd_subQuat(
  qa: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  qb: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  Da: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  Db: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
) -> None:
  """
  Derivatives of mju_subQuat.
  """

def mjd_transitionFD(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  eps: typing.SupportsFloat,
  flg_centered: typing.SupportsInt,
  A: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  B: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  C: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
  D: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ]
  | None,
) -> None:
  """
  Finite differenced transition matrices (control theory notation)   d(x_next) = A*dx + B*du   d(sensor) = C*dx + D*du   required output matrix dimensions:      A: (2*nv+na x 2*nv+na)      B: (2*nv+na x nu)      D: (nsensordata x 2*nv+na)      C: (nsensordata x nu)
  """

def mju_Halton(index: typing.SupportsInt, base: typing.SupportsInt) -> float:
  """
  Generate Halton sequence.
  """

def mju_L1(
  vec: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
) -> float:
  """
  Return L1 norm: sum(abs(vec)).
  """

def mju_add(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  vec2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Set res = vec1 + vec2.
  """

def mju_add3(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  vec1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  vec2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Set res = vec1 + vec2.
  """

def mju_addScl(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  vec2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  scl: typing.SupportsFloat,
) -> None:
  """
  Set res = vec1 + vec2*scl.
  """

def mju_addScl3(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  vec1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  vec2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  scl: typing.SupportsFloat,
) -> None:
  """
  Set res = vec1 + vec2*scl.
  """

def mju_addTo(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Set res = res + vec.
  """

def mju_addTo3(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Set res = res + vec.
  """

def mju_addToScl(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  scl: typing.SupportsFloat,
) -> None:
  """
  Set res = res + vec*scl.
  """

def mju_addToScl3(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  scl: typing.SupportsFloat,
) -> None:
  """
  Set res = res + vec*scl.
  """

def mju_axisAngle2Quat(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  axis: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  angle: typing.SupportsFloat,
) -> None:
  """
  Convert axisAngle to quaternion.
  """

def mju_band2Dense(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  mat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  ntotal: typing.SupportsInt,
  nband: typing.SupportsInt,
  ndense: typing.SupportsInt,
  flg_sym: typing.SupportsInt,
) -> None:
  """
  Convert banded matrix to dense matrix, fill upper triangle if flg_sym>0.
  """

def mju_bandDiag(
  i: typing.SupportsInt,
  ntotal: typing.SupportsInt,
  nband: typing.SupportsInt,
  ndense: typing.SupportsInt,
) -> int:
  """
  Address of diagonal element i in band-dense matrix representation.
  """

def mju_bandMulMatVec(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  vec: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  ntotal: typing.SupportsInt,
  nband: typing.SupportsInt,
  ndense: typing.SupportsInt,
  nvec: typing.SupportsInt,
  flg_sym: typing.SupportsInt,
) -> None:
  """
  Multiply band-diagonal matrix with nvec vectors, include upper triangle if flg_sym>0.
  """

def mju_boxQP(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  R: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  index: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[m, 1]", "flags.writeable"
  ]
  | None,
  H: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  g: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  lower: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"] | None,
  upper: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"] | None,
) -> int:
  """
  minimize 0.5*x'*H*x + x'*g  s.t. lower <= x <= upper, return rank or -1 if failed   inputs:     n           - problem dimension     H           - SPD matrix                n*n     g           - bias vector               n     lower       - lower bounds              n     upper       - upper bounds              n     res         - solution warmstart        n   return value:     nfree <= n  - rank of unconstrained subspace, -1 if failure   outputs (required):     res         - solution                  n     R           - subspace Cholesky factor  nfree*nfree    allocated: n*(n+7)   outputs (optional):     index       - set of free dimensions    nfree          allocated: n   notes:     the initial value of res is used to warmstart the solver     R must have allocatd size n*(n+7), but only nfree*nfree values are used in output     index (if given) must have allocated size n, but only nfree values are used in output     only the lower triangles of H and R and are read from and written to, respectively     the convenience function mju_boxQPmalloc allocates the required data structures
  """

def mju_cholFactor(
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  mindiag: typing.SupportsFloat,
) -> int:
  """
  Cholesky decomposition: mat = L*L'; return rank, decomposition performed in-place into mat.
  """

def mju_cholFactorBand(
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  ntotal: typing.SupportsInt,
  nband: typing.SupportsInt,
  ndense: typing.SupportsInt,
  diagadd: typing.SupportsFloat,
  diagmul: typing.SupportsFloat,
) -> float:
  """
  Band-dense Cholesky decomposition.  Returns minimum value in the factorized diagonal, or 0 if rank-deficient.  mat has (ntotal-ndense) x nband + ndense x ntotal elements.  The first (ntotal-ndense) x nband store the band part, left of diagonal, inclusive.  The second ndense x ntotal store the band part as entire dense rows.  Add diagadd+diagmul*mat_ii to diagonal before factorization.
  """

def mju_cholSolve(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Solve (mat*mat') * res = vec, where mat is a Cholesky factor.
  """

def mju_cholSolveBand(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  mat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  ntotal: typing.SupportsInt,
  nband: typing.SupportsInt,
  ndense: typing.SupportsInt,
) -> None:
  """
  Solve (mat*mat')*res = vec where mat is a band-dense Cholesky factor.
  """

def mju_cholUpdate(
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"],
  flg_plus: typing.SupportsInt,
) -> int:
  """
  Cholesky rank-one update: L*L' +/- x*x'; return rank.
  """

def mju_clip(
  x: typing.SupportsFloat, min: typing.SupportsFloat, max: typing.SupportsFloat
) -> float:
  """
  Clip x to the range [min, max].
  """

def mju_copy(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Set res = vec.
  """

def mju_copy3(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  data: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Set res = vec.
  """

def mju_copy4(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  data: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
) -> None:
  """
  Set res = vec.
  """

def mju_cross(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  a: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  b: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Compute cross-product: res = cross(a, b).
  """

def mju_d2n(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Convert from double to mjtNum.
  """

def mju_decodePyramid(
  force: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  pyramid: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  mu: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Convert pyramid representation to contact force.
  """

def mju_dense2Band(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  ntotal: typing.SupportsInt,
  nband: typing.SupportsInt,
  ndense: typing.SupportsInt,
) -> None:
  """
  Convert dense matrix to banded matrix.
  """

def mju_dense2sparse(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  rownnz: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[m, 1]", "flags.writeable"
  ],
  rowadr: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[m, 1]", "flags.writeable"
  ],
  colind: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[m, 1]", "flags.writeable"
  ],
) -> int:
  """
  Convert matrix from dense to sparse.  nnz is size of res and colind, return 1 if too small, 0 otherwise.
  """

def mju_derivQuat(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
  vel: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Compute time-derivative of quaternion, given 3D rotational velocity.
  """

def mju_dist3(
  pos1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  pos2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
  """
  Return Cartesian distance between 3D vectors pos1 and pos2.
  """

def mju_dot(
  vec1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  vec2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> float:
  """
  Return dot-product of vec1 and vec2.
  """

def mju_dot3(
  vec1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  vec2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
  """
  Return dot-product of vec1 and vec2.
  """

def mju_eig3(
  eigval: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  eigvec: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[9, 1]", "flags.writeable"
  ],
  quat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  mat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[9, 1]"],
) -> int:
  """
  Eigenvalue decomposition of symmetric 3x3 matrix, mat = eigvec * diag(eigval) * eigvec'.
  """

def mju_encodePyramid(
  pyramid: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  force: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  mu: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Convert contact force to pyramid representation.
  """

def mju_euler2Quat(
  quat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  euler: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  seq: str,
) -> None:
  """
  Convert sequence of Euler angles (radians) to quaternion. seq[0,1,2] must be in 'xyzXYZ', lower/upper-case mean intrinsic/extrinsic rotations.
  """

def mju_eye(
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
) -> None:
  """
  Set mat to the identity matrix.
  """

def mju_f2n(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[m, 1]"],
) -> None:
  """
  Convert from float to mjtNum.
  """

def mju_fill(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  val: typing.SupportsFloat,
) -> None:
  """
  Set res = val.
  """

def mju_insertionSort(
  list: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
) -> None:
  """
  Insertion sort, resulting list is in increasing order.
  """

def mju_insertionSortInt(
  list: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[m, 1]", "flags.writeable"
  ],
) -> None:
  """
  Integer insertion sort, resulting list is in increasing order.
  """

def mju_isBad(x: typing.SupportsFloat) -> int:
  """
  Return 1 if nan or abs(x)>mjMAXVAL, 0 otherwise. Used by check functions.
  """

def mju_isZero(
  vec: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
) -> int:
  """
  Return 1 if all elements are 0.
  """

def mju_mat2Quat(
  quat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  mat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[9, 1]"],
) -> None:
  """
  Convert 3D rotation matrix to quaternion.
  """

def mju_mat2Rot(
  quat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  mat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[9, 1]"],
) -> int:
  """
  Extract 3D rotation from an arbitrary 3x3 matrix by refining the input quaternion. Returns the number of iterations required to converge
  """

def mju_max(a: typing.SupportsFloat, b: typing.SupportsFloat) -> float:
  """
  Return max(a,b) with single evaluation of a and b.
  """

def mju_min(a: typing.SupportsFloat, b: typing.SupportsFloat) -> float:
  """
  Return min(a,b) with single evaluation of a and b.
  """

def mju_mulMatMat(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  mat1: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  mat2: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
) -> None:
  """
  Multiply matrices: res = mat1 * mat2.
  """

def mju_mulMatMatT(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  mat1: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  mat2: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
) -> None:
  """
  Multiply matrices, second argument transposed: res = mat1 * mat2'.
  """

def mju_mulMatTMat(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  mat1: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  mat2: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
) -> None:
  """
  Multiply matrices, first argument transposed: res = mat1' * mat2.
  """

def mju_mulMatTVec(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Multiply transposed matrix and vector: res = mat' * vec.
  """

def mju_mulMatTVec3(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  mat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[9, 1]"],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Multiply transposed 3-by-3 matrix by vector: res = mat' * vec.
  """

def mju_mulMatVec(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Multiply matrix and vector: res = mat * vec.
  """

def mju_mulMatVec3(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  mat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[9, 1]"],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Multiply 3-by-3 matrix by vector: res = mat * vec.
  """

def mju_mulPose(
  posres: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  quatres: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  pos1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  quat1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
  pos2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  quat2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
) -> None:
  """
  Multiply two poses.
  """

def mju_mulQuat(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  quat1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
  quat2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
) -> None:
  """
  Multiply quaternions.
  """

def mju_mulQuatAxis(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
  axis: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Multiply quaternion and axis.
  """

def mju_mulVecMatVec(
  vec1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  vec2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> float:
  """
  Multiply square matrix with vectors on both sides: returns vec1' * mat * vec2.
  """

def mju_muscleBias(
  len: typing.SupportsFloat,
  lengthrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
  acc0: typing.SupportsFloat,
  prm: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[9, 1]"],
) -> float:
  """
  Muscle passive force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax).
  """

def mju_muscleDynamics(
  ctrl: typing.SupportsFloat,
  act: typing.SupportsFloat,
  prm: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
  """
  Muscle activation dynamics, prm = (tau_act, tau_deact, smoothing_width).
  """

def mju_muscleGain(
  len: typing.SupportsFloat,
  vel: typing.SupportsFloat,
  lengthrange: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[2, 1]"],
  acc0: typing.SupportsFloat,
  prm: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[9, 1]"],
) -> float:
  """
  Muscle active force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax).
  """

def mju_n2d(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Convert from mjtNum to double.
  """

def mju_n2f(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float32], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Convert from mjtNum to float.
  """

def mju_negPose(
  posres: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  quatres: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
) -> None:
  """
  Conjugate pose, corresponding to the opposite spatial transformation.
  """

def mju_negQuat(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
) -> None:
  """
  Conjugate quaternion, corresponding to opposite rotation.
  """

def mju_norm(
  res: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> float:
  """
  Return vector length (without normalizing vector).
  """

def mju_norm3(
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> float:
  """
  Return vector length (without normalizing the vector).
  """

def mju_normalize(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
) -> float:
  """
  Normalize vector, return length before normalization.
  """

def mju_normalize3(
  vec: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
) -> float:
  """
  Normalize vector, return length before normalization.
  """

def mju_normalize4(
  vec: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
) -> float:
  """
  Normalize vector, return length before normalization.
  """

def mju_printMat(
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
) -> None:
  """
  Print matrix to screen.
  """

def mju_printMatSparse(
  mat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  rownnz: typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"],
  rowadr: typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"],
  colind: typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"],
) -> None:
  """
  Print sparse matrix to screen.
  """

def mju_quat2Mat(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[9, 1]", "flags.writeable"
  ],
  quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
) -> None:
  """
  Convert quaternion to 3D rotation matrix.
  """

def mju_quat2Vel(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
  dt: typing.SupportsFloat,
) -> None:
  """
  Convert quaternion (corresponding to orientation difference) to 3D velocity.
  """

def mju_quatIntegrate(
  quat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  vel: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  scale: typing.SupportsFloat,
) -> None:
  """
  Integrate quaternion given 3D angular velocity.
  """

def mju_quatZ2Vec(
  quat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Construct quaternion performing rotation from z-axis to given vector.
  """

def mju_rayFlex(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  flex_layer: typing.SupportsInt,
  flg_vert: typing.SupportsInt,
  flg_edge: typing.SupportsInt,
  flg_face: typing.SupportsInt,
  flg_skin: typing.SupportsInt,
  flexid: typing.SupportsInt,
  pnt: typing.SupportsFloat,
  vec: typing.SupportsFloat,
  vertid: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[1, 1]", "flags.writeable"
  ],
) -> float:
  """
  Intersect ray with flex, return nearest distance or -1 if no intersection, and also output nearest vertex id.
  """

def mju_rayGeom(
  pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  mat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[9, 1]"],
  size: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  pnt: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  geomtype: typing.SupportsInt,
) -> float:
  """
  Intersect ray with pure geom, return nearest distance or -1 if no intersection.
  """

def mju_raySkin(
  nface: typing.SupportsInt,
  nvert: typing.SupportsInt,
  face: typing.SupportsInt,
  vert: typing.SupportsFloat,
  pnt: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  vertid: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[1, 1]", "flags.writeable"
  ],
) -> float:
  """
  Intersect ray with skin, return nearest distance or -1 if no intersection, and also output nearest vertex id.
  """

def mju_rotVecQuat(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
) -> None:
  """
  Rotate vector by quaternion.
  """

def mju_round(x: typing.SupportsFloat) -> int:
  """
  Round x to nearest integer.
  """

def mju_scl(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  scl: typing.SupportsFloat,
) -> None:
  """
  Set res = vec*scl.
  """

def mju_scl3(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  scl: typing.SupportsFloat,
) -> None:
  """
  Set res = vec*scl.
  """

def mju_sigmoid(x: typing.SupportsFloat) -> float:
  """
  Sigmoid function over 0<=x<=1 using quintic polynomial.
  """

def mju_sign(x: typing.SupportsFloat) -> float:
  """
  Return sign of x: +1, -1 or 0.
  """

def mju_sparse2dense(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  mat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  rownnz: typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"],
  rowadr: typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"],
  colind: typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"],
) -> None:
  """
  Convert matrix from sparse to dense.
  """

def mju_springDamper(
  pos0: typing.SupportsFloat,
  vel0: typing.SupportsFloat,
  Kp: typing.SupportsFloat,
  Kv: typing.SupportsFloat,
  dt: typing.SupportsFloat,
) -> float:
  """
  Integrate spring-damper analytically, return pos(dt).
  """

def mju_sqrMatTD(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
  diag: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ]
  | None,
) -> None:
  """
  Set res = mat' * diag * mat if diag is not NULL, and res = mat' * mat otherwise.
  """

def mju_standardNormal(num2: typing.SupportsFloat | None) -> float:
  """
  Standard normal random number generator (optional second number).
  """

def mju_str2Type(str: str) -> int:
  """
  Convert type name to type id (mjtObj).
  """

def mju_sub(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
  vec2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Set res = vec1 - vec2.
  """

def mju_sub3(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  vec1: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  vec2: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Set res = vec1 - vec2.
  """

def mju_subFrom(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
) -> None:
  """
  Set res = res - vec.
  """

def mju_subFrom3(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Set res = res - vec.
  """

def mju_subQuat(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  qa: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
  qb: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
) -> None:
  """
  Subtract quaternions, express as 3D velocity: qb*quat(res) = qa.
  """

def mju_sum(
  vec: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
) -> float:
  """
  Return sum(vec).
  """

def mju_symmetrize(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
) -> None:
  """
  Symmetrize square matrix res = (mat + mat')/2.
  """

def mju_transformSpatial(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[6, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[6, 1]"],
  flg_force: typing.SupportsInt,
  newpos: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  oldpos: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  rotnew2old: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[9, 1]"],
) -> None:
  """
  Coordinate transform of 6D motion or force vector in rotation:translation format. rotnew2old is 3-by-3, NULL means no rotation; flg_force specifies force or motion type.
  """

def mju_transpose(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64],
    "[m, n]",
    "flags.writeable",
    "flags.c_contiguous",
  ],
  mat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"
  ],
) -> None:
  """
  Transpose matrix: res = mat'.
  """

def mju_trnVecPose(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  quat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Transform vector by pose.
  """

def mju_type2Str(type: typing.SupportsInt) -> str:
  """
  Convert type id (mjtObj) to type name.
  """

def mju_unit4(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
) -> None:
  """
  Set res = (1,0,0,0).
  """

def mju_warningText(warning: typing.SupportsInt, info: typing.SupportsInt) -> str:
  """
  Construct a warning message given the warning type and info.
  """

def mju_writeLog(type: str, msg: str) -> None:
  """
  Write [datetime, type: message] to MUJOCO_LOG.TXT.
  """

def mju_writeNumBytes(nbytes: typing.SupportsInt) -> str:
  """
  Return human readable number of bytes using standard letter suffix.
  """

def mju_zero(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"
  ],
) -> None:
  """
  Set res = 0.
  """

def mju_zero3(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
) -> None:
  """
  Set res = 0.
  """

def mju_zero4(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
) -> None:
  """
  Set res = 0.
  """

def mjv_addGeoms(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  opt: mujoco._structs.MjvOption,
  pert: mujoco._structs.MjvPerturb,
  catmask: typing.SupportsInt,
  scn: mujoco._structs.MjvScene,
) -> None:
  """
  Add geoms from selected categories.
  """

def mjv_alignToCamera(
  res: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  vec: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  forward: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Rotate 3D vec in horizontal plane by angle between (0,1) and (forward_x,forward_y).
  """

def mjv_applyPerturbForce(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  pert: mujoco._structs.MjvPerturb,
) -> None:
  """
  Set perturb force,torque in d->xfrc_applied, if selected body is dynamic.
  """

def mjv_applyPerturbPose(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  pert: mujoco._structs.MjvPerturb,
  flg_paused: typing.SupportsInt,
) -> None:
  """
  Set perturb pos,quat in d->mocap when selected body is mocap, and in d->qpos otherwise. Write d->qpos only if flg_paused and subtree root for selected body has free joint.
  """

def mjv_cameraInModel(
  headpos: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  forward: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  up: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  scn: mujoco._structs.MjvScene,
) -> None:
  """
  Get camera info in model space; average left and right OpenGL cameras.
  """

def mjv_cameraInRoom(
  headpos: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  forward: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  up: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  scn: mujoco._structs.MjvScene,
) -> None:
  """
  Get camera info in room space; average left and right OpenGL cameras.
  """

def mjv_connector(
  geom: mujoco._structs.MjvGeom,
  type: typing.SupportsInt,
  width: typing.SupportsFloat,
  from_: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  to: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
) -> None:
  """
  Set (type, size, pos, mat) for connector-type geom between given points. Assume that mjv_initGeom was already called to set all other properties. Width of mjGEOM_LINE is denominated in pixels.
  """

def mjv_defaultCamera(cam: mujoco._structs.MjvCamera) -> None:
  """
  Set default camera.
  """

def mjv_defaultFigure(fig: mujoco._structs.MjvFigure) -> None:
  """
  Set default figure.
  """

def mjv_defaultFreeCamera(
  m: mujoco._structs.MjModel, cam: mujoco._structs.MjvCamera
) -> None:
  """
  Set default free camera.
  """

def mjv_defaultOption(opt: mujoco._structs.MjvOption) -> None:
  """
  Set default visualization options.
  """

def mjv_defaultPerturb(pert: mujoco._structs.MjvPerturb) -> None:
  """
  Set default perturbation.
  """

def mjv_frustumHeight(scn: mujoco._structs.MjvScene) -> float:
  """
  Get frustum height at unit distance from camera; average left and right OpenGL cameras.
  """

def mjv_initGeom(
  geom: mujoco._structs.MjvGeom,
  type: typing.SupportsInt,
  size: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  pos: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  mat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[9, 1]"],
  rgba: typing.Annotated[numpy.typing.NDArray[numpy.float32], "[4, 1]"],
) -> None:
  """
  Initialize given geom fields when not NULL, set the rest to their default values.
  """

def mjv_initPerturb(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  scn: mujoco._structs.MjvScene,
  pert: mujoco._structs.MjvPerturb,
) -> None:
  """
  Copy perturb pos,quat from selected body; set scale for perturbation.
  """

def mjv_makeLights(
  m: mujoco._structs.MjModel, d: mujoco._structs.MjData, scn: mujoco._structs.MjvScene
) -> None:
  """
  Make list of lights.
  """

def mjv_model2room(
  roompos: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  roomquat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  modelpos: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  modelquat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
  scn: mujoco._structs.MjvScene,
) -> None:
  """
  Transform pose from model to room space.
  """

def mjv_moveCamera(
  m: mujoco._structs.MjModel,
  action: typing.SupportsInt,
  reldx: typing.SupportsFloat,
  reldy: typing.SupportsFloat,
  scn: mujoco._structs.MjvScene,
  cam: mujoco._structs.MjvCamera,
) -> None:
  """
  Move camera with mouse; action is mjtMouse.
  """

def mjv_moveModel(
  m: mujoco._structs.MjModel,
  action: typing.SupportsInt,
  reldx: typing.SupportsFloat,
  reldy: typing.SupportsFloat,
  roomup: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  scn: mujoco._structs.MjvScene,
) -> None:
  """
  Move model with mouse; action is mjtMouse.
  """

def mjv_movePerturb(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  action: typing.SupportsInt,
  reldx: typing.SupportsFloat,
  reldy: typing.SupportsFloat,
  scn: mujoco._structs.MjvScene,
  pert: mujoco._structs.MjvPerturb,
) -> None:
  """
  Move perturb object with mouse; action is mjtMouse.
  """

def mjv_room2model(
  modelpos: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  modelquat: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[4, 1]", "flags.writeable"
  ],
  roompos: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"],
  roomquat: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 1]"],
  scn: mujoco._structs.MjvScene,
) -> None:
  """
  Transform pose from room to model space.
  """

def mjv_select(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  vopt: mujoco._structs.MjvOption,
  aspectratio: typing.SupportsFloat,
  relx: typing.SupportsFloat,
  rely: typing.SupportsFloat,
  scn: mujoco._structs.MjvScene,
  selpnt: typing.Annotated[
    numpy.typing.NDArray[numpy.float64], "[3, 1]", "flags.writeable"
  ],
  geomid: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[1, 1]", "flags.writeable"
  ],
  flexid: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[1, 1]", "flags.writeable"
  ],
  skinid: typing.Annotated[
    numpy.typing.NDArray[numpy.int32], "[1, 1]", "flags.writeable"
  ],
) -> int:
  """
  Select geom, flex or skin with mouse, return bodyid; -1: none selected.
  """

def mjv_updateCamera(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  cam: mujoco._structs.MjvCamera,
  scn: mujoco._structs.MjvScene,
) -> None:
  """
  Update camera.
  """

def mjv_updateScene(
  m: mujoco._structs.MjModel,
  d: mujoco._structs.MjData,
  opt: mujoco._structs.MjvOption,
  pert: mujoco._structs.MjvPerturb | None,
  cam: mujoco._structs.MjvCamera,
  catmask: typing.SupportsInt,
  scn: mujoco._structs.MjvScene,
) -> None:
  """
  Update entire scene given model state.
  """

def mjv_updateSkin(
  m: mujoco._structs.MjModel, d: mujoco._structs.MjData, scn: mujoco._structs.MjvScene
) -> None:
  """
  Update skins.
  """
