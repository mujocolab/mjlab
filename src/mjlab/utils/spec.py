"""MjSpec utils."""

import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from mjlab.actuator.actuator import TransmissionType
from mjlab.utils.xml import fix_spec_xml, strip_buffer_textures

_DEFAULT_SPEC_OPTION = mujoco.MjSpec().option

_OPTION_FIELDS = (
  "ccd_iterations",
  "ccd_tolerance",
  "cone",
  "density",
  "disableactuator",
  "disableflags",
  "enableflags",
  "gravity",
  "impratio",
  "integrator",
  "iterations",
  "jacobian",
  "ls_iterations",
  "ls_tolerance",
  "magnetic",
  "noslip_iterations",
  "noslip_tolerance",
  "o_friction",
  "o_margin",
  "o_solimp",
  "o_solref",
  "sdf_initpoints",
  "sdf_iterations",
  "sleep_tolerance",
  "solver",
  "timestep",
  "tolerance",
  "viscosity",
  "wind",
)


def non_default_option_fields(opt: mujoco._specs.MjOption) -> list[str]:
  """Return option field names that differ from MjSpec defaults."""
  diffs = []
  for name in _OPTION_FIELDS:
    default = getattr(_DEFAULT_SPEC_OPTION, name)
    value = getattr(opt, name)
    if isinstance(default, np.ndarray):
      if not np.array_equal(default, value):
        diffs.append(name)
    elif default != value:
      diffs.append(name)
  return diffs


def export_spec(
  spec: mujoco.MjSpec,
  output_dir: Path,
  *,
  zip: bool = False,
) -> None:
  """Write a spec's XML and referenced mesh assets to a directory.

  Creates ``scene.xml`` and an ``assets/`` subdirectory containing only the assets
  referenced by the generated XML. When *zip* is True the directory is compressed into
  a ``.zip`` archive and removed.

  Operates on a copy of spec to avoid mutation.
  """
  output_dir.mkdir(parents=True, exist_ok=True)
  tmp = spec.copy()
  strip_buffer_textures(tmp)
  xml = fix_spec_xml(tmp.to_xml(), meshdir="assets")
  (output_dir / "scene.xml").write_text(xml)

  # Collect file paths referenced in the XML.
  root = ET.fromstring(xml)
  referenced: set[str] = set()
  for elem in root.iter():
    file_val = elem.get("file")
    if file_val:
      referenced.add(file_val)

  # Write only referenced assets. Match asset keys to XML file attributes by path
  # suffix because keys may carry the original meshdir prefix (e.g.
  # "../../meshes/robot/arm.stl" for a file attribute of "robot/arm.stl").
  assets_dir = output_dir / "assets"
  for ref_path in sorted(referenced):
    for key, data in tmp.assets.items():
      norm = key.replace("\\", "/")
      if norm == ref_path or norm.endswith("/" + ref_path):
        out = assets_dir / ref_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        break

  if zip:
    zip_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
      for file in sorted(output_dir.rglob("*")):
        if file.is_file():
          zf.write(file, file.relative_to(output_dir))
    shutil.rmtree(output_dir)


_TRANSMISSION_TYPE_MAP = {
  TransmissionType.JOINT: mujoco.mjtTrn.mjTRN_JOINT,
  TransmissionType.TENDON: mujoco.mjtTrn.mjTRN_TENDON,
  TransmissionType.SITE: mujoco.mjtTrn.mjTRN_SITE,
}


def auto_wrap_fixed_base_mocap(
  spec_fn: Callable[[], mujoco.MjSpec],
) -> Callable[[], mujoco.MjSpec]:
  """Wraps spec_fn to auto-wrap fixed-base entities in mocap.

  This enables fixed-base entities to be positioned independently per environment.
  Returns original spec unchanged if entity is floating-base or already mocap.

  .. note::
    Mocap wrapping is automatic, but positioning only happens when you call a
    reset event (e.g., reset_root_state_uniform). Without a reset event, all
    fixed-base robots will remain at the world origin.

  See FAQ: "Why are my fixed-base robots all stacked at the origin?"
  """

  def wrapper() -> mujoco.MjSpec:
    original_spec = spec_fn()

    # Check if entity has freejoint (floating-base).
    free_joint = get_free_joint(original_spec)
    if free_joint is not None:
      return original_spec  # Floating-base, no wrapping needed.

    # Check if root body is already mocap.
    root_body = original_spec.bodies[1] if len(original_spec.bodies) > 1 else None
    if root_body and root_body.mocap:
      return original_spec  # Already mocap, no wrapping needed.

    # Extract and delete keyframes before attach (they transfer but we need
    # them on the wrapper spec, not nested in the attached spec).
    keyframes = [
      (np.array(k.qpos), np.array(k.ctrl), k.name) for k in original_spec.keys
    ]
    for k in list(original_spec.keys):
      original_spec.delete(k)

    # Wrap in mocap body.
    wrapper_spec = mujoco.MjSpec()
    mocap_body = wrapper_spec.worldbody.add_body(name="mocap_base", mocap=True)
    frame = mocap_body.add_frame()
    wrapper_spec.attach(child=original_spec, prefix="", frame=frame)

    # Re-add keyframes to wrapper spec.
    for qpos, ctrl, name in keyframes:
      wrapper_spec.add_key(name=name, qpos=qpos.tolist(), ctrl=ctrl.tolist())

    return wrapper_spec

  return wrapper


def get_non_free_joints(spec: mujoco.MjSpec) -> tuple[mujoco.MjsJoint, ...]:
  """Returns all joints except the free joint."""
  joints: list[mujoco.MjsJoint] = []
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      continue
    joints.append(jnt)
  return tuple(joints)


def get_free_joint(spec: mujoco.MjSpec) -> mujoco.MjsJoint | None:
  """Returns the free joint. None if no free joint exists."""
  joint: mujoco.MjsJoint | None = None
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
      joint = jnt
      break
  return joint


def disable_collision(geom: mujoco.MjsGeom) -> None:
  """Disables collision for a geom."""
  geom.contype = 0
  geom.conaffinity = 0


def is_joint_limited(jnt: mujoco.MjsJoint) -> bool:
  """Returns True if a joint is limited."""
  match jnt.limited:
    case mujoco.mjtLimited.mjLIMITED_TRUE:
      return True
    case mujoco.mjtLimited.mjLIMITED_AUTO:
      return jnt.range[0] < jnt.range[1]
    case _:
      return False


def create_motor_actuator(
  spec: mujoco.MjSpec,
  joint_name: str,
  *,
  effort_limit: float,
  gear: float = 1.0,
  armature: float | None = None,
  frictionloss: float | None = None,
  viscous_damping: float | None = None,
  transmission_type: TransmissionType = TransmissionType.JOINT,
) -> mujoco.MjsActuator:
  """Create a <motor> actuator."""
  actuator = spec.add_actuator(name=joint_name, target=joint_name)

  actuator.trntype = _TRANSMISSION_TYPE_MAP[transmission_type]
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_NONE

  actuator.gear[0] = gear
  # Technically redundant to set both but being explicit here.
  actuator.forcelimited = True
  actuator.forcerange[:] = np.array([-effort_limit, effort_limit])
  actuator.ctrllimited = True
  actuator.ctrlrange[:] = np.array([-effort_limit, effort_limit])

  # Set armature, frictionloss, and viscous_damping (None = preserve XML value).
  if transmission_type == TransmissionType.JOINT:
    if armature is not None:
      spec.joint(joint_name).armature = armature
    if frictionloss is not None:
      spec.joint(joint_name).frictionloss = frictionloss
    if viscous_damping is not None:
      spec.joint(joint_name).damping[0] = viscous_damping
  elif transmission_type == TransmissionType.TENDON:
    if armature is not None:
      spec.tendon(joint_name).armature = armature
    if frictionloss is not None:
      spec.tendon(joint_name).frictionloss = frictionloss
    if viscous_damping is not None:
      spec.tendon(joint_name).damping[0] = viscous_damping

  return actuator


def create_position_actuator(
  spec: mujoco.MjSpec,
  joint_name: str,
  *,
  stiffness: float,
  damping: float,
  effort_limit: float | None = None,
  armature: float | None = None,
  frictionloss: float | None = None,
  viscous_damping: float | None = None,
  transmission_type: TransmissionType = TransmissionType.JOINT,
) -> mujoco.MjsActuator:
  """Creates a <position> actuator.

  An important note about this actuator is that we set `ctrllimited` to False. This is
  because we want to allow the policy to output setpoints that are outside the kinematic
  limits of the joint.
  """
  actuator = spec.add_actuator(name=joint_name, target=joint_name)

  actuator.trntype = _TRANSMISSION_TYPE_MAP[transmission_type]
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE

  # Set stiffness and damping.
  actuator.gainprm[0] = stiffness
  actuator.biasprm[1] = -stiffness
  actuator.biasprm[2] = -damping

  # Position actuators must allow setpoints beyond joint limits.
  # Since force = stiffness * (ctrl - pos), clamping ctrl to the joint range would
  # produce zero force when the joint is at its limit. Force is still bounded by
  # forcerange below. Both lines are needed: ctrllimited=False is the primary guard,
  # and inheritrange=0 prevents MuJoCo from resolving the default ctrllimited=AUTO back
  # to True when a joint range exists.
  actuator.inheritrange = 0.0
  actuator.ctrllimited = False
  if effort_limit is not None:
    actuator.forcelimited = True
    actuator.forcerange[:] = np.array([-effort_limit, effort_limit])

    # Informational ctrlrange (not enforced since ctrllimited=False).
    # Assuming zero velocity, force = stiffness * (ctrl - pos). Solving for the ctrl
    # that saturates force at the worst-case position gives:
    #   ctrl_max = joint_high + effort_limit / stiffness
    #   ctrl_min = joint_low  - effort_limit / stiffness
    # Beyond this range, force is always clamped regardless of position.
    if transmission_type == TransmissionType.JOINT:
      target_range = spec.joint(joint_name).range
    elif transmission_type == TransmissionType.TENDON:
      target_range = spec.tendon(joint_name).range
    else:
      target_range = (0.0, 0.0)
    delta = effort_limit / stiffness
    actuator.ctrlrange[:] = np.array([target_range[0] - delta, target_range[1] + delta])
  else:
    actuator.forcelimited = False
    # No forcerange needed.

  # Set armature, frictionloss, and viscous_damping (None = preserve XML value).
  if transmission_type == TransmissionType.JOINT:
    if armature is not None:
      spec.joint(joint_name).armature = armature
    if frictionloss is not None:
      spec.joint(joint_name).frictionloss = frictionloss
    if viscous_damping is not None:
      spec.joint(joint_name).damping[0] = viscous_damping
  elif transmission_type == TransmissionType.TENDON:
    if armature is not None:
      spec.tendon(joint_name).armature = armature
    if frictionloss is not None:
      spec.tendon(joint_name).frictionloss = frictionloss
    if viscous_damping is not None:
      spec.tendon(joint_name).damping[0] = viscous_damping

  return actuator


def create_velocity_actuator(
  spec: mujoco.MjSpec,
  joint_name: str,
  *,
  damping: float,
  effort_limit: float | None = None,
  armature: float | None = None,
  frictionloss: float | None = None,
  viscous_damping: float | None = None,
  transmission_type: TransmissionType = TransmissionType.JOINT,
) -> mujoco.MjsActuator:
  """Creates a <velocity> actuator.

  Control inputs are not clamped so that velocity commands work for any joint,
  including continuous joints that have no range defined. Force output is still
  bounded when effort_limit is set.
  """
  actuator = spec.add_actuator(name=joint_name, target=joint_name)

  actuator.trntype = _TRANSMISSION_TYPE_MAP[transmission_type]
  actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
  actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE

  actuator.inheritrange = 0.0
  actuator.ctrllimited = False
  actuator.gainprm[0] = damping
  actuator.biasprm[2] = -damping

  if effort_limit is not None:
    # Will this throw an error with autolimits=True?
    actuator.forcelimited = True
    actuator.forcerange[:] = np.array([-effort_limit, effort_limit])
  else:
    actuator.forcelimited = False

  # Set armature, frictionloss, and viscous_damping (None = preserve XML value).
  if transmission_type == TransmissionType.JOINT:
    if armature is not None:
      spec.joint(joint_name).armature = armature
    if frictionloss is not None:
      spec.joint(joint_name).frictionloss = frictionloss
    if viscous_damping is not None:
      spec.joint(joint_name).damping[0] = viscous_damping
  elif transmission_type == TransmissionType.TENDON:
    if armature is not None:
      spec.tendon(joint_name).armature = armature
    if frictionloss is not None:
      spec.tendon(joint_name).frictionloss = frictionloss
    if viscous_damping is not None:
      spec.tendon(joint_name).damping[0] = viscous_damping

  return actuator


def create_muscle_actuator(
  spec: mujoco.MjSpec,
  target_name: str,
  *,
  length_range: tuple[float, float] = (0.0, 0.0),
  gear: float = 1.0,
  timeconst: tuple[float, float] = (0.01, 0.04),
  tausmooth: float = 0.0,
  range: tuple[float, float] = (0.75, 1.05),
  force: float = -1.0,
  scale: float = 200.0,
  lmin: float = 0.5,
  lmax: float = 1.6,
  vmax: float = 1.5,
  fpmax: float = 1.3,
  fvmax: float = 1.2,
  transmission_type: TransmissionType = TransmissionType.TENDON,
) -> mujoco.MjsActuator:
  """Create a MuJoCo <muscle> actuator with muscle dynamics.

  Muscles use special activation dynamics and force-length-velocity curves.
  They can actuate tendons or joints.
  """
  actuator = spec.add_actuator(name=target_name, target=target_name)

  if transmission_type not in [TransmissionType.JOINT, TransmissionType.TENDON]:
    raise ValueError("Muscle actuators only support JOINT and TENDON transmissions.")
  actuator.trntype = _TRANSMISSION_TYPE_MAP[transmission_type]
  actuator.dyntype = mujoco.mjtDyn.mjDYN_MUSCLE
  actuator.gaintype = mujoco.mjtGain.mjGAIN_MUSCLE
  actuator.biastype = mujoco.mjtBias.mjBIAS_MUSCLE

  actuator.gear[0] = gear
  actuator.dynprm[0:3] = np.array([*timeconst, tausmooth])
  actuator.gainprm[0:9] = np.array(
    [*range, force, scale, lmin, lmax, vmax, fpmax, fvmax]
  )
  actuator.biasprm[:] = actuator.gainprm[:]
  actuator.lengthrange[0:2] = length_range

  # TODO(kevin): Double check this.
  actuator.ctrllimited = True
  actuator.ctrlrange[:] = np.array([0.0, 1.0])

  return actuator
