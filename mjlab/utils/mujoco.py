import mujoco


def is_position_actuator(actuator) -> bool:
  """Check if an actuator is a position actuator.

  This function works on both model.actuator and spec.actuator objects.
  """
  return (
    actuator.gaintype == mujoco.mjtGain.mjGAIN_FIXED
    and actuator.biastype == mujoco.mjtBias.mjBIAS_AFFINE
    and actuator.dyntype in (mujoco.mjtDyn.mjDYN_NONE, mujoco.mjtDyn.mjDYN_FILTEREXACT)
    and actuator.gainprm[0] == -actuator.biasprm[1]
  )
