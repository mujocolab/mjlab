from dataclasses import dataclass
import numpy as np
import mujoco

from mjlab.core.editors import SpecEditor
from mjlab.entities.robots.robot_config import KeyframeCfg, ActuatorCfg, SensorCfg
from mjlab.utils.string import resolve_expr, filter_exp
from mjlab.utils.spec import get_non_root_joints


@dataclass
class KeyframeEditor(SpecEditor):
  name: str
  cfg: KeyframeCfg

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    jnt_names = [j.name for j in get_non_root_joints(spec)]

    joint_pos = resolve_expr(self.cfg.joint_pos, jnt_names)
    joint_vel = resolve_expr(self.cfg.joint_vel, jnt_names)
    ctrl = (
      joint_pos
      if self.cfg.use_joint_pos_for_ctrl
      else resolve_expr(self.cfg.ctrl, jnt_names)
    )

    qpos = np.concatenate((self.cfg.root_pos, self.cfg.root_quat, joint_pos))
    qvel = np.concatenate((self.cfg.root_lin_vel, self.cfg.root_ang_vel, joint_vel))

    spec.add_key(
      name=self.name,
      time=self.cfg.time,
      qpos=np.asarray(qpos),
      qvel=np.asarray(qvel),
      ctrl=np.asarray(ctrl),
    )


@dataclass
class ActuatorEditor(SpecEditor):
  cfgs: tuple[ActuatorCfg, ...]

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    jnts = get_non_root_joints(spec)
    joint_names = [j.name for j in jnts]

    # Build a list of (cfg, joint_name) by resolving the config regex.
    flat: list[tuple[ActuatorCfg, str]] = []
    for cfg in self.cfgs:
      matched = filter_exp(cfg.joint_names_expr, joint_names)
      for name in matched:
        flat.append((cfg, name))

    # Sort by the joint index in the spec.
    flat.sort(key=lambda pair: joint_names.index(pair[1]))

    for cfg, jn in flat:
      spec.joint(jn).armature = cfg.armature
      spec.joint(jn).frictionloss = cfg.frictionloss

      act = spec.add_actuator(
        name=jn,
        target=jn,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        inheritrange=1.0,
        forcerange=(-cfg.effort_limit, cfg.effort_limit),
      )
      act.gainprm[0] = cfg.stiffness
      act.biasprm[1] = -cfg.stiffness
      act.biasprm[2] = -cfg.damping


@dataclass
class SensorEditor(SpecEditor):
  name: str
  cfg: SensorCfg

  SENSOR_TYPE_MAP = {
    "gyro": mujoco.mjtSensor.mjSENS_GYRO,
    "upvector": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
    "velocimeter": mujoco.mjtSensor.mjSENS_VELOCIMETER,
    "framequat": mujoco.mjtSensor.mjSENS_FRAMEQUAT,
    "framepos": mujoco.mjtSensor.mjSENS_FRAMEPOS,
    "framelinvel": mujoco.mjtSensor.mjSENS_FRAMELINVEL,
    "frameangvel": mujoco.mjtSensor.mjSENS_FRAMEANGVEL,
    "framezaxis": mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
    "accelerometer": mujoco.mjtSensor.mjSENS_ACCELEROMETER,
  }

  SENSOR_OBJECT_TYPE_MAP = {
    "site": mujoco.mjtObj.mjOBJ_SITE,
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "body": mujoco.mjtObj.mjOBJ_BODY,
  }

  def edit_spec(self, spec: mujoco.MjSpec) -> None:
    spec.add_sensor(
      type=self.SENSOR_TYPE_MAP[self.cfg.sensor_type],
      objtype=self.SENSOR_OBJECT_TYPE_MAP[self.cfg.object_type],
      name=self.name,
      objname=self.cfg.object_name,
    )
