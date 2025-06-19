from typing import Dict, Tuple
from mujoco import mjx
from mujoco.mjx._src import math
import jax
import jax.numpy as jp
from mjlab.core import entity


def reset_joints_by_noise_scale(
  rng: jax.Array,
  robot: entity.Entity,
  model: mjx.Model,
  data: mjx.Data,
  position_range: Tuple[float, float] = (1.0, 1.0),
  velocity_range: Tuple[float, float] = (1.0, 1.0),
):
  jnts = robot.get_non_root_joints()
  qpos_ids = jp.array([model.bind(jnt).qposadr for jnt in jnts])
  qvel_ids = jp.array([model.bind(jnt).dofadr for jnt in jnts])
  limits = jp.stack([model.bind(jnt).range for jnt in jnts], axis=0)
  rng, key_pos, key_vel = jax.random.split(rng, 3)
  pos_scale = jax.random.uniform(
    key_pos, minval=position_range[0], maxval=position_range[1]
  )
  vel_scale = jax.random.uniform(
    key_vel, minval=velocity_range[0], maxval=velocity_range[1]
  )
  qpos = data.qpos.at[qpos_ids].set(data.qpos[qpos_ids] * pos_scale)
  qpos = qpos.at[qpos_ids].set(jp.clip(qpos[qpos_ids], limits[:, 0], limits[:, 1]))
  qvel = data.qvel.at[qvel_ids].set(data.qvel[qvel_ids] * vel_scale)
  return data.replace(qpos=qpos, qvel=qvel)


def reset_joints_by_noise_add(
  rng: jax.Array,
  robot: entity.Entity,
  model: mjx.Model,
  data: mjx.Data,
  position_range: Tuple[float, float] = (0.0, 0.0),
  velocity_range: Tuple[float, float] = (0.0, 0.0),
):
  jnts = robot.get_non_root_joints()
  qpos_ids = jp.array([model.bind(jnt).qposadr for jnt in jnts])
  qvel_ids = jp.array([model.bind(jnt).dofadr for jnt in jnts])
  limits = jp.stack([model.bind(jnt).range for jnt in jnts], axis=0)
  rng, key_pos, key_vel = jax.random.split(rng, 3)
  pos_noise = jax.random.uniform(
    key_pos, shape=(len(qpos_ids),), minval=position_range[0], maxval=position_range[1]
  )
  vel_noise = jax.random.uniform(
    key_vel, shape=(len(qvel_ids),), minval=velocity_range[0], maxval=velocity_range[1]
  )
  qpos = data.qpos.at[qpos_ids].set(data.qpos[qpos_ids] + pos_noise)
  qpos = qpos.at[qpos_ids].set(jp.clip(qpos[qpos_ids], limits[:, 0], limits[:, 1]))
  qvel = data.qvel.at[qvel_ids].set(data.qvel[qvel_ids] + vel_noise)
  return data.replace(qpos=qpos, qvel=qvel)


def reset_root_state(
  rng: jax.Array,
  robot: entity.Entity,
  model: mjx.Model,
  data: mjx.Data,
  pose_range: Dict[str, Tuple[float, float]],
  velocity_range: Dict[str, Tuple[float, float]],
):
  allowed_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
  range_list = [pose_range.get(key, (0.0, 0.0)) for key in allowed_keys]
  vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in allowed_keys]
  ranges = jp.vstack([jp.array(r) for r in range_list])
  vel_ranges = jp.vstack([jp.array(r) for r in vel_range_list])
  rng, key_pos, key_vel = jax.random.split(rng, 3)
  samples = jax.random.uniform(key_pos, (6,), minval=ranges[:, 0], maxval=ranges[:, 1])
  root_joint = robot.get_root_joint()
  if root_joint is None:
    raise ValueError("No root joint found.")
  qpos_ids = jp.array([model.bind(root_joint).qposadr]) + jp.arange(7)
  qvel_ids = jp.array([model.bind(root_joint).dofadr]) + jp.arange(6)
  qpos = data.qpos.at[qpos_ids[:3]].set(data.qpos[qpos_ids[:3]] + samples[:3])
  roll_q = math.axis_angle_to_quat(jp.array([1, 0, 0]), samples[3])
  pitch_q = math.axis_angle_to_quat(jp.array([0, 1, 0]), samples[4])
  yaw_q = math.axis_angle_to_quat(jp.array([0, 0, 1]), samples[5])
  delta_quat = math.quat_mul(yaw_q, math.quat_mul(pitch_q, roll_q))
  current_quat = qpos[qpos_ids[3:7]]
  new_quat = math.quat_mul(delta_quat, current_quat)
  qpos = qpos.at[qpos_ids[3:7]].set(new_quat)
  root_vel = jax.random.uniform(
    key_vel, (6,), minval=vel_ranges[:, 0], maxval=vel_ranges[:, 1]
  )
  qvel = data.qvel.at[qvel_ids].set(data.qvel[qvel_ids] + root_vel)
  return data.replace(qpos=qpos, qvel=qvel)
