from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple, Union, cast
import re

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mjlab._src.collision import geoms_colliding
from mjlab._src import MENAGERIE_PATH
from mjlab._src import entity, mjx_env, mjx_task
from mjlab._src.arenas import FlatTerrainArena

from mjlab.control_suite import g1_constants as consts
from mjlab._src.sim_structs import CollisionPair

_HERE = Path(__file__).parent
_G1_XML = _HERE / "g1.xml"

# Foot collision parameters.
_FOOT_SOLREF = (0.008, 1)
_FOOT_FRICTION = (0.6, 0.6)
_FOOT_CONDIM = 3


def get_assets() -> Dict[str, bytes]:
  assets: Dict[str, bytes] = {}
  path = MENAGERIE_PATH / "unitree_g1"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


def get_rz(
  phi: Union[jax.Array, float], swing_height: Union[jax.Array, float] = 0.08
) -> jax.Array:
  def cubic_bezier_interpolation(y_start, y_end, x):
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier

  x = (phi + jp.pi) / (2 * jp.pi)
  stance = cubic_bezier_interpolation(0, swing_height, 2 * x)
  swing = cubic_bezier_interpolation(swing_height, 0, 2 * x - 1)
  return jp.where(x <= 0.5, stance, swing)


class G1(entity.Entity):
  """G1 entity."""

  def __init__(self, spec: mujoco.MjSpec):
    super().__init__(spec)

    self.add_pd_actuators_from_patterns(consts.ACTUATOR_SPECS)

    for sensor in consts.SENSORS:
      self.add_sensor(sensor)

    for keyframe in consts.KEYFRAMES:
      self.add_keyframe(keyframe, ctrl=keyframe.joint_angles)

    for collision_pair in consts.SELF_COLLISIONS:
      self.add_collision_pair(collision_pair)

    foot_floor_pairs = []
    for side in ["left", "right"]:
      for i in range(1, 4):
        pair = CollisionPair(
          geom1=f"{side}_foot{i}_collision",
          geom2="floor",
          condim=_FOOT_CONDIM,
          friction=_FOOT_FRICTION,
          solref=_FOOT_SOLREF,
        )
        self.add_collision_pair(pair)
        foot_floor_pairs.append(pair.full_name())
    self._foot_floor_pairs = tuple(foot_floor_pairs)

    self._torso_body = self.spec.body(consts.TORSO_BODY)
    self._imu_site = self.spec.site(consts.PELVIS_IMU_SITE)
    self._joints = self.get_non_root_joints()
    self._ankle_joints = [
      j for j in self._joints if re.match(r".*_ankle_(pitch|roll)_joint", j.name)
    ]
    self._actuators = self.spec.actuators
    self._gyro_sensor = self.spec.sensor("gyro")
    self._local_linvel_sensor = self.spec.sensor("local_linvel")
    self._upvector_sensor = self.spec.sensor("upvector")

  @property
  def foot_floor_pairs(self) -> Tuple[str, ...]:
    return self._foot_floor_pairs

  @property
  def joints(self) -> Tuple[mujoco.MjsJoint]:
    return self._joints

  @property
  def ankle_joints(self) -> Tuple[mujoco.MjsJoint]:
    return self._ankle_joints

  @property
  def actuators(self) -> Tuple[mujoco.MjsActuator]:
    return self._actuators

  def joint_angles(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._joints).qpos

  def ankle_joint_angles(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._ankle_joints).qpos

  def joint_velocities(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._joints).qvel

  def joint_accelerations(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._joints).qacc

  def joint_torques(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._actuators).force

  def gyro(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._gyro_sensor).sensordata

  def local_linvel(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._local_linvel_sensor).sensordata

  def projected_gravity(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._imu_site).xmat.T @ jp.array([0, 0, -1.0])

  def upvector(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
    return data.bind(model, self._upvector_sensor).sensordata


@dataclass(frozen=True)
class CommandConfig:
  min: Tuple[float, float, float] = (-1.5, -0.8, -1.5)
  max: Tuple[float, float, float] = (1.5, 0.8, 1.5)


@dataclass(frozen=True)
class RewardScales:
  tracking_lin_vel: float = 1.0
  tracking_ang_vel: float = 0.5
  lin_vel_z: float = -0.5
  ang_vel_xy: float = -0.05
  flat_orientation: float = -5.0
  dof_pos_limits: float = -1.0
  pose: float = -0.5
  feet_phase: float = 1.0
  torques: float = -0.0002
  dof_acc: float = -2.5e-7
  action_rate: float = -0.01
  energy: float = -0.001
  collision: float = -1.0


@dataclass(frozen=True)
class RewardConfig:
  scales: RewardScales = RewardScales()
  max_foot_height: float = 0.1
  lin_vel_std: float = 0.5
  ang_vel_std: float = 0.5
  foot_height_std: float = 0.07


@dataclass(frozen=True)
class NoiseConfig:
  joint_pos: float = 0.03
  joint_vel: float = 1.5
  gyro: float = 0.2
  projected_gravity: float = 0.05
  linvel: float = 0.1


@dataclass(frozen=True)
class G1Config(mjx_task.TaskConfig):
  """G1 configuration."""

  sim_dt: float = 0.004
  ctrl_dt: float = 0.02
  solver_iterations: int = 5
  solver_ls_iterations: int = 8
  integrator: str = "implicitfast"
  euler_damping: bool = False
  max_episode_length: int = 1_000
  noise_level: float = 1.0
  action_scale: float = 0.5
  soft_joint_pos_limit_factor: float = 0.9
  command_config: CommandConfig = CommandConfig()
  reward_config: RewardConfig = RewardConfig()
  noise_config: NoiseConfig = NoiseConfig()


class G1Env(mjx_task.MjxTask[G1Config]):
  def __init__(self, config: G1Config = G1Config()):
    root, entities = G1Env.build_scene(config)
    super().__init__(config, root.spec, entities=entities)
    self.g1: G1 = cast(G1, entities["g1"])
    self._reward_scales = asdict(self.cfg.reward_config.scales)

    self._torso_body_id = self.model.body(consts.TORSO_BODY).id
    self._feet_site_id = np.array([self.model.site(n).id for n in consts.FEET_SITES])
    self._init_q = jp.array(self.model.keyframe("knees_bent").qpos)
    self._default_pose = jp.array(self.model.keyframe("knees_bent").qpos[7:])
    joint_ids = [j.id for j in self.g1.ankle_joints]
    lowers, uppers = self.model.jnt_range[joint_ids].T
    c = (lowers + uppers) / 2
    r = uppers - lowers
    self._soft_lowers = c - 0.5 * r * self.cfg.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self.cfg.soft_joint_pos_limit_factor
    self._collision_pairs = []
    for pair in consts.SELF_COLLISIONS:
      geom1_id = self.model.geom(pair.geom1).id
      geom2_id = self.model.geom(pair.geom2).id
      self._collision_pairs.append((geom1_id, geom2_id))

  @staticmethod
  def build_scene(config: G1Config) -> Tuple[entity.Entity, Dict[str, entity.Entity]]:
    del config  # Unused.
    assets = get_assets()
    g1_entity = G1.from_file(_G1_XML, assets=assets)

    arena = FlatTerrainArena()
    arena.add_skybox()

    arena.spec.stat.meansize = 0.03

    frame = arena.spec.worldbody.add_frame()
    arena.spec.attach(g1_entity.spec, prefix="", frame=frame)

    return arena, {"g1": g1_entity, "arena": arena}

  def domain_randomize(self, model: mjx.Model, rng: jax.Array) -> Tuple[mjx.Model, Any]:
    joint_qpos_ids = jp.array([model.bind(j).qposadr for j in self.g1.joints])
    collision_pair_ids = jp.array(
      [self.spec.pair(p).id for p in self.g1.foot_floor_pairs]
    )

    @jax.vmap
    def _randomize(rng):
      # Floor friction: *U(0.4, 1.0).
      rng, key = jax.random.split(rng)
      friction = jax.random.uniform(key, minval=0.6, maxval=1.0)
      pair_friction = model.pair_friction.at[collision_pair_ids, 0:2].set(friction)

      # Link masses: *U(0.9, 1.1).
      rng, key = jax.random.split(rng)
      dmass = jax.random.uniform(key, shape=(model.nbody,), minval=0.9, maxval=1.1)
      body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

      # Link center of masses: +U(-0.05, 0.05).
      rng, key = jax.random.split(rng)
      dpos = jax.random.uniform(key, shape=(model.nbody, 3), minval=-0.02, maxval=0.02)
      body_ipos = model.body_ipos.at[:].set(model.body_ipos + dpos)

      # Joint calibration offsets: +U(-0.05, 0.05).
      rng, key = jax.random.split(rng)
      dqpos = jax.random.uniform(
        key, shape=(len(joint_qpos_ids),), minval=-0.05, maxval=0.05
      )
      qpos0 = model.qpos0.at[joint_qpos_ids].set(model.qpos0[joint_qpos_ids] + dqpos)

      return pair_friction, body_mass, body_ipos, qpos0

    pair_friction, body_mass, body_ipos, qpos0 = _randomize(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
      {
        "pair_friction": 0,
        "body_mass": 0,
        "body_ipos": 0,
        "qpos0": 0,
      }
    )
    model = model.tree_replace(
      {
        "pair_friction": pair_friction,
        "body_mass": body_mass,
        "body_ipos": body_ipos,
        "qpos0": qpos0,
      }
    )
    return model, in_axes

  def before_step(self, action: jax.Array, state: mjx_env.State) -> mjx.Data:
    motor_targets = self._default_pose + action * self.cfg.action_scale
    return super().before_step(motor_targets, state)

  def initialize_episode(self, data: mjx.Data, rng: jax.Array):
    qpos = self._init_q
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)
    data = data.replace(qpos=qpos)
    rng, cmd_rng = jax.random.split(rng)
    command = self._sample_command(cmd_rng)
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.75)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0, jp.pi])
    info = {
      "rng": rng,
      "step": 0,
      "last_act": jp.zeros(self.mjx_model.nu),
      "command": command,
      "phase": phase,
      "phase_dt": phase_dt,
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._reward_scales.keys()}
    metrics["metrics/lin_vel_norm"] = jp.zeros(())
    metrics["metrics/ang_vel_norm"] = jp.zeros(())
    metrics["metrics/phase_norm"] = jp.zeros(())
    return data, info, metrics

  def _apply_noise(
    self, info: dict[str, Any], value: jax.Array, scale: float
  ) -> jax.Array:
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noise = 2 * jax.random.uniform(noise_rng, shape=value.shape) - 1
    noisy_value = value + noise * self.cfg.noise_level * scale
    return noisy_value

  def get_observation(self, data: mjx.Data, state: mjx_env.State):
    cfg: NoiseConfig = self.cfg.noise_config

    # Ground-truth observations.
    gyro = self.g1.gyro(self.mjx_model, data)
    projected_gravity = self.g1.projected_gravity(self.mjx_model, data)
    joint_angles = self.g1.joint_angles(self.mjx_model, data)
    joint_velocities = self.g1.joint_velocities(self.mjx_model, data)
    local_linvel = self.g1.local_linvel(self.mjx_model, data)

    # Noisy observations.
    noisy_gyro = self._apply_noise(state.info, gyro, cfg.gyro)
    noisy_projected_gravity = self._apply_noise(
      state.info, projected_gravity, cfg.projected_gravity
    )
    noisy_joint_angles = self._apply_noise(state.info, joint_angles, cfg.joint_pos)
    noisy_joint_velocities = self._apply_noise(
      state.info, joint_velocities, cfg.joint_vel
    )
    noisy_local_linvel = self._apply_noise(state.info, local_linvel, cfg.linvel)

    obs = jp.hstack(
      [
        noisy_gyro,
        noisy_projected_gravity,
        noisy_joint_angles,
        noisy_joint_velocities,
        noisy_local_linvel,
        state.info["last_act"],
        state.info["command"],
        jp.concatenate([jp.cos(state.info["phase"]), jp.sin(state.info["phase"])]),
      ]
    )
    privileged_obs = jp.hstack(
      [
        obs,
        gyro,
        projected_gravity,
        joint_angles,
        joint_velocities,
        local_linvel,
      ]
    )

    return {
      "state": obs,
      "privileged_state": privileged_obs,
    }

  def get_reward(self, data, state, action, done):
    del done  # Unused.
    local_linvel = self.g1.local_linvel(self.mjx_model, data)
    gyro = self.g1.gyro(self.mjx_model, data)
    projected_gravity = self.g1.projected_gravity(self.mjx_model, data)
    joint_torques = self.g1.joint_torques(self.mjx_model, data)
    joint_accelerations = self.g1.joint_accelerations(self.mjx_model, data)
    joint_angles = self.g1.joint_angles(self.mjx_model, data)
    joint_velocities = self.g1.joint_velocities(self.mjx_model, data)
    ankle_joint_angles = self.g1.ankle_joint_angles(self.mjx_model, data)
    reward_terms = {
      "tracking_lin_vel": self._reward_tracking_lin_vel(
        state.info["command"], local_linvel, state
      ),
      "tracking_ang_vel": self._reward_tracking_ang_vel(
        state.info["command"], gyro, state
      ),
      "lin_vel_z": self._cost_lin_vel_z(local_linvel),
      "ang_vel_xy": self._cost_ang_vel_xy(gyro),
      "flat_orientation": self._cost_flat_orientation(projected_gravity),
      "torques": self._cost_torques(joint_torques),
      "action_rate": self._cost_action_rate(action, state.info["last_act"]),
      "dof_acc": self._cost_dof_acc(joint_accelerations),
      "pose": self._cost_pose(joint_angles),
      "dof_pos_limits": self._cost_joint_pos_limits(ankle_joint_angles),
      "energy": self._cost_energy(joint_velocities, joint_torques),
      "feet_phase": self._reward_feet_phase(data, state),
      "collision": self._cost_collision(data),
    }
    rewards = {k: v * self._reward_scales[k] for k, v in reward_terms.items()}
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v / self.dt
    reward = sum(rewards.values()) * self.dt
    return reward

  def after_step(
    self,
    data: mjx.Data,
    state: mjx_env.State,
    action: jax.Array,
    done: jax.Array,
  ) -> mjx.Data:
    state.info["step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    state.info["command"] = jp.where(
      state.info["step"] > 500, self._sample_command(cmd_rng), state.info["command"]
    )
    state.info["step"] = jp.where(
      done | (state.info["step"] > 500), 0, state.info["step"]
    )
    return data

  def should_terminate_episode(self, data: mjx.Data, state: mjx_env.State) -> jax.Array:
    del state  # Unused.
    fall_termination = self.g1.upvector(self.mjx_model, data)[-1] < 0.0
    return fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

  def _sample_command(self, rng: jax.Array) -> jax.Array:
    cfg: CommandConfig = self.cfg.command_config
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    lin_vel_x = jax.random.uniform(rng1, minval=cfg.min[0], maxval=cfg.max[0])
    lin_vel_y = jax.random.uniform(rng2, minval=cfg.min[1], maxval=cfg.max[1])
    ang_vel_yaw = jax.random.uniform(rng3, minval=cfg.min[2], maxval=cfg.max[2])
    return jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw])

  # Reward functions.

  def _exp_reward(self, error: jax.Array, sigma: jax.Array) -> jax.Array:
    return jp.exp(-(error**2) / sigma**2)

  def _reward_tracking_lin_vel(
    self, commands: jax.Array, local_vel: jax.Array, state: mjx_env.State
  ) -> jax.Array:
    lin_vel_error = jp.linalg.norm(commands[:2] - local_vel[:2])
    state.metrics["metrics/lin_vel_norm"] = lin_vel_error
    return self._exp_reward(lin_vel_error, self.cfg.reward_config.lin_vel_std)

  def _reward_tracking_ang_vel(
    self,
    commands: jax.Array,
    ang_vel: jax.Array,
    state: mjx_env.State,
  ) -> jax.Array:
    ang_vel_error = commands[2] - ang_vel[2]
    state.metrics["metrics/ang_vel_norm"] = ang_vel_error
    return self._exp_reward(ang_vel_error, self.cfg.reward_config.ang_vel_std)

  def _cost_lin_vel_z(self, local_vel) -> jax.Array:
    return jp.square(local_vel[2])

  def _cost_ang_vel_xy(self, ang_vel) -> jax.Array:
    return jp.sum(jp.square(ang_vel[:2]))

  def _cost_flat_orientation(self, gravity: jax.Array) -> jax.Array:
    return jp.sum(jp.square(gravity[:2]))

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_energy(self, qvel: jax.Array, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(torques))

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
    return jp.sum(jp.square(act - last_act))

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - self._default_pose))

  def _cost_joint_pos_limits(self, ankle_qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(ankle_qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(ankle_qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_collision(self, data: mjx.Data) -> jax.Array:
    coll = []
    for geom1_id, geom2_id in self._collision_pairs:
      coll.append(geoms_colliding(data, geom1_id, geom2_id))
    return jp.any(jp.stack(coll))

  def _reward_feet_phase(self, data: mjx.Data, state: mjx_env.State) -> jax.Array:
    foot_z = data.site_xpos[self._feet_site_id][..., -1]
    rz = get_rz(
      state.info["phase"], swing_height=self.cfg.reward_config.max_foot_height
    )
    error = jp.mean(rz - foot_z)
    state.metrics["metrics/foot_height_norm"] = error
    return self._exp_reward(error, self.cfg.reward_config.foot_height_std)

  # Visualization.

  def visualize(self, state: mjx_env.State, scn):
    torso_pos = np.asarray(state.data.xpos[self._torso_body_id])
    torso_rot = np.asarray(state.data.xmat[self._torso_body_id])

    def local_to_world(vec: np.ndarray) -> np.ndarray:
      return torso_pos + torso_rot @ vec

    def make_arrow(
      from_local: np.ndarray, to_local: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
      return local_to_world(from_local), local_to_world(to_local)

    def add_arrow(from_w, to_w, rgba, width=0.015, size=(0.005, 0.02, 0.02)):
      scn.ngeom += 1
      geom = scn.geoms[scn.ngeom - 1]
      geom.category = mujoco.mjtCatBit.mjCAT_DECOR

      mujoco.mjv_initGeom(
        geom=geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW.value,
        size=np.array(size, dtype=np.float32),
        pos=np.zeros(3),
        mat=np.zeros(9),
        rgba=np.asarray(rgba, dtype=np.float32),
      )

      mujoco.mjv_connector(
        geom=geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW.value,
        width=width,
        from_=from_w,
        to=to_w,
      )

    scale = 0.75  # Scale the arrows to 75% of their original size.
    z_offset = 0.7  # Leave a 70cm gap between the base and the arrows.

    # Commanded velocities.
    cmd = state.info["command"]
    cmd_lin_from = np.array([0, 0, z_offset]) * scale
    cmd_lin_to = cmd_lin_from + np.array([cmd[0], cmd[1], 0]) * scale
    cmd_ang_from = cmd_lin_from
    cmd_ang_to = cmd_ang_from + np.array([0, 0, cmd[2]]) * scale
    add_arrow(*make_arrow(cmd_lin_from, cmd_lin_to), rgba=[0.2, 0.2, 0.6, 0.6])
    add_arrow(*make_arrow(cmd_ang_from, cmd_ang_to), rgba=[0.2, 0.6, 0.2, 0.6])

    # Actual velocities.
    linvel = self.g1.local_linvel(self.mjx_model, state.data)
    gyro = self.g1.gyro(self.mjx_model, state.data)
    act_lin_from = np.array([0, 0, z_offset]) * scale
    act_lin_to = act_lin_from + np.array([linvel[0], linvel[1], 0]) * scale
    act_ang_from = act_lin_from
    act_ang_to = act_ang_from + np.array([0, 0, gyro[2]]) * scale
    add_arrow(*make_arrow(act_lin_from, act_lin_to), rgba=[0.0, 0.6, 1.0, 0.7])
    add_arrow(*make_arrow(act_ang_from, act_ang_to), rgba=[0.0, 1.0, 0.4, 0.7])


if __name__ == "__main__":
  import mujoco.viewer
  import tyro

  def build_and_compile_and_launch(cfg: G1Config):
    root, _ = G1Env.build_scene(config=cfg)
    cfg.apply_defaults(root.spec)
    mujoco.viewer.launch(root.spec.compile())

  build_and_compile_and_launch(tyro.cli(G1Config))
